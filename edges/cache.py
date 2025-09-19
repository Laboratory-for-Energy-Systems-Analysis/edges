"""
Cache scaffolding for exchange-based LCIA mapping in `edges`.

Drop-in helpers to persist and reload mapping results between runs so that
`map_exchanges()` (and regional fallback steps) can be skipped or reduced
when nothing relevant changed.

Main ideas
----------
- Cache at the **logical** level: supplier/consumer activity IDs + signatures,
  not matrix positions.
- Validate cache with strong keys: method/params/weights/geo hashes and
  activity-side signatures (name/ref prod/location/classifications, etc.).
- Keep symbolic CF expressions in the cache; evaluate separately and optionally
  cache numerical evaluations per scenario index.

How to use
----------
1) Instantiate a backend once (e.g., ParquetCacheBackend) and wire into EdgeLCIA.
2) At the beginning of mapping, call `MappingCache.load_for_context(...)` to
   get previously matched pairs and pre-populate `cfs_mapping`.
3) After mapping phases, call `MappingCache.save_new_matches(...)` to persist.
4) Optionally use `MethodCache` to persist CF index / prefix needs.

This module is dependency-light (pandas/pyarrow optional; falls back to CSV).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import typing as t
import time
import shutil
import numpy as np

import platformdirs

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Default cache directory using platformdirs
CACHE_DIR = platformdirs.user_data_path(appname="edges", appauthor="psi-lea")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_obj(obj: t.Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _norm_none(x):
    return None if x in ("", "__NONE__") else x


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class CacheBackend:
    """Abstract-ish persistence backend.

    Implement two small tables:
      - method blobs (key -> bytes)
      - mapping rows (tabular)
      - eval rows (tabular; evaluated numeric values for exprs)
    """

    def __init__(self, root: t.Union[str, Path] = CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "method").mkdir(exist_ok=True)
        (self.root / "map").mkdir(exist_ok=True)
        (self.root / "eval").mkdir(exist_ok=True)

    @classmethod
    def default(cls) -> "CacheBackend":
        """Return a backend pointing at the default platformdirs cache location."""
        return cls(CACHE_DIR)

    def clear(self) -> None:
        """Delete all cached data (method, map, eval)."""
        if self.root.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "method").mkdir(exist_ok=True)
            (self.root / "map").mkdir(exist_ok=True)
            (self.root / "eval").mkdir(exist_ok=True)

    # ------------- method blobs -------------
    def put_method_blob(self, key: str, payload: dict) -> None:
        path = self.root / "method" / f"{key}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def get_method_blob(self, key: str) -> t.Optional[dict]:
        path = self.root / "method" / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------- mapping table -------------
    def _map_path(self, context_key: str, parquet: bool) -> Path:
        ext = "parquet" if parquet else "csv"
        return self.root / "map" / f"{context_key}.{ext}"

    def read_mapping_rows(self, context_key: str) -> list[dict]:
        """Return a list of mapping rows for a given context key.

        Falls back to CSV if Parquet/unavailable.
        """
        parquet = pd is not None and hasattr(pd, "read_parquet")
        p = self._map_path(context_key, parquet)
        if not p.exists():
            # Try alternate extension
            p_alt = self._map_path(context_key, not parquet)
            if not p_alt.exists():
                return []
            p = p_alt
            parquet = not parquet

        if pd is None:
            # very small CSV reader
            rows = []
            with p.open("r", encoding="utf-8") as f:
                header = None
                for i, line in enumerate(f):
                    parts = [c.strip() for c in line.rstrip("\n").split(",")]
                    if i == 0:
                        header = parts
                        continue
                    rows.append(dict(zip(header, parts)))
            return rows

        if parquet and p.suffix == ".parquet":
            try:
                df = pd.read_parquet(p)
            except Exception:
                return []
        else:
            df = pd.read_csv(p)
        return df.to_dict(orient="records")

    def write_mapping_rows(self, context_key: str, rows: list[dict]) -> None:
        if not rows:
            return
        parquet = pd is not None and hasattr(pd, "DataFrame")
        p = self._map_path(context_key, parquet)
        p.parent.mkdir(exist_ok=True, parents=True)

        if pd is None:
            # naive CSV writer
            keys = list(rows[0].keys())
            with p.open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
            return

        df = pd.DataFrame(rows)
        if p.suffix == ".parquet":
            try:
                df.to_parquet(p, index=False)
            except Exception:
                # fallback to CSV
                p = self._map_path(context_key, parquet=False)
                df.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)

    # ------------- eval table -------------
    def _eval_path(self, eval_key: str, parquet: bool) -> Path:
        ext = "parquet" if parquet else "csv"
        return self.root / "eval" / f"{eval_key}.{ext}"

    def upsert_eval_rows(self, eval_key: str, rows: list[dict]) -> None:
        if not rows:
            return
        existing = self.read_eval_rows(eval_key)
        merged = {(r["cf_hash"], r["scenario_idx"]): r for r in existing}
        for r in rows:
            merged[(r["cf_hash"], r["scenario_idx"])] = r
        out = list(merged.values())
        parquet = pd is not None
        p = self._eval_path(eval_key, parquet)
        if pd is None:
            keys = list(out[0].keys())
            with p.open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for r in out:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
            return
        df = pd.DataFrame(out)
        if p.suffix == ".parquet":
            try:
                df.to_parquet(p, index=False)
            except Exception:
                p = self._eval_path(eval_key, parquet=False)
                df.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)

    def read_eval_rows(self, eval_key: str) -> list[dict]:
        parquet = pd is not None
        p = self._eval_path(eval_key, parquet)
        if not p.exists():
            p = self._eval_path(eval_key, not parquet)
            if not p.exists():
                return []
        if pd is None:
            rows = []
            with p.open("r", encoding="utf-8") as f:
                header = None
                for i, line in enumerate(f):
                    parts = [c.strip() for c in line.rstrip("\n").split(",")]
                    if i == 0:
                        header = parts
                        continue
                    rows.append(dict(zip(header, parts)))
            return rows
        if p.suffix == ".parquet":
            try:
                df = pd.read_parquet(p)
            except Exception:
                return []
        else:
            df = pd.read_csv(p)
        return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Data rows
# ---------------------------------------------------------------------------


@dataclass
class MappingRow:
    # identity
    supplier_act_id: str  # bw2data key/ID
    consumer_act_id: str
    direction: str  # "biosphere-technosphere" | "technosphere-technosphere"

    # validation signatures
    supplier_sig: str  # sha256 of normalized supplier info
    consumer_sig: str  # sha256 of normalized consumer info

    # source provenance
    method_hash: str
    params_hash: str
    weights_hash: str
    geo_hash: str

    # CF
    value_expr: t.Optional[str]
    value_numeric: t.Optional[float]
    uncertainty_json: t.Optional[str]

    # optional diagnosis (which CF entry produced it)
    matched_cf_hash: t.Optional[str] = None

    # housekeeping
    created_ts: float = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class EvalRow:
    cf_hash: str  # hash of the CF expression/definition
    method_hash: str
    params_hash: str
    scenario_idx: t.Union[int, str]
    value_numeric: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Higher-level caches
# ---------------------------------------------------------------------------


class MethodCache:
    """Persist method-level computed structures (cf_index, prefix sets, fields).
    Caller is responsible to ensure objects are JSON-serializable or converted.
    """

    def __init__(self, backend: CacheBackend):
        self.backend = backend

    def load(self, method_hash: str) -> t.Optional[dict]:
        return self.backend.get_method_blob(method_hash)

    def save(self, method_hash: str, payload: dict) -> None:
        self.backend.put_method_blob(method_hash, payload)


class MappingCache:
    """Load and save logical mapping rows for a specific context.

    Context key composition: project/db -> method_hash -> weights_hash -> geo_hash -> params_hash
    The exact composition is delegated to the caller; we just treat it as an opaque string.
    """

    REQUIRED_COLS = [
        "supplier_act_id",
        "consumer_act_id",
        "direction",
        "supplier_sig",
        "consumer_sig",
        "method_hash",
        "params_hash",
        "weights_hash",
        "geo_hash",
        "value_expr",
        "value_numeric",
        "uncertainty_json",
        "matched_cf_hash",
        "created_ts",
    ]

    def __init__(self, backend: CacheBackend):
        self.backend = backend

    # ---------- key utilities ----------
    @staticmethod
    def make_context_key(
        project: str,
        method_hash: str,
        weights_hash: str,
        geo_hash: str,
        params_hash: str,
    ) -> str:
        """Return a fixed-length, filesystem-safe key for this context.

        We hash the structured payload so filenames stay small (<255 chars) on all OSes.
        """
        payload = {
            "project": project,
            "method": method_hash,
            "weights": weights_hash,
            "geo": geo_hash,
            "params": params_hash,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # ---------- load ----------
    def load_for_context(self, context_key: str) -> list[MappingRow]:
        rows = self.backend.read_mapping_rows(context_key)
        out: list[MappingRow] = []
        for r in rows:
            # Normalize and coerce types
            try:
                out.append(
                    MappingRow(
                        supplier_act_id=str(r["supplier_act_id"]),
                        consumer_act_id=str(r["consumer_act_id"]),
                        direction=str(r["direction"]),
                        supplier_sig=str(r["supplier_sig"]),
                        consumer_sig=str(r["consumer_sig"]),
                        method_hash=str(r["method_hash"]),
                        params_hash=str(r["params_hash"]),
                        weights_hash=str(r["weights_hash"]),
                        geo_hash=str(r["geo_hash"]),
                        value_expr=_norm_none(r.get("value_expr")),
                        value_numeric=(
                            None if r.get("value_numeric") in (None, "") else float(r["value_numeric"])  # type: ignore
                        ),
                        uncertainty_json=_norm_none(r.get("uncertainty_json")),
                        matched_cf_hash=_norm_none(r.get("matched_cf_hash")),
                        created_ts=float(r.get("created_ts", time.time())),
                    )
                )
            except Exception:
                # skip malformed rows silently
                continue
        return out

    # ---------- save ----------
    def save_for_context(self, context_key: str, rows: list[MappingRow]) -> None:
        self.backend.write_mapping_rows(context_key, [r.to_dict() for r in rows])

    def append_mapping_rows(
        self, context_key: str, new_rows: list[MappingRow | dict]
    ) -> int:
        # read existing (dicts)
        existing = self.backend.read_mapping_rows(context_key)
        # normalize new rows to dicts
        to_add = [r.to_dict() if hasattr(r, "to_dict") else r for r in new_rows]
        # append and write
        existing.extend(to_add)
        self.backend.write_mapping_rows(context_key, existing)
        return len(to_add)


# ---------------------------------------------------------------------------
# Glue helpers for EdgeLCIA
# ---------------------------------------------------------------------------


class EdgeCacheGlue:
    """Small adapter with pure functions that EdgeLCIA can call.

    This avoids importing the full EdgeLCIA in this file. Integrate by
    creating an instance and calling the methods at the right steps.
    """

    def __init__(self, mapping_cache: MappingCache):
        self.mapping_cache = mapping_cache

    # --- Hash builders bound to an EdgeLCIA instance ---
    @staticmethod
    def method_hash(edge) -> str:
        return _sha256_bytes(edge.filepath.read_bytes())

    @staticmethod
    def params_hash(edge) -> str:
        payload = {"scenario": edge.scenario, "parameters": edge.parameters}
        return _sha256_obj(payload)

    @staticmethod
    def weights_hash(edge) -> str:
        """Hashable, JSON-serializable view of edge.weights.

        edge.weights is a dict with tuple keys: {(supplier_loc, consumer_loc): weight}
        JSON requires string keys, so we convert to a sorted list of triples.
        """
        w = edge.weights or {}
        if isinstance(w, dict):
            triples = []
            for k, v in w.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    s, c = k
                else:
                    # fallback: treat as single key; keep behavior stable
                    s, c = k, "__ANY__"
                triples.append([str(s), str(c), float(v)])
            triples.sort()  # deterministic
            payload = {"pairs": triples}
        else:
            payload = w  # unexpected type; still hashable via default=str

        return _sha256_obj(payload)

    @staticmethod
    def geo_hash(edge) -> str:
        # geo should expose a stable payload if possible; fallback to hash of weights + class name
        payload = getattr(edge.geo, "version_payload", None)
        if callable(payload):
            return _sha256_obj(payload())
        return _sha256_obj({"klass": edge.geo.__class__.__name__})

    @staticmethod
    def _supplier_sig_from_idx(edge, idx: int, direction: str) -> str:
        rev = (
            edge.reversed_supplier_lookup_bio
            if direction == "biosphere-technosphere"
            else edge.reversed_supplier_lookup_tech
        )
        info = rev.get(idx, {})
        # reuse the existing normalization
        from .edgelcia import _equality_supplier_signature_cached, make_hashable  # type: ignore

        return _sha256_obj(
            tuple(_equality_supplier_signature_cached(make_hashable(info)))
        )

    @staticmethod
    def _consumer_sig_from_idx(edge, idx: int) -> str:
        info = edge._get_consumer_info(idx)  # attaches classifications if missing
        # Normalize similar to supplier path
        from .edgelcia import make_hashable  # type: ignore

        norm = {
            k: info.get(k)
            for k in (
                "name",
                "reference product",
                "unit",
                "location",
                "classifications",
            )
            if k in info
        }
        return _sha256_obj(tuple(make_hashable(norm)))

    @staticmethod
    def _supplier_act_id(edge, i: int, direction: str) -> str:
        if direction == "biosphere-technosphere":
            return str(edge.reversed_biosphere[i])
        return str(edge.reversed_activity[i])

    @staticmethod
    def _consumer_act_id(edge, j: int) -> str:
        return str(edge.reversed_activity[j])

    # ---------- PRELOAD: turn cached rows into add_cf_entry calls ----------
    def preload_from_cache(self, edge) -> int:
        """Attempt to short-circuit mapping using cached rows.

        Returns the number of (i,j) positions that were restored.
        """
        ctx_key = MappingCache.make_context_key(
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
            self.params_hash(edge),
        )
        rows = self.mapping_cache.load_for_context(ctx_key)
        if not rows:
            return 0

        from .edgelcia import add_cf_entry  # type: ignore

        restored = 0
        for r in rows:
            try:
                direction = r.direction

                # Resolve current (i, j) indices from cached act IDs
                if direction == "biosphere-technosphere":
                    if not hasattr(edge, "_bio_id_to_row"):
                        edge._bio_id_to_row = {
                            str(v): k for k, v in edge.reversed_biosphere.items()
                        }
                    i = edge._bio_id_to_row.get(str(r.supplier_act_id))
                else:
                    if not hasattr(edge, "_act_id_to_row"):
                        edge._act_id_to_row = {
                            str(v): k for k, v in edge.reversed_activity.items()
                        }
                    i = edge._act_id_to_row.get(str(r.supplier_act_id))

                if not hasattr(edge, "_act_id_to_col"):
                    edge._act_id_to_col = {
                        str(v): k for k, v in edge.reversed_activity.items()
                    }
                j = edge._act_id_to_col.get(str(r.consumer_act_id))

                # Missing in current inventory -> skip
                if i is None or j is None:
                    continue

                # Validate signatures against CURRENT metadata
                s_sig_now = self._supplier_sig_from_idx(edge, i, direction)
                c_sig_now = self._consumer_sig_from_idx(edge, j)
                if s_sig_now != r.supplier_sig or c_sig_now != r.consumer_sig:
                    continue

                supplier_info = (
                    edge.reversed_supplier_lookup_bio.get(i, {})
                    if direction == "biosphere-technosphere"
                    else edge.reversed_supplier_lookup_tech.get(i, {})
                )
                consumer_info = edge._get_consumer_info(j)

                val = (
                    r.value_numeric
                    if r.value_numeric is not None
                    else (r.value_expr or 0)
                )
                unc = json.loads(r.uncertainty_json) if r.uncertainty_json else None

                add_cf_entry(
                    cfs_mapping=edge.cfs_mapping,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    direction=direction,
                    indices=[(i, j)],
                    value=val,
                    uncertainty=unc,
                )
                restored += 1
            except Exception:
                # If anything goes wrong for a row, skip it and keep going
                continue

        if restored:
            edge._update_unprocessed_edges()
        return restored

    def save_new_matches(self, edge, start_len: int) -> int:
        """Persist rows in `edge.cfs_mapping` beyond `start_len`.
        Returns number of rows written by the cache backend.
        """
        ctx_key = MappingCache.make_context_key(
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
            self.params_hash(edge),
        )

        new = edge.cfs_mapping[start_len:]
        if not new:
            return 0

        rows: list[dict] = []
        for cf in new:
            direction = cf["direction"]
            val = cf.get("value")
            value_numeric = (
                float(val) if isinstance(val, (int, float, np.floating)) else None
            )
            value_expr = (
                None
                if value_numeric is not None
                else (str(val) if val is not None else None)
            )
            uncertainty_json = (
                json.dumps(cf.get("uncertainty"), sort_keys=True)
                if cf.get("uncertainty") is not None
                else None
            )

            for i, j in cf["positions"]:
                rows.append(
                    {
                        "direction": direction,
                        "supplier_act_id": self._supplier_act_id(edge, i, direction),
                        "consumer_act_id": self._consumer_act_id(edge, j),
                        "supplier_sig": self._supplier_sig_from_idx(edge, i, direction),
                        "consumer_sig": self._consumer_sig_from_idx(edge, j),
                        "value_numeric": value_numeric,
                        "value_expr": value_expr,
                        "uncertainty_json": uncertainty_json,
                    }
                )

        # ⬇️ append via cache and return the backend’s count
        written = self.mapping_cache.append_mapping_rows(ctx_key, rows)
        return int(written or 0)
