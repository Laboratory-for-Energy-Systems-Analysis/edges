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

from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
import typing as t
import time
import shutil
import numpy as np
import math

import platformdirs

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

PARQUET_AVAILABLE = False
if pd is not None:
    try:
        import pyarrow  # noqa: F401

        PARQUET_AVAILABLE = True
    except Exception:
        try:
            import fastparquet  # noqa: F401

            PARQUET_AVAILABLE = True
        except Exception:
            PARQUET_AVAILABLE = False

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
import logging

log = logging.getLogger("edges.cache")


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
        (self.root / "neg").mkdir(exist_ok=True)
        (self.root / "allow").mkdir(exist_ok=True)

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
        parquet = PARQUET_AVAILABLE
        p = self._map_path(context_key, parquet)
        if not p.exists():
            p_alt = self._map_path(context_key, not parquet)
            if not p_alt.exists():
                log.info(
                    "CACHE READ: no file for ctx=%s (looked for %s and %s)",
                    context_key,
                    p,
                    p_alt,
                )
                return []
            p, parquet = p_alt, (not parquet)

        log.info(
            "CACHE READ: using path=%s (parquet=%s) size=%s bytes",
            p,
            parquet,
            p.stat().st_size,
        )

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
            log.info("CACHE READ: CSV rows=%d", len(rows))
            return rows

        if parquet and p.suffix == ".parquet":
            try:
                df = pd.read_parquet(
                    p
                )  # engine will be present if PARQUET_AVAILABLE is True
            except Exception as e:
                log.exception(
                    "CACHE READ: failed reading %s (%s). Returning [].",
                    p,
                    type(e).__name__,
                )
                # fall back to CSV sibling if present
                p_alt = p.with_suffix(".csv")
                if p_alt.exists():
                    df = pd.read_csv(p_alt)
                else:
                    # Optional: log a warning here
                    return []
        else:
            df = pd.read_csv(p)

        log.info(
            "CACHE READ: df shape=%s cols=%s dtypes=%s",
            df.shape,
            list(df.columns),
            dict(df.dtypes.astype(str)),
        )
        key_cols = [
            "direction",
            "supplier_act_id",
            "consumer_act_id",
            "supplier_sig",
            "consumer_sig",
        ]
        before = len(df)
        # keep the last row (newest) for any duplicates
        present = [c for c in key_cols if c in df.columns]
        if len(present) == len(key_cols):
            df = df.drop_duplicates(subset=key_cols, keep="last")
            logging.getLogger("edges.cache").info(
                "CACHE READ DEDUP: before=%d after=%d dropped=%d",
                before,
                len(df),
                before - len(df),
            )
        return df.to_dict(orient="records")

    def write_mapping_rows(self, context_key: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = self._map_path(context_key, PARQUET_AVAILABLE)
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
        if PARQUET_AVAILABLE:
            try:
                df.to_parquet(p, index=False)
                return
            except Exception:
                # if somehow this still fails, fall through to CSV
                pass

        # CSV fallback
        p = self._map_path(context_key, parquet=False)
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

    def _neg_path(self, context_key: str, parquet: bool) -> Path:
        ext = "parquet" if parquet else "csv"
        return self.root / "neg" / f"{context_key}.{ext}"

    def read_negative_rows(self, context_key: str) -> list[dict]:
        parquet = pd is not None
        p = self._neg_path(context_key, parquet)
        if not p.exists():
            p = self._neg_path(context_key, not parquet)
            if not p.exists():
                return []
        if pd is None:
            rows, header = [], None
            with p.open("r", encoding="utf-8") as f:
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

    def write_negative_rows(self, context_key: str, rows: list[dict]) -> None:
        if not rows:
            return
        parquet = pd is not None
        p = self._neg_path(context_key, parquet)
        p.parent.mkdir(parents=True, exist_ok=True)
        if pd is None:
            keys = list(rows[0].keys())
            with p.open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
            return
        df = pd.DataFrame(rows)
        # de-dup by (direction, supplier_act_id, consumer_act_id)
        df = df.drop_duplicates(
            subset=["direction", "supplier_act_id", "consumer_act_id"]
        )
        if p.suffix == ".parquet":
            try:
                df.to_parquet(p, index=False)
            except Exception:
                p = self._neg_path(context_key, parquet=False)
                df.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)

    def _allow_path(self, context_key: str, parquet: bool) -> Path:
        ext = "parquet" if (pd is not None and parquet) else "csv"
        return self.root / "allow" / f"{context_key}.{ext}"

    def read_allow_rows(self, context_key: str) -> list[dict]:
        parquet = pd is not None
        p = self._allow_path(context_key, parquet)
        if not p.exists():
            p_alt = self._allow_path(context_key, not parquet)
            if not p_alt.exists():
                return []
            p = p_alt
        if pd is None:
            rows, header = [], None
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = [c.strip() for c in line.rstrip("\n").split(",")]
                    if i == 0:
                        header = parts
                        continue
                    rows.append(dict(zip(header, parts)))
            return rows
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return df.to_dict(orient="records")

    def write_allow_rows(self, context_key: str, rows: list[dict]) -> None:
        if not rows:
            return
        parquet = pd is not None
        p = self._allow_path(context_key, parquet)
        p.parent.mkdir(parents=True, exist_ok=True)
        if pd is None:
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
                p = self._allow_path(context_key, parquet=False)
                df.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)


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
    created_ts: float = field(default_factory=time.time)

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

        def _none_if_nan(x):
            # treat pandas NA/NaN like None
            try:
                if x is None:
                    return None
                if isinstance(x, float) and math.isnan(x):
                    return None
                # pandas NA types sometimes stringify to 'NA'
                if x == "" or str(x).strip().upper() in {"NA", "NAN", "NONE"}:
                    return None
            except Exception:
                pass
            return x

        raw = self.backend.read_mapping_rows(context_key)
        out: list[MappingRow] = []
        skipped = 0
        sample_exc = None

        for r in raw:
            try:
                out.append(
                    MappingRow(
                        supplier_act_id=str(r["supplier_act_id"]),
                        consumer_act_id=str(r["consumer_act_id"]),
                        direction=str(r["direction"]),
                        supplier_sig=str(r["supplier_sig"]),
                        consumer_sig=str(r["consumer_sig"]),
                        method_hash=str(r.get("method_hash", "")),
                        params_hash=str(r.get("params_hash", "")),
                        weights_hash=str(r.get("weights_hash", "")),
                        geo_hash=str(r.get("geo_hash", "")),
                        value_expr=_none_if_nan(r.get("value_expr")),
                        value_numeric=(
                            None
                            if _none_if_nan(r.get("value_numeric")) is None
                            else float(r.get("value_numeric"))
                        ),
                        uncertainty_json=_none_if_nan(r.get("uncertainty_json")),
                        matched_cf_hash=_none_if_nan(r.get("matched_cf_hash")),
                        created_ts=float(
                            _none_if_nan(r.get("created_ts")) or time.time()
                        ),
                    )
                )
            except Exception as e:
                skipped += 1
                if sample_exc is None:
                    sample_exc = (e, dict(r))
                continue

        if skipped:
            log.warning(
                "CACHE LOAD COERCE: kept=%d skipped=%d (example error: %s on row sample keys=%s)",
                len(out),
                skipped,
                type(sample_exc[0]).__name__ if sample_exc else "n/a",
                list(sample_exc[1].keys()) if sample_exc else [],
            )
        else:
            log.info("CACHE LOAD COERCE: kept=%d skipped=0", len(out))

        return out

    # ---------- save ----------
    def save_for_context(self, context_key: str, rows: list[MappingRow]) -> None:
        self.backend.write_mapping_rows(context_key, [r.to_dict() for r in rows])

    def append_mapping_rows(
        self, context_key: str, new_rows: list[MappingRow | dict]
    ) -> int:
        # read existing (dicts)
        existing = self.backend.read_mapping_rows(context_key)
        to_add = [r.to_dict() if hasattr(r, "to_dict") else r for r in new_rows]

        key = lambda r: (
            r["direction"],
            r["supplier_act_id"],
            r["consumer_act_id"],
            r["supplier_sig"],
            r["consumer_sig"],
        )

        seen = {key(r) for r in existing}
        deduped = []
        for r in to_add:
            k = key(r)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(r)

        merged = existing + deduped
        self.backend.write_mapping_rows(context_key, merged)
        return len(deduped)

    def load_negative(self, context_key: str) -> list[dict]:
        return self.backend.read_negative_rows(context_key)

    def append_negative_rows(self, context_key: str, new_rows: list[dict]) -> int:
        existing = self.backend.read_negative_rows(context_key)
        existing.extend(new_rows)
        self.backend.write_negative_rows(context_key, existing)
        return len(new_rows)

    def load_allow(self, context_key: str, stage: str) -> list[dict]:
        return self.backend.read_allow_rows(f"{context_key}__{stage}")

    def save_allow(self, context_key: str, stage: str, rows: list[dict]) -> int:
        self.backend.write_allow_rows(f"{context_key}__{stage}", rows)
        return len(rows)


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

        edge.logger.info(
            "CACHE CTX: project=%s | method=%s | params=%s | weights=%s | geo=%s",
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.params_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
        )
        ctx_key = MappingCache.make_context_key(
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
            self.params_hash(edge),
        )

        p_parq = self.mapping_cache.backend._map_path(ctx_key, PARQUET_AVAILABLE)
        p_csv = self.mapping_cache.backend._map_path(ctx_key, parquet=False)
        edge.logger.info(
            "CACHE PRELOAD: probe exists parq=%s (%s) csv=%s (%s)",
            p_parq,
            p_parq.exists(),
            p_csv,
            p_csv.exists(),
        )

        edge.logger.info("CACHE CTX_KEY (64hex) = %s (len=%d)", ctx_key, len(ctx_key))
        edge.logger.info("CACHE BACKEND = %s", type(self.mapping_cache).__name__)
        edge.logger.info("CACHE BACKEND REPR = %r", self.mapping_cache)
        root_path = getattr(self.mapping_cache.backend, "root", None)
        edge.logger.info("CACHE BACKEND ROOT = %r", root_path)

        edge.logger.info(
            "CACHE CTX: project=%s | method=%s | params=%s | weights=%s | geo=%s",
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge)[:12],
            self.params_hash(edge)[:12],
            self.weights_hash(edge)[:12],
            self.geo_hash(edge)[:12],
        )
        edge.logger.info("CACHE PRELOAD: ctx_key=%s", ctx_key)
        try:
            edge.logger.info(
                "CACHE PRELOAD: backing_path=%s",
                getattr(self.mapping_cache, "backing_path", None),
            )
        except Exception:
            pass

        rows = self.mapping_cache.load_for_context(ctx_key)
        edge.logger.info(
            "CACHE PRELOAD: ctx_key=%s | loaded_rows=%d",
            ctx_key[:16],
            len(rows) if rows else 0,
        )
        if not rows:
            return 0

        from .edgelcia import add_cf_entry  # type: ignore

        restored = 0
        skipped_no_index = 0  # no (i,j) mapping in current inventory
        skipped_sig_mismatch = 0
        skipped_other = 0
        seen_positions = set()

        for r in rows:
            try:
                direction = r.direction
                # map supplier index
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

                if i is None or j is None:
                    skipped_no_index += 1
                    continue

                s_sig_now = self._supplier_sig_from_idx(edge, i, direction)
                c_sig_now = self._consumer_sig_from_idx(edge, j)
                if s_sig_now != r.supplier_sig or c_sig_now != r.consumer_sig:
                    skipped_sig_mismatch += 1
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

                pos = (i, j)
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)

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
                skipped_other += 1
                continue

        edge.logger.info(
            "CACHE PRELOAD SUMMARY: restored=%d | skipped_no_index=%d | skipped_sig_mismatch=%d | skipped_other=%d",
            restored,
            skipped_no_index,
            skipped_sig_mismatch,
            skipped_other,
        )

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

        edge.logger.info(
            "CACHE SAVE CTX: project=%s | method=%s | params=%s | weights=%s | geo=%s | new_entries=%d",
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge)[:12],
            self.params_hash(edge)[:12],
            self.weights_hash(edge)[:12],
            self.geo_hash(edge)[:12],
            sum(len(cf["positions"]) for cf in edge.cfs_mapping[start_len:]),
        )
        edge.logger.info("CACHE SAVE: ctx_key=%s", ctx_key)

        edge.logger.info(
            "CACHE CTX: project=%s | method=%s | params=%s | weights=%s | geo=%s",
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.params_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
        )
        ctx_key = MappingCache.make_context_key(
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
            self.params_hash(edge),
        )
        edge.logger.info("CACHE CTX_KEY (64hex) = %s (len=%d)", ctx_key, len(ctx_key))
        edge.logger.info("CACHE BACKEND = %s", type(self.mapping_cache).__name__)
        edge.logger.info("CACHE BACKEND REPR = %r", self.mapping_cache)
        root_path = getattr(self.mapping_cache.backend, "root", None)
        edge.logger.info("CACHE BACKEND ROOT = %r", root_path)

        try:
            edge.logger.info(
                "CACHE SAVE: backing_path=%s",
                getattr(self.mapping_cache, "backing_path", None),
            )
        except Exception:
            pass

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

    def _ctx_key(self, edge) -> str:
        return MappingCache.make_context_key(
            getattr(edge.lca, "project", "bw2"),
            self.method_hash(edge),
            self.weights_hash(edge),
            self.geo_hash(edge),
            self.params_hash(edge),
        )

    def preload_negative(self, edge) -> dict[tuple[str, str, str], tuple[str, str]]:
        """Load negative cache → {(direction, supplier_act_id, consumer_act_id): (supplier_sig, consumer_sig)}"""
        ctx_key = self._ctx_key(edge)
        rows = self.mapping_cache.load_negative(ctx_key)
        out = {}
        for r in rows:
            try:
                k = (
                    str(r["direction"]),
                    str(r["supplier_act_id"]),
                    str(r["consumer_act_id"]),
                )
                v = (str(r["supplier_sig"]), str(r["consumer_sig"]))
                out[k] = v
            except Exception:
                continue
        return out

    def save_negative(self, edge, misses: list[tuple[str, int, int]]) -> int:
        """Persist non-location misses as negatives.
        `misses` elements are (direction, i, j) using CURRENT matrix indices.
        """
        if not misses:
            return 0
        ctx_key = self._ctx_key(edge)
        rows = []
        for direction, i, j in misses:
            s_id = self._supplier_act_id(edge, i, direction)
            c_id = self._consumer_act_id(edge, j)
            s_sig = self._supplier_sig_from_idx(edge, i, direction)
            c_sig = self._consumer_sig_from_idx(edge, j)
            rows.append(
                {
                    "direction": direction,
                    "supplier_act_id": s_id,
                    "consumer_act_id": c_id,
                    "supplier_sig": s_sig,
                    "consumer_sig": c_sig,
                }
            )
        return self.mapping_cache.append_negative_rows(ctx_key, rows)

    def preload_allow(
        self, edge, stage: str
    ) -> dict[tuple[str, str, str], tuple[str, str]]:
        ctx_key = self._ctx_key(edge)
        rows = self.mapping_cache.load_allow(ctx_key, stage)
        return {
            (r["direction"], str(r["supplier_act_id"]), str(r["consumer_act_id"])): (
                str(r["supplier_sig"]),
                str(r["consumer_sig"]),
            )
            for r in rows
        }

    def save_allow(self, edge, stage: str, entries: list[tuple[str, int, int]]) -> int:
        if not entries:
            return 0
        ctx_key = self._ctx_key(edge)
        out = []
        for direction, i, j in entries:
            out.append(
                {
                    "direction": direction,
                    "supplier_act_id": self._supplier_act_id(edge, i, direction),
                    "consumer_act_id": self._consumer_act_id(edge, j),
                    "supplier_sig": self._supplier_sig_from_idx(edge, i, direction),
                    "consumer_sig": self._consumer_sig_from_idx(edge, j),
                }
            )
        return self.mapping_cache.save_allow(ctx_key, stage, out)
