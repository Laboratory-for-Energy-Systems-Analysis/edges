"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import math
from collections import defaultdict
import json
from typing import Optional
from pathlib import Path
import bw2calc
import numpy as np
import sparse
import pandas as pd
from prettytable import PrettyTable
import bw2data
from tqdm import tqdm
from textwrap import fill
from functools import lru_cache


from .utils import (
    format_data,
    get_flow_matrix_positions,
    safe_eval_cached,
    validate_parameter_lengths,
    make_hashable,
    assert_no_nans_in_cf_list,
)
from .matrix_builders import initialize_lcia_matrix, build_technosphere_edges_matrix
from .flow_matching import (
    preprocess_cfs,
    matches_classifications,
    normalize_classification_entries,
    build_cf_index,
    cached_match_with_index,
    preprocess_flows,
    build_index,
    compute_cf_memoized_factory,
    resolve_candidate_locations,
    group_edges_by_signature,
    compute_average_cf,
)
from .georesolver import GeoResolver
from .uncertainty import sample_cf_distribution, make_distribution_key, get_rng_for_key
from .filesystem_constants import DATA_DIR

import logging

logger = logging.getLogger(__name__)

try:
    from edges.cache import CacheBackend, MappingCache, EdgeCacheGlue
except Exception:
    CacheBackend = MappingCache = EdgeCacheGlue = None  # fallback


def add_cf_entry(
    cfs_mapping, supplier_info, consumer_info, direction, indices, value, uncertainty
):
    """
    Append a characterized-exchange entry to the in-memory CF mapping.

    :param cfs_mapping: Target list that collects CF entries.
    :param supplier_info: Supplier-side metadata for this CF (matrix, location, classifications, etc.).
    :param consumer_info: Consumer-side metadata for this CF (location, classifications, etc.).
    :param direction: Exchange direction the CF applies to.
    :param indices: Pairs of (supplier_idx, consumer_idx) covered by this CF.
    :param value: CF value or symbolic expression.
    :param uncertainty: Optional uncertainty specification for this CF.
    :return: None
    """

    supplier_entry = dict(supplier_info)
    consumer_entry = dict(consumer_info)

    supplier_entry["matrix"] = (
        "biosphere" if direction == "biosphere-technosphere" else "technosphere"
    )
    consumer_entry["matrix"] = "technosphere"

    entry = {
        "supplier": supplier_entry,
        "consumer": consumer_entry,
        "positions": indices,
        "direction": direction,
        "value": value,
    }
    if uncertainty is not None:
        entry["uncertainty"] = uncertainty
    cfs_mapping.append(entry)


@lru_cache(maxsize=None)
def _equality_supplier_signature_cached(hashable_supplier_info: tuple) -> tuple:
    """
    Create a normalized, hashable signature for supplier matching (cached).

    :param hashable_supplier_info: Pre-hashable supplier info tuple.
    :return: A tuple representing the normalized supplier signature.
    """
    info = dict(hashable_supplier_info)

    if "classifications" in info:
        classifications = info["classifications"]

        if isinstance(classifications, (list, tuple)):
            try:
                info["classifications"] = tuple(
                    sorted((str(s), str(c)) for s, c in classifications)
                )
            except Exception:
                info["classifications"] = ()
        elif isinstance(classifications, dict):
            info["classifications"] = tuple(
                (scheme, tuple(sorted(map(str, codes))))
                for scheme, codes in sorted(classifications.items())
            )
        else:
            info["classifications"] = ()

    return make_hashable(info)


@lru_cache(maxsize=None)
def _equality_consumer_signature_cached(hashable_consumer_info: tuple) -> tuple:
    info = dict(hashable_consumer_info)
    if "classifications" in info:
        info["classifications"] = _norm_cls(info["classifications"])
    # keep only stable fields
    keys = ("name", "reference product", "unit", "location", "classifications")
    norm = {k: info.get(k) for k in keys if k in info}
    return make_hashable(norm)


class SigIntern:
    __slots__ = ("_map", "_rev")

    def __init__(self):
        self._map = {}  # tuple -> int id
        self._rev = []  # id -> tuple

    def intern(self, tup):
        m = self._map
        if tup in m:
            return m[tup]
        i = len(m)
        m[tup] = i
        self._rev.append(tup)
        return i

    def get_tuple(self, i):
        return self._rev[i]


def _collect_cf_prefixes_used_by_method(raw_cfs_data):
    """
    Collect all classification prefixes that appear in a CF method.

    :param data: Raw LCIA method data.
    :return: A set of prefixes found in CF entries.
    """
    needed = {}

    def _push(scheme, code):
        if code is None:
            return
        sc = str(scheme).lower().strip()
        c = str(code).split(":", 1)[0].strip()
        if not c:
            return
        needed.setdefault(sc, set()).add(c)

    for cf in raw_cfs_data:
        for side in ("supplier", "consumer"):
            cls = cf.get(side, {}).get("classifications")
            if not cls:
                continue
            # normalize to (("SCHEME", ("code", ...)), ...)
            norm = _norm_cls(cls)
            for scheme, codes in norm:
                for code in codes:
                    _push(scheme, code)

    return {k: frozenset(v) for k, v in needed.items()}


def _build_prefix_index_restricted(
    idx_to_norm_classes: dict[int, tuple], required_prefixes: dict[str, frozenset[str]]
):
    """
    Build an index mapping classification prefixes to activities.

    :param activities: Iterable of activity datasets.
    :param required_prefixes: Prefixes to include in the index.
    :return: Dict mapping prefix -> set of activity keys.
    """
    out = {
        scheme: {p: set() for p in prefs} for scheme, prefs in required_prefixes.items()
    }

    for idx, norm in idx_to_norm_classes.items():
        if not norm:
            continue
        for scheme, codes in norm:
            sch = str(scheme).lower().strip()
            wanted = required_prefixes.get(sch)
            if not wanted:
                continue
            for code in codes:
                base = str(code).split(":", 1)[0].strip()
                if not base:
                    continue
                # generate progressive prefixes: '01.12' -> '0','01','01.','01.1','01.12'
                # (progressive is safest because your CF can be any prefix)
                for k in range(1, len(base) + 1):
                    pref = base[:k]
                    if pref in wanted:
                        out[sch][pref].add(idx)
    return out


def _cls_candidates_from_cf(
    cf_classifications,
    prefix_index_by_scheme: dict[str, dict[str, set[int]]],
    adjacency_keys: set[int] | None = None,
) -> set[int]:
    """
    Return candidate indices for a CF's classifications using the given prefix index.

    Faster version:
      - Avoids per-code O(len(code)) prefix generation; only slices lengths that
        actually exist in the index's keys for that scheme.
      - Minimizes string allocations and dict.get calls via local bindings.
      - Applies optional adjacency filtering at the end.

    Expected index shape:
      prefix_index_by_scheme[scheme_lower][prefix] -> set[int]
      where 'prefix' is a classification code prefix (already pre-built).
    """
    if not cf_classifications:
        return set()

    norm = _norm_cls(cf_classifications)  # (("SCHEME", ("code", ...)), ...)
    if not norm:
        return set()

    # Cache "available prefix lengths per scheme" derived from the index keys.
    # Keyed by the *id* of each per-scheme bucket dict to stay valid even if
    # different dict objects are passed across calls.
    _lengths_cache = getattr(_cls_candidates_from_cf, "_lengths_cache", None)
    if _lengths_cache is None:
        _lengths_cache = {}
        setattr(_cls_candidates_from_cf, "_lengths_cache", _lengths_cache)

    out = set()
    out_update = out.update  # local binding for speed

    get_scheme_bucket = prefix_index_by_scheme.get

    for scheme, codes in norm:
        sch = scheme.lower().strip()
        bucket = get_scheme_bucket(sch)
        if not bucket:
            continue

        # lengths of prefixes that actually exist in the index for this scheme
        bucket_id = id(bucket)
        lens = _lengths_cache.get(bucket_id)
        if lens is None:
            # Derive once: e.g., {"01", "011", "0112"} -> (2, 3, 4)
            # Tuple for fast iteration and stable order (small to large).
            lens = tuple(sorted({len(k) for k in bucket.keys()}))
            _lengths_cache[bucket_id] = lens

        bget = bucket.get  # local

        for code in codes or ():
            if code is None:
                continue
            # Use text before ":" as the "base" (same logic as elsewhere)
            base = str(code)
            colon = base.find(":")
            if colon != -1:
                base = base[:colon]
            base = base.strip()
            if not base:
                continue

            n = len(base)
            # Only slice the lengths we actually have in the index.
            for L in lens:
                if L > n:
                    break
                hits = bget(base[:L])
                if hits:
                    out_update(hits)

    if adjacency_keys is not None and out:
        # Intersect at the end to avoid repeated set ops inside the loop.
        out &= adjacency_keys

    return out


def _norm_cls(x):
    """
    Normalize 'classifications' to a canonical, hashable form:
      (("SCHEME", ("code1","code2", ...)), ("SCHEME2", (...)), ...)
    Accepts:
      - dict: {"CPC": ["01","02"], "ISIC": ["A"]}
      - list/tuple of pairs: [("CPC","01"), ("CPC",["02","03"]), ("ISIC","A")]

    :param c: Classification entry (tuple or dict).
    :return: Normalized classification tuple.
    """
    if not x:
        return ()
    # Accumulate into {scheme: set(codes)}
    bag = {}
    if isinstance(x, dict):
        for scheme, codes in x.items():
            if codes is None:
                continue
            if isinstance(codes, (list, tuple, set)):
                codes_iter = codes
            else:
                codes_iter = [codes]
            bag.setdefault(str(scheme), set()).update(str(c) for c in codes_iter)
    elif isinstance(x, (list, tuple)):
        for item in x:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            scheme, codes = item
            if codes is None:
                continue
            if isinstance(codes, (list, tuple, set)):
                codes_iter = codes
            else:
                codes_iter = [codes]
            bag.setdefault(str(scheme), set()).update(str(c) for c in codes_iter)
    else:
        return ()

    # Canonical: schemes sorted; codes sorted; all tuples
    return tuple((scheme, tuple(sorted(bag[scheme]))) for scheme in sorted(bag))


class EdgeLCIA:
    """
    Class that implements the calculation of the regionalized life cycle impact assessment (LCIA) results.
    Relies on bw2data.LCA class for inventory calculations and matrices.
    """

    def __init__(
        self,
        demand: dict,
        method: Optional[tuple] = None,
        weight: Optional[str] = "population",
        parameters: Optional[dict] = None,
        scenario: Optional[str] = None,
        filepath: Optional[str] = None,
        allowed_functions: Optional[dict] = None,
        use_distributions: Optional[bool] = False,
        random_seed: Optional[int] = None,
        iterations: Optional[int] = 100,
        cache_enabled: bool = True,
        cache_backend: Optional[object] = None,
    ):
        """
        Initialize an EdgeLCIA object for exchange-level life cycle impact assessment.

        :param demand: Dictionary of {activity: amount} for the functional unit.
        :param method: Tuple identifying the LCIA method (e.g. ("AWARE 2.0", "Country", "all", "yearly")).
        :param weight: Weighting scheme for CFs (e.g. "population", "gdp", or None).
        :param parameters: Optional dict of parameter definitions for scenarios.
        :param scenario: Optional scenario name to select a parameter set.
        :param filepath: Optional path to a JSON file with LCIA method data.
        :param allowed_functions: Optional dict of user-defined functions for symbolic evaluation.
        :param use_distributions: Whether to sample from uncertainty distributions.
        :param random_seed: Seed for random number generation (for reproducibility).
        :param iterations: Number of Monte Carlo iterations for uncertainty propagation.
        :param cache_enabled: Whether to enable caching of mapping results.
        :param cache_backend: Optional custom cache backend instance.

        Notes
        -----
        After initialization, the standard evaluation sequence is:
        1. `lci()`
        2. `map_exchanges()`
        3. Optionally: regional mapping methods
        4. `evaluate_cfs()`
        5. `lcia()`
        6. Optionally: `statistics()`, `generate_df_table()`
        """
        self.cf_index = None
        self.scenario_cfs = None
        self.method_metadata = None
        self.demand = demand
        self.weights = None
        self.consumer_lookup = None
        self.reversed_consumer_lookup = None
        self.processed_technosphere_edges = None
        self.processed_biosphere_edges = None
        self.raw_cfs_data = None
        self.unprocessed_technosphere_edges = []
        self.unprocessed_biosphere_edges = []
        self.score = None
        self.cfs_number = None
        self.filepath = Path(filepath) if filepath else None
        self.reversed_biosphere = None
        self.reversed_activity = None
        self.characterization_matrix = None
        self.method = method  # Store the method argument in the instance
        self.position_to_technosphere_flows_lookup = None
        self.technosphere_flows_lookup = defaultdict(list)
        self.technosphere_edges = []
        self.technosphere_flow_matrix = None
        self.biosphere_edges = []
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.ignored_method_exchanges = list()
        self.weight_scheme: str = weight

        # Accept both "parameters" and "scenarios" for flexibility
        self.parameters = parameters or {}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.scenario = scenario  # New: store default scenario
        self.scenario_length = validate_parameter_lengths(parameters=self.parameters)
        self.use_distributions = use_distributions
        self.iterations = iterations
        self.random_seed = random_seed if random_seed is not None else 42
        self.random_state = np.random.default_rng(self.random_seed)

        self.lca = bw2calc.LCA(demand=self.demand)
        self._load_raw_lcia_data()
        self.cfs_mapping = []

        self.SAFE_GLOBALS = {
            "__builtins__": None,
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log10": math.log10,
        }

        # Allow user-defined trusted functions explicitly
        if allowed_functions:
            self.SAFE_GLOBALS.update(allowed_functions)

        self._cached_supplier_keys = self._get_candidate_supplier_keys()

        self._cache_enabled = bool(cache_enabled)
        self._cache_last_saved_len = 0  # we’ll track what’s new after each step

        # Lazy cache wiring
        self._cache_glue = None
        if self._cache_enabled and EdgeCacheGlue is not None:
            try:
                backend = cache_backend or CacheBackend.default()
                self._mapping_cache = MappingCache(backend)
                self._cache_glue = EdgeCacheGlue(self._mapping_cache)
            except Exception:
                self.logger.exception(
                    "Failed to initialize cache backend. Continuing without cache."
                )
                self._cache_glue = None

        self._sig_supplier_cache = {}  # key: (direction, supplier_idx) -> str
        self._sig_consumer_cache = {}  # key: consumer_idx -> str

        # Optional: memoize act-id lookups too (avoid dict gets in tight loops)
        self._supplier_actid_cache = {}  # key: (direction, supplier_idx) -> str
        self._consumer_actid_cache = {}  # key: consumer_idx -> str

        self._ctx_hashes = {}

        self._sig_intern = SigIntern()
        self._sig_hash_cache = {}  # id(int) -> hashed string (computed lazily)

    def _stable_sig_bytes(self, x):
        # fast, deterministic encoder for canonical tuples
        if x is None:
            return b"n"
        if isinstance(x, (int, float)):
            return f"f{repr(x)}".encode()
        if isinstance(x, str):
            return b"s|" + x.encode()
        if isinstance(x, tuple):
            return b"t|" + b"|".join(self._stable_sig_bytes(e) for e in x)
        if isinstance(x, list):
            return b"l|" + b"|".join(self._stable_sig_bytes(e) for e in x)
        if isinstance(x, dict):
            items = sorted(x.items(), key=lambda kv: kv[0])
            return b"d|" + b"|".join(
                self._stable_sig_bytes(k) + b"=" + self._stable_sig_bytes(v)
                for k, v in items
            )
        return b"o|" + str(x).encode()

    def _sig_hash(self, sig_id: int) -> str:
        """Memoized blake2s hash of the interned signature tuple."""
        hs = self._sig_hash_cache.get(sig_id)
        if hs is not None:
            return hs
        import hashlib

        tup = self._sig_intern.get_tuple(sig_id)
        h = hashlib.blake2s(digest_size=16)
        h.update(self._stable_sig_bytes(tup))
        hs = h.hexdigest()
        self._sig_hash_cache[sig_id] = hs
        return hs

    def _ctx_hash(self, kind: str, fn):
        v = self._ctx_hashes.get(kind)
        if v is None:
            v = fn()
            self._ctx_hashes[kind] = v
        return v

    def _supplier_sig(self, i: int, direction: str) -> str:
        key = (direction, i)
        sig = self._sig_supplier_cache.get(key)
        if sig is not None:
            return sig
        # Reuse EdgeCacheGlue impl but cache result
        sig = (
            self._cache_glue._supplier_sig_from_idx(self, i, direction)
            if self._cache_glue
            else ""
        )
        self._sig_supplier_cache[key] = sig
        return sig

    def _consumer_sig(self, j: int) -> str:
        sig = self._sig_consumer_cache.get(j)
        if sig is not None:
            return sig
        sig = (
            self._cache_glue._consumer_sig_from_idx(self, j) if self._cache_glue else ""
        )
        self._sig_consumer_cache[j] = sig
        return sig

    def _supplier_actid(self, i: int, direction: str) -> str:
        key = (direction, i)
        v = self._supplier_actid_cache.get(key)
        if v is not None:
            return v
        v = (
            str(self.reversed_biosphere[i])
            if direction == "biosphere-technosphere"
            else str(self.reversed_activity[i])
        )
        self._supplier_actid_cache[key] = v
        return v

    def _consumer_actid(self, j: int) -> str:
        v = self._consumer_actid_cache.get(j)
        if v is not None:
            return v
        v = str(self.reversed_activity[j])
        self._consumer_actid_cache[j] = v
        return v

    def _load_raw_lcia_data(self):
        """
        Load and validate raw LCIA data for a given method.

        :param method: Method identifier.
        :return: Parsed LCIA data structure.
        """
        if self.filepath is None:
            self.filepath = DATA_DIR / f"{'_'.join(self.method)}.json"
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Store full method metadata except exchanges and parameters
        self.raw_cfs_data, self.method_metadata = format_data(raw, self.weight_scheme)
        # check for NaNs in the raw CF data
        assert_no_nans_in_cf_list(self.raw_cfs_data, file_source=self.filepath)
        self.raw_cfs_data = normalize_classification_entries(self.raw_cfs_data)
        self.cfs_number = len(self.raw_cfs_data)

        # Extract parameters or scenarios from method file if not already provided
        if not self.parameters:
            self.parameters = raw.get("scenarios", raw.get("parameters", {}))
        if not self.parameters:
            self.logger.warning(
                f"No parameters or scenarios found in method file: {self.filepath}"
            )

        # Fallback to default scenario
        if self.scenario and self.scenario not in self.parameters:
            self.logger.error(
                f"Scenario '{self.scenario}' not found in method file. Available scenarios: {list(self.parameters)}"
            )
            raise ValueError(
                f"Scenario '{self.scenario}' not found in available parameters: {list(self.parameters)}"
            )

        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight", "position", "excludes"}
        }

        self.cf_index = build_cf_index(self.raw_cfs_data)

    def _initialize_weights(self):
        """
        Initialize weights for scenarios and parameters.

        :return: None
        """

        if self.weights is not None:
            return

        if not self.raw_cfs_data:
            self.weights = {}
            return

        self.weights = {}
        for cf in self.raw_cfs_data:
            supplier = cf.get("supplier", {})
            consumer = cf.get("consumer", {})
            supplier_location = supplier.get("location", "__ANY__")
            consumer_location = consumer.get("location", "__ANY__")
            weight = cf.get("weight", 0)

            self.weights[(supplier_location, consumer_location)] = float(weight)

        if hasattr(self, "_geo") and self._geo is not None:
            self._geo._cached_lookup.cache_clear()

    def _get_candidate_supplier_keys(self):
        """
        Get possible supplier activity keys matching a CF entry.

        :param cf: Characterization factor entry.
        :return: List of supplier activity keys.
        """

        if hasattr(self, "_cached_supplier_keys"):
            return self._cached_supplier_keys

        grouping_mode = self._detect_cf_grouping_mode()
        cfs_lookup = preprocess_cfs(self.raw_cfs_data, by=grouping_mode)

        keys = set()
        for cf_list in cfs_lookup.values():
            for cf in cf_list:
                filtered = {
                    k: cf["supplier"].get(k)
                    for k in self.required_supplier_fields
                    if cf["supplier"].get(k) is not None
                }

                # Normalize classification field
                if "classifications" in filtered:
                    c = filtered["classifications"]
                    if isinstance(c, dict):
                        filtered["classifications"] = tuple(
                            (scheme, tuple(vals)) for scheme, vals in sorted(c.items())
                        )
                    elif isinstance(c, list):
                        filtered["classifications"] = tuple(c)

                keys.add(make_hashable(filtered))

        self._cached_supplier_keys = keys
        return keys

    def _detect_cf_grouping_mode(self):
        """
        Detect the grouping mode of a CF entry (e.g. technosphere vs biosphere).

        :param cf: Characterization factor entry.
        :return: Grouping mode string.
        """

        has_consumer_locations = any(
            "location" in cf.get("consumer", {}) for cf in self.raw_cfs_data
        )
        has_supplier_locations = any(
            "location" in cf.get("supplier", {}) for cf in self.raw_cfs_data
        )
        if has_consumer_locations and not has_supplier_locations:
            return "consumer"
        elif has_supplier_locations and not has_consumer_locations:
            return "supplier"
        else:
            return "both"

    def _resolve_parameters_for_scenario(
        self, scenario_idx: int, scenario_name: Optional[str] = None
    ) -> dict:
        """
        Resolve symbolic parameters for a given scenario.

        :param params: Dict of parameter definitions.
        :param scenario: Scenario name.
        :return: Dict of resolved parameter values.
        """

        scenario_name = scenario_name or self.scenario

        param_set = self.parameters.get(scenario_name)

        if param_set is None:
            self.logger.warning(
                f"No parameter set found for scenario '{scenario_name}'. Using empty defaults."
            )

        resolved = {}
        if param_set is not None:
            for k, v in param_set.items():
                if isinstance(v, dict):
                    resolved[k] = v.get(str(scenario_idx), list(v.values())[-1])
                else:
                    resolved[k] = v
        return resolved

    def _update_unprocessed_edges(self):
        """
        Add new edges to the list of unprocessed edges.

        :param new_edges: Iterable of edges.
        :return: None
        """

        self.processed_biosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "biosphere-technosphere"
            for pos in cf["positions"]
        }

        self.processed_technosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "technosphere-technosphere"
            for pos in cf["positions"]
        }

        logger.info(
            "Processed edges: %d",
            len(self.processed_biosphere_edges)
            + len(self.processed_technosphere_edges),
        )

        self.unprocessed_biosphere_edges = [
            edge
            for edge in self.biosphere_edges
            if edge not in self.processed_biosphere_edges
        ]

        self.unprocessed_technosphere_edges = [
            edge
            for edge in self.technosphere_edges
            if edge not in self.processed_technosphere_edges
        ]

    def _preprocess_lookups(self):
        """
        Precompute all lookup structures, hot-field caches, classification prefix indexes,
        and per-index **interned** signatures (small ints; no hashing here).

        Results:
          - supplier_lookup_bio / supplier_lookup_tech / consumer_lookup
          - reversed_supplier_lookup_bio / reversed_supplier_lookup_tech / reversed_consumer_lookup
          - supplier_loc_bio / supplier_loc_tech / consumer_loc
          - supplier_cls_bio / supplier_cls_tech / consumer_cls   (normalized tuples)
          - cls_prefidx_supplier_bio / cls_prefidx_supplier_tech / cls_prefidx_consumer
          - supplier_sig_bio / supplier_sig_tech / consumer_sig   (interned int IDs)
          - supplier_lookup (merged, for back-compat)
        """

        # -------- 1) Required CONSUMER fields (ignore control/meta) -----------------
        IGNORED_FIELDS = {"matrix", "operator", "weight", "classifications", "position"}
        self.required_consumer_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        # -------- 2) Build forward lookups -----------------------------------------
        # Supplier lookups (per matrix)
        if self.biosphere_flows:
            self.supplier_lookup_bio = preprocess_flows(
                flows_list=self.biosphere_flows,
                mandatory_fields=self.required_supplier_fields,
            )
        else:
            self.supplier_lookup_bio = {}

        if self.technosphere_flows:
            self.supplier_lookup_tech = preprocess_flows(
                flows_list=self.technosphere_flows,
                mandatory_fields=self.required_supplier_fields,
            )
        else:
            self.supplier_lookup_tech = {}

        # Consumer lookup (always technosphere)
        self.consumer_lookup = preprocess_flows(
            flows_list=self.technosphere_flows,
            mandatory_fields=self.required_consumer_fields,
        )

        # -------- 3) Materialize reversed lookups (pos -> dict of fields) ----------
        def _materialize_reversed(lookup: dict[int, list[int]]) -> dict[int, dict]:
            # map pos -> dict(key) so callers can use it directly (no per-call dict(...) in hot loops)
            return {
                pos: dict(key) for key, positions in lookup.items() for pos in positions
            }

        self.reversed_supplier_lookup_bio = _materialize_reversed(
            self.supplier_lookup_bio
        )
        self.reversed_supplier_lookup_tech = _materialize_reversed(
            self.supplier_lookup_tech
        )
        self.reversed_consumer_lookup = _materialize_reversed(self.consumer_lookup)

        # Enrich consumer reversed lookups with full metadata from technosphere flow table (once).
        # Ensures 'classifications' and 'location' are present without extra work later.
        ptfl = self.position_to_technosphere_flows_lookup or {}
        for idx, info in self.reversed_consumer_lookup.items():
            extra = ptfl.get(idx)
            if not extra:
                continue
            if "location" not in info and "location" in extra:
                info["location"] = extra["location"]
            if "classifications" not in info and "classifications" in extra:
                info["classifications"] = extra["classifications"]

        # Back-compat merged supplier lookup
        if self.supplier_lookup_bio and not self.supplier_lookup_tech:
            self.supplier_lookup = self.supplier_lookup_bio
        elif self.supplier_lookup_tech and not self.supplier_lookup_bio:
            self.supplier_lookup = self.supplier_lookup_tech
        else:
            merged = {}
            for src in (self.supplier_lookup_bio, self.supplier_lookup_tech):
                for k, v in src.items():
                    if k in merged:
                        merged[k].extend(v)
                    else:
                        merged[k] = list(v)
            self.supplier_lookup = merged

        # -------- 4) Hot-field caches (locations + normalized classifications) ------
        # (Avoid repeated dict lookups + allocations later)
        self.supplier_loc_bio = {
            i: d.get("location") for i, d in self.reversed_supplier_lookup_bio.items()
        }
        self.supplier_loc_tech = {
            i: d.get("location") for i, d in self.reversed_supplier_lookup_tech.items()
        }
        self.consumer_loc = {
            i: d.get("location") for i, d in self.reversed_consumer_lookup.items()
        }

        self.supplier_cls_bio = {
            i: _norm_cls(d.get("classifications"))
            for i, d in self.reversed_supplier_lookup_bio.items()
        }
        self.supplier_cls_tech = {
            i: _norm_cls(d.get("classifications"))
            for i, d in self.reversed_supplier_lookup_tech.items()
        }
        self.consumer_cls = {
            i: _norm_cls(d.get("classifications"))
            for i, d in self.reversed_consumer_lookup.items()
        }

        # -------- 5) CF-needed classification prefixes + by-length optimization -----
        # Collect only the prefixes that appear in the CFs (per scheme)
        self._cf_needed_prefixes = _collect_cf_prefixes_used_by_method(
            self.raw_cfs_data
        )
        # Precompute "wanted by length" to avoid progressive 1..N slicing later
        #   wanted_by_len[scheme][L] = {prefixes of length L}
        self._wanted_by_len = {
            sch: {L: {p for p in prefs if len(p) == L} for L in {len(p) for p in prefs}}
            for sch, prefs in self._cf_needed_prefixes.items()
        }

        def _build_prefix_index_restricted_fast(
            idx_to_norm_classes: dict[int, tuple],
            required_prefixes: dict[str, frozenset[str]],
            wanted_by_len: dict[str, dict[int, set[str]]],
        ):
            """Map (scheme, prefix) -> set(indices) using only actually-needed prefix lengths."""
            out = {
                sch: {p: set() for p in prefs}
                for sch, prefs in required_prefixes.items()
            }
            for idx, norm in idx_to_norm_classes.items():
                if not norm:
                    continue
                for scheme, codes in norm:
                    sch = str(scheme).lower().strip()
                    if sch not in required_prefixes:
                        continue
                    lens_map = wanted_by_len.get(sch)
                    if not lens_map:
                        continue
                    for code in codes:
                        base = str(code)
                        # take part before ":" once
                        colon = base.find(":")
                        if colon != -1:
                            base = base[:colon]
                        base = base.strip()
                        if not base:
                            continue
                        n = len(base)
                        for L, prefset in lens_map.items():
                            if L > n:
                                continue
                            pref = base[:L]
                            if pref in prefset:  # O(1) set membership
                                out[sch][pref].add(idx)
            return out

        # Suppliers/Consumers prefix index tables
        self.cls_prefidx_supplier_bio = _build_prefix_index_restricted_fast(
            self.supplier_cls_bio, self._cf_needed_prefixes, self._wanted_by_len
        )
        self.cls_prefidx_supplier_tech = _build_prefix_index_restricted_fast(
            self.supplier_cls_tech, self._cf_needed_prefixes, self._wanted_by_len
        )
        self.cls_prefidx_consumer = _build_prefix_index_restricted_fast(
            self.consumer_cls, self._cf_needed_prefixes, self._wanted_by_len
        )

        # -------- 6) INTERN signatures (NO hashing here) ----------------------------
        # We work with small integer IDs during the run to avoid expensive hashing/comparisons per (i,j).
        # Hashing (if needed) is done later in bulk when persisting/validating.
        # Ensure the interner exists
        if not hasattr(self, "_sig_intern"):
            # Define tiny class if not present (preferably you placed this at module top)
            class SigIntern:
                __slots__ = ("_map", "_rev")

                def __init__(self):
                    self._map, self._rev = {}, []

                def intern(self, tup):
                    m = self._map
                    if tup in m:
                        return m[tup]
                    i = len(m)
                    m[tup] = i
                    self._rev.append(tup)
                    return i

                def get_tuple(self, i):
                    return self._rev[i]

            self._sig_intern = SigIntern()

        # Keep a hash cache for later persistence (id -> hash string)
        if not hasattr(self, "_sig_hash_cache"):
            self._sig_hash_cache = {}

        # Supplier signature tuples -> interned ids
        from .edgelcia import (
            _equality_supplier_signature_cached,
            make_hashable,
        )  # reuse your existing normalizer

        self.supplier_sig_bio = {}
        for i, info in self.reversed_supplier_lookup_bio.items():
            tup = tuple(_equality_supplier_signature_cached(make_hashable(info)))
            self.supplier_sig_bio[i] = self._sig_intern.intern(tup)

        self.supplier_sig_tech = {}
        for i, info in self.reversed_supplier_lookup_tech.items():
            tup = tuple(_equality_supplier_signature_cached(make_hashable(info)))
            self.supplier_sig_tech[i] = self._sig_intern.intern(tup)

        # Consumer signature tuples -> interned ids
        self.consumer_sig = {}
        for j, info in self.reversed_consumer_lookup.items():
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
            # ✅ use the same canonicalizer as cache glue
            tup = tuple(_equality_consumer_signature_cached(make_hashable(norm)))
            self.consumer_sig[j] = self._sig_intern.intern(tup)

    # ---- Transparent cache helpers -------------------------------------------
    def _cache_preload_if_any(self):
        """Restore previously matched edges (if cache exists)."""
        # --- before calling self._cache_glue.preload_from_cache(self)
        if self._cache_glue:
            mh = self._cache_glue.method_hash(self)
            ph = self._cache_glue.params_hash(self)
            wh = self._cache_glue.weights_hash(self)
            gh = self._cache_glue.geo_hash(self)
            proj = getattr(self.lca, "project", "bw2")
            self.logger.info(
                "CACHE CTX: project=%s | method=%s | params=%s | weights=%s | geo=%s",
                proj,
                mh[:12],
                ph[:12],
                wh[:12],
                gh[:12],
            )

        # preconditions: lci() done; lookups ready
        try:
            return self._cache_glue.preload_from_cache(self)
        finally:
            self.logger.info("Cache preload complete.")
            # track baseline for 'new rows' to persist later
            self._cache_last_saved_len = len(self.cfs_mapping)

    def _cache_save_new(self):
        """Persist only rows added since the last save baseline."""
        if not self._cache_glue:
            return 0
        try:
            written = self._cache_glue.save_new_matches(
                self, self._cache_last_saved_len
            )
            self._cache_last_saved_len = len(self.cfs_mapping)
            return written
        except Exception:
            self.logger.exception("Failed to save new cache entries.")
            return 0

    @classmethod
    def clear_cache(cls):
        """Public convenience to clear all edges caches."""
        if CacheBackend is None:
            return
        try:
            CacheBackend.default().clear()
        except Exception:
            pass

    def _get_consumer_info(self, consumer_idx):
        """
        Extract consumer information from an exchange.

        :param exc: Exchange dataset.
        :return: Dict with consumer attributes.
        """

        info = self.reversed_consumer_lookup.get(consumer_idx, {})
        if "location" not in info or "classifications" not in info:
            fallback = self.position_to_technosphere_flows_lookup.get(consumer_idx, {})
            if fallback:
                if "location" not in info and "location" in fallback:
                    loc = fallback["location"]
                    info["location"] = loc
                    self.consumer_loc[consumer_idx] = loc
                if "classifications" not in info and "classifications" in fallback:
                    cls = fallback["classifications"]
                    info["classifications"] = cls
                    self.consumer_cls[consumer_idx] = _norm_cls(cls)
        return info

    @lru_cache(maxsize=None)
    def _extract_excluded_subregions(self, idx: int, decomposed_exclusions: frozenset):
        """
        Get excluded subregions for a dynamic supplier or consumer.

        :param idx: Index of the supplier or consumer flow.
        :param decomposed_exclusions: A frozenset of decomposed exclusions for the flow.
        :return: A frozenset of excluded subregions.
        """
        decomposed_exclusions = dict(decomposed_exclusions)

        act = self.position_to_technosphere_flows_lookup.get(idx, {})
        name = act.get("name")
        reference_product = act.get("reference product")
        exclusions = self.technosphere_flows_lookup.get((name, reference_product), [])

        excluded_subregions = []
        for loc in exclusions:
            if loc in ["RoW", "RoE"]:
                continue
            excluded_subregions.extend(decomposed_exclusions.get(loc, [loc]))

        return frozenset(excluded_subregions)

    def lci(self) -> None:
        """
        Perform the life cycle inventory (LCI) calculation and extract relevant exchanges.

        This step computes the inventory matrix using Brightway2 and stores the
        biosphere and/or technosphere exchanges relevant for impact assessment.

        It also builds lookups for flow indices, supplier and consumer locations,
        and initializes flow matrices used in downstream CF mapping.

        Must be called before `map_exchanges()` or any mapping or evaluation step.

        :return: None
        """

        self.lca.lci()

        if all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        ):
            self.technosphere_flow_matrix = build_technosphere_edges_matrix(
                self.lca.technosphere_matrix, self.lca.supply_array
            )
            self.technosphere_edges = set(
                list(zip(*self.technosphere_flow_matrix.nonzero()))
            )
        else:
            self.biosphere_edges = set(list(zip(*self.lca.inventory.nonzero())))

        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)

        if len(unique_biosphere_flows) > 0:
            self.biosphere_flows = get_flow_matrix_positions(
                {
                    k: v
                    for k, v in self.lca.biosphere_dict.items()
                    if v in unique_biosphere_flows
                }
            )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.lca.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.lca.reverse_dict()

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }

    def map_exchanges(self):
        """
        Direction-aware matching with per-direction adjacency, indices, and allowlists.
        Leaves near-misses due to 'location' for later geo steps.

        Memory-minded version: chunked position emission + progressive saving.
        """

        log = self.logger.getChild("map")  # edges.edgelcia.EdgeLCIA.map

        # -------- setup -----------------------------------------------------------
        self._initialize_weights()
        self._preprocess_lookups()

        pre_len = len(self.cfs_mapping)
        restored = self._cache_preload_if_any()
        post_len = len(self.cfs_mapping)

        added_entries = self.cfs_mapping[pre_len:post_len]
        pos_from_cache = [p for cf in added_entries for p in cf.get("positions", ())]
        self.logger.info(
            "CACHE PROOF: entries_from_cache=%d, positions_from_cache=%d, sample_positions=%s",
            len(added_entries),
            len(pos_from_cache),
            pos_from_cache[:5],
        )
        if restored:
            self.logger.info(
                "Cache restored %d characterized exchange positions.", restored
            )
            from_cache = sum(
                1 for cf in self.cfs_mapping if cf.get("origin") == "cache-preload"
            )
            pos_from_cache = len(getattr(self, "_cache_restored_positions", []))
            sample = getattr(self, "_cache_restored_positions", [])[:5]
            self.logger.info(
                "CACHE PROOF: entries_from_cache=%d, positions_from_cache=%d, sample_positions=%s",
                from_cache,
                pos_from_cache,
                sample,
            )

        # ---- Build direction-specific bundles -----------------------------------
        DIR_BIO = "biosphere-technosphere"
        DIR_TECH = "technosphere-technosphere"

        def build_adj(edges):
            ebs, ebc = defaultdict(set), defaultdict(set)
            rem = set(edges)
            for s, c in rem:
                ebs[s].add(c)
                ebc[c].add(s)
            return rem, ebs, ebc

        rem_bio, ebs_bio, ebc_bio = build_adj(self.biosphere_edges)
        rem_tec, ebs_tec, ebc_tec = build_adj(self.technosphere_edges)

        def _prune_processed(rem, ebs, ebc, processed):
            if not processed:
                return
            for s, c in list(processed):
                if (s, c) in rem:
                    rem.remove((s, c))
                    if s in ebs:
                        ebs[s].discard(c)
                        if not ebs[s]:
                            del ebs[s]
                    if c in ebc:
                        ebc[c].discard(s)
                        if not ebc[c]:
                            del ebc[c]

        log.info("REM before prune: bio=%d tech=%d", len(rem_bio), len(rem_tec))
        _prune_processed(
            rem_bio, ebs_bio, ebc_bio, getattr(self, "processed_biosphere_edges", set())
        )
        _prune_processed(
            rem_tec,
            ebs_tec,
            ebc_tec,
            getattr(self, "processed_technosphere_edges", set()),
        )
        log.info("REM after prune:  bio=%d tech=%d", len(rem_bio), len(rem_tec))

        # ---- EARLY RETURN 1: all edges restored by positive cache
        if not rem_bio and not rem_tec:
            log.info(
                "CF loop skipped: nothing left after processed-prune (cache restored all)."
            )
            self.eligible_edges_for_next_bio = set()
            self.eligible_edges_for_next_tech = set()
            written = self._cache_save_new()
            self.logger.info(
                "Cache saved %d newly characterized positions (map_exchanges).",
                written or 0,
            )
            log.info("REM final (unmatched): bio=0 tech=0")
            log.info("ALLOW (loc-only):     bio=0 tech=0")
            log.info("NON-LOC misses:       bio=0 tech=0")
            return

        # ---- NEGATIVE prune (non-location misses from prior runs)
        neg = self._cache_glue.preload_negative(self) if self._cache_glue else {}

        def prune_with_negative(rem, direction):
            if not neg:
                return rem
            keep = set()
            _s_act = self._cache_glue._supplier_act_id
            _c_act = self._cache_glue._consumer_act_id
            _s_sig = self._cache_glue._supplier_sig_from_idx
            _c_sig = self._cache_glue._consumer_sig_from_idx
            for s, c in rem:
                k = (
                    direction,
                    self._supplier_actid(s, direction),
                    self._consumer_actid(c),
                )
                sig = neg.get(k)
                if not sig:
                    keep.add((s, c))
                    continue
                if sig == (self._supplier_sig(s, direction), self._consumer_sig(c)):
                    continue  # prune
                keep.add((s, c))
            return keep

        rem_bio = prune_with_negative(rem_bio, DIR_BIO)
        rem_tec = prune_with_negative(rem_tec, DIR_TECH)
        log.info("REM after neg:    bio=%d tech=%d", len(rem_bio), len(rem_tec))

        # ---- EARLY RETURN 2: negatives + positives settle everything
        if not rem_bio and not rem_tec:
            log.info(
                "CF loop skipped: nothing left after negative-prune (only cache involved)."
            )
            self.eligible_edges_for_next_bio = set()
            self.eligible_edges_for_next_tech = set()
            written = self._cache_save_new()
            self.logger.info(
                "Cache saved %d newly characterized positions (map_exchanges).",
                written or 0,
            )
            log.info("REM final (unmatched): bio=0 tech=0")
            log.info("ALLOW (loc-only):     bio=0 tech=0")
            log.info("NON-LOC misses:       bio=0 tech=0")
            return

        # --- Allowlist preload (location-only misses from last run)
        STAGE = "map_exchanges"
        cached_allow = (
            self._cache_glue.preload_allow(self, STAGE) if self._cache_glue else {}
        )
        allow_bio = set()
        allow_tec = set()

        def _apply_allow_cache(rem, direction, allow_bucket):
            if not cached_allow:
                return rem
            keep = set()
            _s_act = self._cache_glue._supplier_act_id
            _c_act = self._cache_glue._consumer_act_id
            _s_sig = self._cache_glue._supplier_sig_from_idx
            _c_sig = self._cache_glue._consumer_sig_from_idx
            for s, c in rem:
                sigs = cached_allow.get(
                    (direction, _s_act(self, s, direction), _c_act(self, c))
                )
                if not sigs:
                    keep.add((s, c))
                    continue
                if sigs == (_s_sig(self, s, direction), _c_sig(self, c)):
                    allow_bucket.add((s, c))  # still location-only near-miss
                else:
                    keep.add((s, c))
            return keep

        rem_bio = _apply_allow_cache(rem_bio, DIR_BIO, allow_bio)
        rem_tec = _apply_allow_cache(rem_tec, DIR_TECH, allow_tec)
        log.info(
            "REM after %s-allow: bio=%d tech=%d | allow_bio=%d allow_tec=%d",
            STAGE,
            len(rem_bio),
            len(rem_tec),
            len(allow_bio),
            len(allow_tec),
        )

        # ---- EARLY RETURN 3: only allowlist edges remain → skip CF loop
        if not rem_bio and not rem_tec:
            self.eligible_edges_for_next_bio = allow_bio
            self.eligible_edges_for_next_tech = allow_tec
            written = self._cache_save_new()
            self.logger.info(
                "Cache saved %d newly characterized positions (map_exchanges).",
                written or 0,
            )
            log.info("CF loop skipped: only location-only edges (allowlist) remain.")
            log.info("REM final (unmatched): bio=0 tech=0")
            log.info(
                "ALLOW (loc-only):     bio=%d tech=%d", len(allow_bio), len(allow_tec)
            )
            log.info("NON-LOC misses:       bio=0 tech=0")
            return

        # -------- indices & hot locals -------------------------------------------
        supplier_index_bio = build_index(
            self.supplier_lookup_bio, self.required_supplier_fields
        )
        supplier_index_tec = build_index(
            self.supplier_lookup_tech, self.required_supplier_fields
        )
        consumer_index = build_index(
            self.consumer_lookup, self.required_consumer_fields
        )

        consumer_lookup = self.consumer_lookup
        reversed_consumer_lookup = self.reversed_consumer_lookup

        # allowlists we will output from this pass
        allow_bio = set()
        allow_tec = set()

        # req field tuples (no 'classifications')
        req_sup_nc = getattr(self, "_req_sup_nc", None)
        if req_sup_nc is None:
            self._req_sup_nc = tuple(
                sorted(
                    k for k in self.required_supplier_fields if k != "classifications"
                )
            )
            self._req_con_nc = tuple(
                sorted(
                    k for k in self.required_consumer_fields if k != "classifications"
                )
            )
        req_sup_nc = self._req_sup_nc
        req_con_nc = self._req_con_nc

        # batching/progressive save
        BATCH_SAVE = 10_000  # positions
        positions_since_save = 0

        def _maybe_flush():
            nonlocal positions_since_save
            if positions_since_save >= BATCH_SAVE:
                written = self._cache_save_new()
                if written:
                    import gc

                    gc.collect()
                positions_since_save = 0

        def get_dir_bundle(supplier_matrix: str):
            if supplier_matrix == "biosphere":
                return (
                    DIR_BIO,
                    rem_bio,
                    ebs_bio,
                    ebc_bio,
                    supplier_index_bio,
                    self.supplier_lookup_bio,
                    self.reversed_supplier_lookup_bio,
                )
            else:
                return (
                    DIR_TECH,
                    rem_tec,
                    ebs_tec,
                    ebc_tec,
                    supplier_index_tec,
                    self.supplier_lookup_tech,
                    self.reversed_supplier_lookup_tech,
                )

        def _short(d, limit=180):
            try:
                s = str(d)
            except Exception:
                s = repr(d)
            return s if len(s) <= limit else s[: limit - 1] + "…"

        def _count_none(x):
            return 0 if x is None else (len(x) if hasattr(x, "__len__") else 1)

        log.debug(
            "START map_exchanges | biosphere_edges=%d | technosphere_edges=%d | CFs=%d | req_supplier=%s | req_consumer=%s",
            len(self.biosphere_edges),
            len(self.technosphere_edges),
            len(self.raw_cfs_data),
            sorted(self.required_supplier_fields),
            sorted(self.required_consumer_fields),
        )
        log.debug(
            "Lookups | supplier_bio=%d keys | supplier_tech=%d keys | consumer=%d keys",
            len(self.supplier_lookup_bio),
            len(self.supplier_lookup_tech),
            len(self.consumer_lookup),
        )

        matched_positions_total = 0
        allow_bio_added = 0
        allow_tec_added = 0

        # -------- main CF loop (chunked emission) --------------------------------
        for i, cf in enumerate(tqdm(self.raw_cfs_data, desc="Mapping exchanges")):
            s_crit = cf["supplier"]
            c_crit = cf["consumer"]

            dir_name, rem, ebs, ebc, s_index, s_lookup, s_reversed = get_dir_bundle(
                s_crit.get("matrix", "biosphere")
            )
            if not rem:
                log.debug("CF[%d] dir=%s skipped: no remaining edges.", i, dir_name)
                continue

            # ---------- SUPPLIER side ----------
            if "classifications" in s_crit:
                s_class_hits = _cls_candidates_from_cf(
                    s_crit["classifications"],
                    (
                        self.cls_prefidx_supplier_bio
                        if dir_name == DIR_BIO
                        else self.cls_prefidx_supplier_tech
                    ),
                    adjacency_keys=set(ebs.keys()),
                )
            else:
                s_class_hits = None

            cached_match_with_index.index = s_index
            cached_match_with_index.lookup_mapping = s_lookup
            cached_match_with_index.reversed_lookup = s_reversed

            s_nonclass = {k: v for k, v in s_crit.items() if k != "classifications"}
            s_out = cached_match_with_index(make_hashable(s_nonclass), req_sup_nc)

            # produce minimal candidate containers
            s_matches_raw = tuple(s_out.matches)
            if s_class_hits is not None:
                s_cands = [s for s in s_matches_raw if s in s_class_hits]
            else:
                s_cands = list(s_matches_raw)
            s_cands = [s for s in s_cands if s in ebs]

            s_loc_only = set(s_out.location_only_rejects)
            if s_class_hits is not None:
                s_loc_only &= set(s_class_hits)
            s_loc_required = ("location" in s_crit) and (
                s_crit.get("location") is not None
            )

            # ---------- CONSUMER side ----------
            if "classifications" in c_crit:
                c_class_hits = _cls_candidates_from_cf(
                    c_crit["classifications"],
                    self.cls_prefidx_consumer,
                    adjacency_keys=set(ebc.keys()),
                )
            else:
                c_class_hits = None

            cached_match_with_index.index = consumer_index
            cached_match_with_index.lookup_mapping = consumer_lookup
            cached_match_with_index.reversed_lookup = reversed_consumer_lookup

            c_nonclass = {k: v for k, v in c_crit.items() if k != "classifications"}
            c_out = cached_match_with_index(make_hashable(c_nonclass), req_con_nc)

            c_matches_raw = tuple(c_out.matches)
            if c_class_hits is not None:
                c_cands = [c for c in c_matches_raw if c in c_class_hits]
            else:
                c_cands = list(c_matches_raw)
            c_cands = [c for c in c_cands if c in ebc]

            c_loc_only = set(c_out.location_only_rejects)
            if c_class_hits is not None:
                c_loc_only &= set(c_class_hits)
            c_loc_required = ("location" in c_crit) and (
                c_crit.get("location") is not None
            )

            # ---- DEBUG: empty reasons
            if not s_cands:
                reason = []
                if not s_matches_raw:
                    reason.append("no-index-match")
                else:
                    reason.append(f"raw-matches={len(s_matches_raw)}")
                    if s_class_hits is not None and not (
                        set(s_matches_raw) & set(s_class_hits)
                    ):
                        reason.append("class-filtered-out")
                    if s_class_hits is None:
                        reason.append("no-class-filter")
                    pruned = [s for s in s_matches_raw if s not in ebs]
                    if pruned and len(pruned) == len(s_matches_raw):
                        reason.append("all-pruned-by-adjacency")
                log.debug(
                    "CF[%d] dir=%s supplier candidates empty | reasons=%s | s_crit=%s | raw=%d class_hits=%s ebs_keys=%d",
                    i,
                    dir_name,
                    ",".join(reason),
                    _short(s_crit),
                    len(s_matches_raw),
                    _count_none(s_class_hits),
                    len(ebs),
                )

            if not c_cands:
                reason = []
                if not c_matches_raw:
                    reason.append("no-index-match")
                else:
                    reason.append(f"raw-matches={len(c_matches_raw)}")
                    if c_class_hits is not None and not (
                        set(c_matches_raw) & set(c_class_hits)
                    ):
                        reason.append("class-filtered-out")
                    if c_class_hits is None:
                        reason.append("no-class-filter")
                    pruned = [c for c in c_matches_raw if c not in ebc]
                    if pruned and len(pruned) == len(c_matches_raw):
                        reason.append("all-pruned-by-adjacency")
                log.debug(
                    "CF[%d] dir=%s consumer candidates empty | reasons=%s | c_crit=%s | raw=%d class_hits=%s ebc_keys=%d",
                    i,
                    dir_name,
                    ",".join(reason),
                    _short(c_crit),
                    len(c_matches_raw),
                    _count_none(c_class_hits),
                    len(ebc),
                )

            # ---------- Combine matches (chunked) ----------
            if s_cands and c_cands:
                # Use sets for O(1) membership & avoid dict.get
                s_cand_set = set(s_cands)
                c_cand_set = set(c_cands)

                # Only consider suppliers that actually have outgoing edges
                # NOTE: dict.__contains__ checks keys; intersect directly with the dict itself.
                eligible_s = s_cand_set.intersection(
                    ebs
                )  # ebs is a dict; iterates keys

                CHUNK = 4096
                buf = []
                add_buf = buf.append

                # Fast locals
                ebs_local = ebs
                rem_local = rem
                ebc_local = ebc
                add_cf = add_cf_entry

                for s in eligible_s:
                    cs = ebs_local[s]  # 1 dict indexing, not .get()
                    hit = cs & c_cand_set  # set intersection (fast in C)
                    if not hit:
                        continue
                    for c in hit:
                        add_buf((s, c))
                        if len(buf) >= CHUNK:
                            add_cf(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=s_crit,
                                consumer_info=c_crit,
                                direction=dir_name,
                                indices=buf,
                                value=cf["value"],
                                uncertainty=cf.get("uncertainty"),
                            )
                            matched_positions_total += len(buf)
                            positions_since_save += len(buf)

                            # prune matched edges quickly without .get
                            for _s, _c in buf:
                                if (_s, _c) in rem_local:
                                    rem_local.remove((_s, _c))
                                    ebs_local[_s].discard(_c)
                                    if not ebs_local[_s]:
                                        del ebs_local[_s]
                                    ebc_local[_c].discard(_s)
                                    if not ebc_local[_c]:
                                        del ebc_local[_c]
                            buf = []
                            add_buf = buf.append
                            _maybe_flush()

                # flush tail
                if buf:
                    add_cf(
                        cfs_mapping=self.cfs_mapping,
                        supplier_info=s_crit,
                        consumer_info=c_crit,
                        direction=dir_name,
                        indices=buf,
                        value=cf["value"],
                        uncertainty=cf.get("uncertainty"),
                    )
                    matched_positions_total += len(buf)
                    positions_since_save += len(buf)
                    for _s, _c in buf:
                        if (_s, _c) in rem_local:
                            rem_local.remove((_s, _c))
                            ebs_local[_s].discard(_c)
                            if not ebs_local[_s]:
                                del ebs_local[_s]
                            ebc_local[_c].discard(_s)
                            if not ebc_local[_c]:
                                del ebc_local[_c]
                    buf = []
                    _maybe_flush()
            else:
                log.debug(
                    "CF[%d] dir=%s NO-MATCH | s_cands=%d c_cands=%d | s_loc_only=%d c_loc_only=%d | rem=%d",
                    i,
                    dir_name,
                    len(s_cands),
                    len(c_cands),
                    len(s_loc_only),
                    len(c_loc_only),
                    len(rem),
                )

            # ---------- Near-miss allowlists (location-only) ----------
            # supplier near-miss with consumer full matches
            if s_loc_required and s_loc_only and c_cands:
                cset = set(c_cands)
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for s in s_loc_only:
                    cs = ebs.get(s)
                    if not cs:
                        continue
                    hit = cs & cset
                    if hit:
                        for c in hit:
                            if (s, c) in rem:
                                bucket.add((s, c))
                                added += 1
                if added:
                    if dir_name == DIR_BIO:
                        allow_bio_added += added
                    else:
                        allow_tec_added += added

            # consumer near-miss with supplier full matches
            if c_loc_required and c_loc_only and s_cands:
                sset = set(s_cands)
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for c in c_loc_only:
                    ss = ebc.get(c)
                    if not ss:
                        continue
                    hit = ss & sset
                    if hit:
                        for s in hit:
                            if (s, c) in rem:
                                bucket.add((s, c))
                                added += 1
                if added:
                    if dir_name == DIR_BIO:
                        allow_bio_added += added
                    else:
                        allow_tec_added += added

            # both sides near-miss
            if s_loc_required and c_loc_required and s_loc_only and c_loc_only:
                cset = set(c_loc_only)
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for s in s_loc_only:
                    cs = ebs.get(s)
                    if not cs:
                        continue
                    hit = cs & cset
                    if hit:
                        for c in hit:
                            if (s, c) in rem:
                                bucket.add((s, c))
                                added += 1
                if added:
                    if dir_name == DIR_BIO:
                        allow_bio_added += added
                    else:
                        allow_tec_added += added

        # final flush (if any)
        if positions_since_save:
            written = self._cache_save_new()
            if written:
                import gc

                gc.collect()

        # -------- wrap up ---------------------------------------------------------
        self._update_unprocessed_edges()

        # store per-direction allowlists for later passes
        self.eligible_edges_for_next_bio = allow_bio
        self.eligible_edges_for_next_tech = allow_tec

        written = self._cache_save_new()
        if written:
            self.logger.info(
                "Cache saved %d newly characterized positions (map_exchanges).", written
            )
        else:
            self.logger.info("No new cache entries to save (map_exchanges).")

        log.info("REM final (unmatched): bio=%d tech=%d", len(rem_bio), len(rem_tec))
        log.info("ALLOW (loc-only):     bio=%d tech=%d", len(allow_bio), len(allow_tec))
        log.info(
            "NON-LOC misses:       bio=%d tech=%d",
            len(rem_bio - allow_bio),
            len(rem_tec - allow_tec),
        )

        # ---- NEGATIVE CACHE: save only non-location misses ----------------------
        if self._cache_glue:
            nonloc = []
            for s, c in rem_bio - allow_bio:
                nonloc.append((DIR_BIO, s, c))
            for s, c in rem_tec - allow_tec:
                nonloc.append((DIR_TECH, s, c))
            saved_neg = self._cache_glue.save_negative(self, nonloc)
            log.info("NEGATIVE cache saved: %d non-location misses", saved_neg)

            stage_allow = []
            for s, c in allow_bio:
                stage_allow.append((DIR_BIO, s, c))
            for s, c in allow_tec:
                stage_allow.append((DIR_TECH, s, c))
            saved = self._cache_glue.save_allow(self, STAGE, stage_allow)
            log.info("ALLOW cache saved for %s: %d entries", STAGE, saved)

    def map_aggregate_locations(self) -> None:
        """
        Map unmatched exchanges using CFs from broader (aggregated) regions.

        Memory-minded version: chunked index emission + progressive cache flush.
        """

        import gc
        from collections import defaultdict
        from tqdm import tqdm

        self._initialize_weights()
        logger.info("Handling static regions…")

        STAGE = "map_aggregate_locations"

        # --- Positive preload (restores prior results for THIS stage)
        pre_len = len(self.cfs_mapping)
        restored = self._cache_preload_if_any()
        post_len = len(self.cfs_mapping)
        if restored:
            pos_from_cache = [
                p
                for cf in self.cfs_mapping[pre_len:post_len]
                for p in cf.get("positions", ())
            ]
            logger.info(
                "CACHE PRELOAD (%s): entries=%d positions=%d sample=%s",
                STAGE,
                post_len - pre_len,
                len(pos_from_cache),
                pos_from_cache[:5],
            )

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # --- batching/progressive cache save
        BATCH_SAVE = 10_000  # positions
        CHUNK = 4096  # indices chunk per add_cf_entry
        positions_since_save = 0

        def _maybe_flush():
            nonlocal positions_since_save
            if positions_since_save >= BATCH_SAVE:
                written = self._cache_save_new()
                if written:
                    gc.collect()
                positions_since_save = 0

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

            # Pick the correct reversed supplier dict for this direction
            rev_sup = (
                self.reversed_supplier_lookup_bio
                if direction == "biosphere-technosphere"
                else self.reversed_supplier_lookup_tech
            )

            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )
            processed_flows = set(processed_flows)

            # keep only edges this stage can handle
            allowed = (
                self.eligible_edges_for_next_bio
                if direction == "biosphere-technosphere"
                else self.eligible_edges_for_next_tech
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

            # --- NEGATIVE cache prune (global, non-location misses)
            if getattr(self, "_cache_glue", None):
                neg = self._cache_glue.preload_negative(self)
                if neg:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = neg.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        # skip only if signatures still match exactly
                        if sigs != (s_sig, c_sig):
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            # --- STAGE allowlist preload
            stage_allow = set()
            if getattr(self, "_cache_glue", None):
                allow_rows = self._cache_glue.preload_allow(self, STAGE)
                if allow_rows:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = allow_rows.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        if sigs == (s_sig, c_sig):
                            stage_allow.add((s, c))  # still valid to pass onward
                        else:
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            logger.info(
                "ALLOW PRELOAD (%s): direction=%s | allow=%d | remaining=%d",
                STAGE,
                direction,
                len(stage_allow),
                len(unprocessed_edges),
            )

            # Early exit per direction
            if not unprocessed_edges:
                if direction == "biosphere-technosphere":
                    self.eligible_edges_for_next_bio = stage_allow
                else:
                    self.eligible_edges_for_next_tech = stage_allow
                continue

            # --- Build pairs grouped by (consumer_loc, supplier_loc)
            edges_index = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue
                consumer_loc = self.consumer_loc.get(consumer_idx)
                if not consumer_loc:
                    raise ValueError(
                        f"Consumer flow {consumer_idx} has no 'location' field. "
                        "Ensure all consumer flows have a valid location."
                    )
                supplier_loc = (
                    self.supplier_loc_bio.get(supplier_idx)
                    if direction == "biosphere-technosphere"
                    else self.supplier_loc_tech.get(supplier_idx)
                )
                edges_index[(consumer_loc, supplier_loc)].append(
                    (supplier_idx, consumer_idx)
                )

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            # --- Build candidate location sets & partition into prefiltered/remaining
            for (consumer_location, supplier_location), edges in edges_index.items():
                # skip dynamic placeholders here; other stages handle them
                if any(
                    x in ("RoW", "RoE") for x in (consumer_location, supplier_location)
                ):
                    continue

                if supplier_location is None:
                    candidate_suppliers_locations = ["__ANY__"]
                else:
                    candidate_suppliers_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=supplier_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        supplier=True,
                    )
                    if not candidate_suppliers_locations:
                        candidate_suppliers_locations = [supplier_location]

                if consumer_location is None:
                    candidate_consumer_locations = ["__ANY__"]
                else:
                    candidate_consumer_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=consumer_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        supplier=False,
                    )
                    if not candidate_consumer_locations:
                        candidate_consumer_locations = [consumer_location]

                # If neither is composite, let later stages handle
                if (
                    len(candidate_suppliers_locations) == 1
                    and len(candidate_consumer_locations) == 1
                ):
                    continue

                for supplier_idx, consumer_idx in edges:
                    supplier_info = rev_sup[supplier_idx]
                    consumer_info = self._get_consumer_info(consumer_idx)

                    sig = _equality_supplier_signature_cached(
                        make_hashable(supplier_info)
                    )
                    if sig in self._cached_supplier_keys:
                        prefiltered_groups[sig].append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_suppliers_locations,
                                candidate_consumer_locations,
                            )
                        )
                    else:
                        if any(op in cf_operators for op in ["contains", "startswith"]):
                            remaining_edges.append(
                                (
                                    supplier_idx,
                                    consumer_idx,
                                    supplier_info,
                                    consumer_info,
                                    candidate_suppliers_locations,
                                    candidate_consumer_locations,
                                )
                            )

            del edges_index  # free
            gc.collect()

            # Helper: emit many (i,j) in chunks with progressive cache flush
            def _emit_group(
                direction, supplier_info, consumer_info, indices, value, unc
            ):
                nonlocal positions_since_save
                if not indices:
                    return
                # chunk indices so we don't create huge add_cf_entry payloads
                for k in range(0, len(indices), CHUNK):
                    chunk = indices[k : k + CHUNK]
                    add_cf_entry(
                        cfs_mapping=self.cfs_mapping,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        direction=direction,
                        indices=chunk,
                        value=value,
                        uncertainty=unc,
                    )
                    positions_since_save += len(chunk)
                    _maybe_flush()

            # ---------------- Pass 1 (prefiltered) --------------------------------
            if prefiltered_groups:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(), desc="Processing static groups (pass 1)"
                ):
                    supplier_info = group_edges[0][2]
                    consumer_info = group_edges[0][3]
                    candidate_supplier_locations = group_edges[0][-2]
                    candidate_consumer_locations = group_edges[0][-1]

                    new_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                        candidate_suppliers=candidate_supplier_locations,
                        candidate_consumers=candidate_consumer_locations,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        cf_index=self.cf_index,
                    )

                    if new_cf != 0:
                        # collect indices for this signature and emit chunked
                        idx_buf = [(s_i, c_i) for (s_i, c_i, *_rest) in group_edges]
                        _emit_group(
                            direction,
                            supplier_info,
                            consumer_info,
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            supplier_info,
                            consumer_info,
                            candidate_supplier_locations,
                            candidate_consumer_locations,
                        )

            del prefiltered_groups
            gc.collect()

            # ---------------- Pass 2 (fallback) -----------------------------------
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=remaining_edges,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
            )

            del remaining_edges
            gc.collect()

            if grouped_edges:
                for (
                    s_key,
                    c_key,
                    (candidate_suppliers, candidate_consumers),
                ), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing static groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        s_key, c_key, candidate_suppliers, candidate_consumers
                    )
                    if new_cf != 0:
                        idx_buf = [(si, cj) for (si, cj) in edge_group]
                        _emit_group(
                            direction,
                            dict(s_key),
                            dict(c_key),
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            s_key,
                            c_key,
                            candidate_suppliers,
                            candidate_consumers,
                        )

            del grouped_edges
            gc.collect()

            # Any pairs still unprocessed at the end of this direction that we marked as
            # stage-allow earlier should be handed to the next stage
            if direction == "biosphere-technosphere":
                self.eligible_edges_for_next_bio = stage_allow
            else:
                self.eligible_edges_for_next_tech = stage_allow

        # final flush (if any)
        if positions_since_save:
            written = self._cache_save_new()
            if written:
                gc.collect()

        self._update_unprocessed_edges()

        written = self._cache_save_new()
        if written:
            self.logger.info(
                "Cache saved %d newly characterized positions (%s).", written, STAGE
            )
        else:
            self.logger.info("No new cache entries to save (%s).", STAGE)

        # Persist this stage's allowlist (for early pruning next run)
        if getattr(self, "_cache_glue", None):
            to_save = []
            for s, c in getattr(self, "eligible_edges_for_next_bio", set()):
                to_save.append(("biosphere-technosphere", s, c))
            for s, c in getattr(self, "eligible_edges_for_next_tech", set()):
                to_save.append(("technosphere-technosphere", s, c))
            saved = self._cache_glue.save_allow(self, STAGE, to_save)
            logger.info("ALLOW cache saved for %s: %d entries", STAGE, saved)

    def map_dynamic_locations(self) -> None:
        """
        Memory-minded dynamic/relative region mapping (RoW/RoE).

        - Restores prior results (positive cache)
        - Prunes with negative cache (non-location misses)
        - Uses stage allowlist to skip heavy grouping when possible
        - Emits indices in chunks + progressive cache flush to keep RAM steady
        """

        import gc
        from collections import defaultdict
        from tqdm import tqdm

        self._initialize_weights()
        logger.info("Handling dynamic regions…")

        STAGE = "map_dynamic_locations"

        # ---- Positive preload (proof only)
        pre_len = len(self.cfs_mapping)
        self._cache_preload_if_any()
        post_len = len(self.cfs_mapping)
        if post_len > pre_len:
            pos_from_cache = [
                p
                for cf in self.cfs_mapping[pre_len:post_len]
                for p in cf.get("positions", ())
            ]
            logger.info(
                "CACHE PRELOAD (%s): entries=%d positions=%d sample=%s",
                STAGE,
                post_len - pre_len,
                len(pos_from_cache),
                pos_from_cache[:5],
            )

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # Build exclusions once (for RoW/RoE decomposition)
        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        raw_exclusion_locs = {
            loc
            for locs in self.technosphere_flows_lookup.values()
            for loc in locs
            if loc not in ["RoW", "RoE"]
        }
        decomposed_exclusions = self.geo.batch(
            locations=list(raw_exclusion_locs), containing=True
        )
        decomposed_exclusions = frozenset(
            (k, tuple(v)) for k, v in decomposed_exclusions.items()
        )

        # batching / progressive cache save
        BATCH_SAVE = 10_000  # positions
        CHUNK = 4096  # indices per add_cf_entry chunk
        positions_since_save = 0

        def _maybe_flush():
            nonlocal positions_since_save
            if positions_since_save >= BATCH_SAVE:
                written = self._cache_save_new()
                if written:
                    gc.collect()
                positions_since_save = 0

        def _emit_group(direction, supplier_info, consumer_info, indices, value, unc):
            """Emit many (i,j) in chunks and flush progressively."""
            nonlocal positions_since_save
            if not indices:
                return
            for k in range(0, len(indices), CHUNK):
                chunk = indices[k : k + CHUNK]
                add_cf_entry(
                    cfs_mapping=self.cfs_mapping,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    direction=direction,
                    indices=chunk,
                    value=value,
                    uncertainty=unc,
                )
                positions_since_save += len(chunk)
                _maybe_flush()

        # Helper: identify RoW/RoE at either side
        def _is_dynamic_pair(direction, s_idx, c_idx) -> bool:
            s_loc = (
                self.supplier_loc_bio.get(s_idx)
                if direction == "biosphere-technosphere"
                else self.supplier_loc_tech.get(s_idx)
            )
            c_loc = self.consumer_loc.get(c_idx)
            return (s_loc in ("RoW", "RoE")) or (c_loc in ("RoW", "RoE"))

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

            # correct reversed supplier dict for this direction
            rev_sup = (
                self.reversed_supplier_lookup_bio
                if direction == "biosphere-technosphere"
                else self.reversed_supplier_lookup_tech
            )

            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = set(
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # keep only edges this stage should consider (eligible from previous steps)
            allowed_stage_input = (
                set(self.eligible_edges_for_next_bio or [])
                if direction == "biosphere-technosphere"
                else set(self.eligible_edges_for_next_tech or [])
            )
            if allowed_stage_input:
                unprocessed_edges = [
                    e for e in unprocessed_edges if e in allowed_stage_input
                ]

            # --- NEGATIVE cache prune (global, non-location misses)
            if getattr(self, "_cache_glue", None):
                neg = self._cache_glue.preload_negative(self)
                if neg:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = neg.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        if sigs != (s_sig, c_sig):  # metadata changed, re-try
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            # --- STAGE allowlist preload (if previously “allowed here” and sigs unchanged)
            stage_allow = set()
            if getattr(self, "_cache_glue", None):
                allow_rows = self._cache_glue.preload_allow(
                    self, STAGE
                )  # {(dir,s_id,c_id):(s_sig,c_sig)}
                if allow_rows:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = allow_rows.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        if sigs == (s_sig, c_sig):
                            stage_allow.add((s, c))
                        else:
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            # Now prefilter to dynamic pairs only
            dynamic_edges = [
                (s, c)
                for (s, c) in unprocessed_edges
                if (s, c) not in processed_flows and _is_dynamic_pair(direction, s, c)
            ]
            logger.info(
                "ALLOW PRELOAD (%s): direction=%s | allow=%d | dynamic_candidates=%d",
                STAGE,
                direction,
                len(stage_allow),
                len(dynamic_edges),
            )

            # Early exit if nothing to do for this direction
            if not dynamic_edges:
                # keep stage_allow available to next stage
                if direction == "biosphere-technosphere":
                    self.eligible_edges_for_next_bio = stage_allow
                else:
                    self.eligible_edges_for_next_tech = stage_allow
                continue

            # ---------------- Build groups ----------------
            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for supplier_idx, consumer_idx in dynamic_edges:
                consumer_info = self._get_consumer_info(consumer_idx)
                supplier_info = rev_sup[supplier_idx]

                supplier_loc = (
                    self.supplier_loc_bio.get(supplier_idx)
                    if direction == "biosphere-technosphere"
                    else self.supplier_loc_tech.get(supplier_idx)
                )
                consumer_loc = self.consumer_loc.get(consumer_idx)

                dynamic_supplier = supplier_loc in ["RoW", "RoE"]
                dynamic_consumer = consumer_loc in ["RoW", "RoE"]

                suppliers_excluded_subregions = self._extract_excluded_subregions(
                    supplier_idx, decomposed_exclusions
                )
                consumers_excluded_subregions = self._extract_excluded_subregions(
                    consumer_idx, decomposed_exclusions
                )

                # Resolve fallback candidate locations
                if dynamic_supplier:
                    candidate_suppliers_locs = resolve_candidate_locations(
                        geo=self.geo,
                        location="GLO",
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        exceptions=suppliers_excluded_subregions,
                        supplier=True,
                    )
                else:
                    candidate_suppliers_locs = (
                        ["__ANY__"] if supplier_loc is None else [supplier_loc]
                    )

                if dynamic_consumer:
                    candidate_consumers_locs = resolve_candidate_locations(
                        geo=self.geo,
                        location="GLO",
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        exceptions=consumers_excluded_subregions,
                        supplier=False,
                    )
                else:
                    candidate_consumers_locs = (
                        ["__ANY__"] if consumer_loc is None else [consumer_loc]
                    )

                sig = _equality_supplier_signature_cached(make_hashable(supplier_info))
                if sig in self._cached_supplier_keys:
                    prefiltered_groups[sig].append(
                        (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            candidate_suppliers_locs,
                            candidate_consumers_locs,
                        )
                    )
                else:
                    if any(op in cf_operators for op in ["contains", "startswith"]):
                        remaining_edges.append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_suppliers_locs,
                                candidate_consumers_locs,
                            )
                        )

            del dynamic_edges
            gc.collect()

            # ---------------- Pass 1 (prefiltered) ----------------
            if prefiltered_groups:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing dynamic groups (pass 1)",
                ):
                    rep_supplier = group_edges[0][2]
                    rep_consumer = group_edges[0][3]
                    cand_sup_locs = group_edges[0][-2]
                    cand_con_locs = group_edges[0][-1]

                    new_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                        candidate_suppliers=cand_sup_locs,
                        candidate_consumers=cand_con_locs,
                        supplier_info=rep_supplier,
                        consumer_info=rep_consumer,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        cf_index=self.cf_index,
                    )

                    if new_cf:
                        idx_buf = [(s_i, c_i) for (s_i, c_i, *_rest) in group_edges]
                        _emit_group(
                            direction,
                            rep_supplier,
                            rep_consumer,
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            rep_supplier,
                            rep_consumer,
                            cand_sup_locs,
                            cand_con_locs,
                        )

            del prefiltered_groups
            gc.collect()

            # ---------------- Pass 2 (fallback) ----------------
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=remaining_edges,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
            )
            del remaining_edges
            gc.collect()

            if grouped_edges:
                for (s_key, c_key, (cand_sup_locs, cand_con_locs)), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing dynamic groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        s_key, c_key, cand_sup_locs, cand_con_locs
                    )
                    if new_cf:
                        idx_buf = [(si, cj) for (si, cj) in edge_group]
                        _emit_group(
                            direction,
                            dict(s_key),
                            dict(c_key),
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            s_key,
                            c_key,
                            cand_sup_locs,
                            cand_con_locs,
                        )

            del grouped_edges
            gc.collect()

            # keep stage_allow for whatever comes next
            if direction == "biosphere-technosphere":
                self.eligible_edges_for_next_bio = stage_allow
            else:
                self.eligible_edges_for_next_tech = stage_allow

        # final progressive flush (if any)
        if positions_since_save:
            written = self._cache_save_new()
            if written:
                gc.collect()

        self._update_unprocessed_edges()

        written = self._cache_save_new()
        if written:
            self.logger.info(
                "Cache saved %d newly characterized positions (%s).", written, STAGE
            )

        # Persist allowlist for early pruning on future runs
        if getattr(self, "_cache_glue", None):
            to_save = []
            for s, c in getattr(self, "eligible_edges_for_next_bio", set()):
                to_save.append(("biosphere-technosphere", s, c))
            for s, c in getattr(self, "eligible_edges_for_next_tech", set()):
                to_save.append(("technosphere-technosphere", s, c))
            saved = self._cache_glue.save_allow(self, STAGE, to_save)
            logger.info("ALLOW cache saved for %s: %d entries", STAGE, saved)

    def map_contained_locations(self) -> None:
        """
        Resolve unmatched exchanges by assigning CFs from spatially containing regions.
        (Inverse of map_aggregate_locations: use containing regions instead of contained.)
        """

        import gc
        from collections import defaultdict
        from tqdm import tqdm

        self._initialize_weights()
        logger.info("Handling contained locations…")

        STAGE = "map_contained_locations"

        # --- Positive preload (restores prior results from THIS context)
        pre_len = len(self.cfs_mapping)
        self._cache_preload_if_any()
        post_len = len(self.cfs_mapping)
        if post_len > pre_len:
            pos_from_cache = [
                p
                for cf in self.cfs_mapping[pre_len:post_len]
                for p in cf.get("positions", ())
            ]
            logger.info(
                "CACHE PRELOAD (%s): entries=%d positions=%d sample=%s",
                STAGE,
                post_len - pre_len,
                len(pos_from_cache),
                pos_from_cache[:5],
            )

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # Per-direction allowlists we’ll propagate to whatever comes next
        stage_allow_bio: set[tuple[int, int]] = set()
        stage_allow_tec: set[tuple[int, int]] = set()

        # batching / progressive cache save
        BATCH_SAVE = 10_000  # positions
        CHUNK = 4096  # indices per add_cf_entry chunk
        positions_since_save = 0

        def _maybe_flush():
            nonlocal positions_since_save
            if positions_since_save >= BATCH_SAVE:
                written = self._cache_save_new()
                if written:
                    gc.collect()
                positions_since_save = 0

        def _emit_group(direction, supplier_info, consumer_info, indices, value, unc):
            """Emit many (i,j) in chunks and flush progressively."""
            nonlocal positions_since_save
            if not indices:
                return
            for k in range(0, len(indices), CHUNK):
                chunk = indices[k : k + CHUNK]
                add_cf_entry(
                    cfs_mapping=self.cfs_mapping,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    direction=direction,
                    indices=chunk,
                    value=value,
                    uncertainty=unc,
                )
                positions_since_save += len(chunk)
                _maybe_flush()

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            # Pick the correct reversed supplier dict for this direction
            rev_sup = (
                self.reversed_supplier_lookup_bio
                if direction == "biosphere-technosphere"
                else self.reversed_supplier_lookup_tech
            )

            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = set(
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Keep only edges that prior stages explicitly allowed to reach here
            allowed = (
                getattr(self, "eligible_edges_for_next_bio", set())
                if direction == "biosphere-technosphere"
                else getattr(self, "eligible_edges_for_next_tech", set())
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

            # --- NEGATIVE cache prune (global, non-location misses)
            if getattr(self, "_cache_glue", None):
                neg = self._cache_glue.preload_negative(self)
                if neg:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = neg.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        # Skip only if signatures still match exactly; otherwise keep
                        if sigs != (s_sig, c_sig):
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            # --- STAGE allowlist preload (fast-skip edges that we know belong here)
            stage_allow = set()
            if getattr(self, "_cache_glue", None):
                allow_rows = self._cache_glue.preload_allow(
                    self, STAGE
                )  # {(dir,s_id,c_id): (s_sig,c_sig)}
                if allow_rows:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = allow_rows.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        if sigs == (s_sig, c_sig):
                            stage_allow.add(
                                (s, c)
                            )  # remember for next stage, and skip here
                        else:
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            logger.info(
                "ALLOW PRELOAD (%s): direction=%s | allow=%d | remaining=%d",
                STAGE,
                direction,
                len(stage_allow),
                len(unprocessed_edges),
            )

            # Early continue if this direction has nothing left after pruning
            if not unprocessed_edges:
                if direction == "biosphere-technosphere":
                    stage_allow_bio |= stage_allow
                else:
                    stage_allow_tec |= stage_allow
                continue

            # -------- Build candidate groups (contained → containing) -----
            edges_index = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_loc = self.consumer_loc.get(consumer_idx)
                if not consumer_loc:
                    raise ValueError(
                        f"Consumer flow {consumer_idx} has no 'location' field. "
                        "Ensure all consumer flows have a valid location."
                    )

                supplier_loc = (
                    self.supplier_loc_bio.get(supplier_idx)
                    if direction == "biosphere-technosphere"
                    else self.supplier_loc_tech.get(supplier_idx)
                )
                edges_index[(consumer_loc, supplier_loc)].append(
                    (supplier_idx, consumer_idx)
                )

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for (consumer_location, supplier_location), edges in edges_index.items():
                if any(
                    x in ("RoW", "RoE") for x in (consumer_location, supplier_location)
                ):
                    continue

                # Containing regions: set `containing=False` to go "up" in hierarchy
                if supplier_location is None:
                    candidate_suppliers_locations = ["__ANY__"]
                else:
                    candidate_suppliers_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=supplier_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=False,  # contained → containing
                        supplier=True,
                    )

                if consumer_location is None:
                    candidate_consumer_locations = ["__ANY__"]
                else:
                    candidate_consumer_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=consumer_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=False,  # contained → containing
                        supplier=False,
                    )

                # If neither side has a containing path, nothing to do for this group
                if (
                    not candidate_suppliers_locations
                    and not candidate_consumer_locations
                ):
                    continue

                for supplier_idx, consumer_idx in edges:
                    supplier_info = rev_sup[supplier_idx]
                    consumer_info = self._get_consumer_info(consumer_idx)

                    sig = _equality_supplier_signature_cached(
                        make_hashable(supplier_info)
                    )
                    if sig in self._cached_supplier_keys:
                        prefiltered_groups[sig].append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_suppliers_locations,
                                candidate_consumer_locations,
                            )
                        )
                    else:
                        if any(op in cf_operators for op in ["contains", "startswith"]):
                            remaining_edges.append(
                                (
                                    supplier_idx,
                                    consumer_idx,
                                    supplier_info,
                                    consumer_info,
                                    candidate_suppliers_locations,
                                    candidate_consumer_locations,
                                )
                            )

            # Pass 1 (prefiltered)
            if prefiltered_groups:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing contained groups (pass 1)",
                ):
                    supplier_info = group_edges[0][2]
                    consumer_info = group_edges[0][3]
                    cand_sup = group_edges[0][-2]
                    cand_con = group_edges[0][-1]

                    new_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                        candidate_suppliers=cand_sup,
                        candidate_consumers=cand_con,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        cf_index=self.cf_index,
                    )

                    if new_cf:
                        idx_buf = [(si, cj) for (si, cj, *_rest) in group_edges]
                        _emit_group(
                            direction,
                            supplier_info,
                            consumer_info,
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            supplier_info,
                            consumer_info,
                            cand_sup,
                            cand_con,
                        )

            del prefiltered_groups
            gc.collect()

            # Pass 2 (fallback, memoized)
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
            )
            grouped_edges = group_edges_by_signature(
                edge_list=remaining_edges,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
            )
            del remaining_edges
            gc.collect()

            if grouped_edges:
                for (s_key, c_key, (cand_sup, cand_con)), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing contained groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        s_key, c_key, cand_sup, cand_con
                    )
                    if new_cf:
                        idx_buf = [(si, cj) for (si, cj) in edge_group]
                        _emit_group(
                            direction,
                            dict(s_key),
                            dict(c_key),
                            idx_buf,
                            new_cf,
                            agg_uncertainty,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            s_key,
                            c_key,
                            cand_sup,
                            cand_con,
                        )

            del grouped_edges
            gc.collect()

            # Carry over this direction’s stage-allow to next stage
            if direction == "biosphere-technosphere":
                stage_allow_bio |= stage_allow
            else:
                stage_allow_tec |= stage_allow

        # final progressive flush (if any)
        if positions_since_save:
            written = self._cache_save_new()
            if written:
                gc.collect()

        # Refresh processed/unprocessed sets after we added entries
        self._update_unprocessed_edges()

        # Save positive cache rows for new matches
        written = self._cache_save_new()
        if written:
            self.logger.info(
                "Cache saved %d newly characterized positions (%s).", written, STAGE
            )
        else:
            self.logger.info("No new cache entries to save (%s).", STAGE)

        # Persist this stage's allowlist (for early pruning next run)
        if getattr(self, "_cache_glue", None):
            to_save = []
            for s, c in stage_allow_bio:
                to_save.append(("biosphere-technosphere", s, c))
            for s, c in stage_allow_tec:
                to_save.append(("technosphere-technosphere", s, c))
            saved = self._cache_glue.save_allow(self, STAGE, to_save)
            logger.info("ALLOW cache saved for %s: %d entries", STAGE, saved)

        # Expose for the next stage in this run, too
        self.eligible_edges_for_next_bio = stage_allow_bio
        self.eligible_edges_for_next_tech = stage_allow_tec

    def map_remaining_locations_to_global(self) -> None:
        """
        Assign global fallback CFs ("GLO") to exchanges that remain unmatched after all regional mapping steps.
        """

        import gc
        from collections import defaultdict
        from tqdm import tqdm

        self._initialize_weights()
        logger.info("Handling remaining exchanges…")

        STAGE = "map_remaining_locations_to_global"

        # --- Positive preload (restore prior results for this exact context)
        pre_len = len(self.cfs_mapping)
        self._cache_preload_if_any()
        post_len = len(self.cfs_mapping)
        if post_len > pre_len:
            pos_from_cache = [
                p
                for cf in self.cfs_mapping[pre_len:post_len]
                for p in cf.get("positions", ())
            ]
            logger.info(
                "CACHE PRELOAD (%s): entries=%d positions=%d sample=%s",
                STAGE,
                post_len - pre_len,
                len(pos_from_cache),
                pos_from_cache[:5],
            )

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # Resolve candidate locations for GLO once
        global_locations = resolve_candidate_locations(
            geo=self.geo,
            location="GLO",
            weights=frozenset(k for k, v in self.weights.items()),
            containing=True,
        )

        # Per-direction allowlists to propagate (probably empty in this final stage)
        stage_allow_bio: set[tuple[int, int]] = set()
        stage_allow_tec: set[tuple[int, int]] = set()

        # batching / progressive cache save to reduce peak RAM
        BATCH_SAVE = 10_000  # positions threshold to flush cache
        CHUNK = 4096  # indices per add_cf_entry chunk
        positions_since_save = 0

        def _maybe_flush():
            nonlocal positions_since_save
            if positions_since_save >= BATCH_SAVE:
                written = self._cache_save_new()
                if written:
                    gc.collect()
                positions_since_save = 0

        def _emit_group(direction, supplier_info, consumer_info, indices, value, unc):
            """Emit many (i,j) in chunks and flush progressively."""
            nonlocal positions_since_save
            if not indices:
                return
            for k in range(0, len(indices), CHUNK):
                chunk = indices[k : k + CHUNK]
                add_cf_entry(
                    cfs_mapping=self.cfs_mapping,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    direction=direction,
                    indices=chunk,
                    value=value,
                    uncertainty=unc,
                )
                positions_since_save += len(chunk)
                _maybe_flush()

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

            # Pick the correct reversed supplier dict for this direction
            rev_sup = (
                self.reversed_supplier_lookup_bio
                if direction == "biosphere-technosphere"
                else self.reversed_supplier_lookup_tech
            )

            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = set(
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Keep only edges that earlier stages explicitly allowed to reach here
            allowed = (
                getattr(self, "eligible_edges_for_next_bio", set())
                if direction == "biosphere-technosphere"
                else getattr(self, "eligible_edges_for_next_tech", set())
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

            # --- NEGATIVE cache prune (global, non-location misses)
            if getattr(self, "_cache_glue", None):
                neg = self._cache_glue.preload_negative(self)
                if neg:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = neg.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        # Skip only if signatures still match exactly; otherwise keep
                        if sigs != (s_sig, c_sig):
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            # --- STAGE allowlist preload (fast-skip edges already destined for this stage)
            stage_allow = set()
            if getattr(self, "_cache_glue", None):
                allow_rows = self._cache_glue.preload_allow(
                    self, STAGE
                )  # {(dir,s_id,c_id): (s_sig,c_sig)}
                if allow_rows:
                    keep = []
                    for s, c in unprocessed_edges:
                        s_id = self._cache_glue._supplier_act_id(self, s, direction)
                        c_id = self._cache_glue._consumer_act_id(self, c)
                        sigs = allow_rows.get((direction, s_id, c_id))
                        if not sigs:
                            keep.append((s, c))
                            continue
                        s_sig = self._cache_glue._supplier_sig_from_idx(
                            self, s, direction
                        )
                        c_sig = self._cache_glue._consumer_sig_from_idx(self, c)
                        if sigs == (s_sig, c_sig):
                            stage_allow.add(
                                (s, c)
                            )  # remember; skip heavy work for these
                        else:
                            keep.append((s, c))
                    unprocessed_edges = keep
                    del keep

            logger.info(
                "ALLOW PRELOAD (%s): direction=%s | allow=%d | remaining=%d",
                STAGE,
                direction,
                len(stage_allow),
                len(unprocessed_edges),
            )

            # Early continue if nothing left after pruning
            if not unprocessed_edges:
                if direction == "biosphere-technosphere":
                    stage_allow_bio |= stage_allow
                else:
                    stage_allow_tec |= stage_allow
                continue

            # -------- Build candidate groups --------
            edges_index = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_loc = self.consumer_loc.get(consumer_idx)
                if not consumer_loc:
                    raise ValueError(
                        f"Consumer flow {consumer_idx} has no 'location' field. "
                        "Ensure all consumer flows have a valid location."
                    )

                supplier_loc = (
                    self.supplier_loc_bio.get(supplier_idx)
                    if direction == "biosphere-technosphere"
                    else self.supplier_loc_tech.get(supplier_idx)
                )

                edges_index[(consumer_loc, supplier_loc)].append(
                    (supplier_idx, consumer_idx)
                )

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for (consumer_location, supplier_location), edges in edges_index.items():
                candidate_suppliers_locations = (
                    ["__ANY__"] if supplier_location is None else global_locations
                )
                candidate_consumers_locations = (
                    ["__ANY__"] if consumer_location is None else global_locations
                )

                for supplier_idx, consumer_idx in edges:
                    supplier_info = rev_sup[supplier_idx]
                    consumer_info = self._get_consumer_info(consumer_idx)

                    sig = _equality_supplier_signature_cached(
                        make_hashable(supplier_info)
                    )
                    if sig in self._cached_supplier_keys:
                        prefiltered_groups[sig].append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_suppliers_locations,
                                candidate_consumers_locations,
                            )
                        )
                    else:
                        if any(op in cf_operators for op in ["contains", "startswith"]):
                            remaining_edges.append(
                                (
                                    supplier_idx,
                                    consumer_idx,
                                    supplier_info,
                                    consumer_info,
                                    candidate_suppliers_locations,
                                    candidate_consumers_locations,
                                )
                            )

            # Pass 1 (prefiltered)
            if prefiltered_groups:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(), desc="Processing global groups (pass 1)"
                ):
                    supplier_info = group_edges[0][2]
                    consumer_info = group_edges[0][3]

                    new_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                        candidate_suppliers=global_locations,
                        candidate_consumers=global_locations,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        cf_index=self.cf_index,
                    )
                    unc = (
                        agg_uncertainty
                        if agg_uncertainty is not None
                        else (
                            matched_cf_obj.get("uncertainty")
                            if matched_cf_obj
                            else None
                        )
                    )

                    if new_cf:
                        idx_buf = [(si, cj) for (si, cj, *_rest) in group_edges]
                        _emit_group(
                            direction,
                            supplier_info,
                            consumer_info,
                            idx_buf,
                            new_cf,
                            unc,
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            supplier_info,
                            consumer_info,
                            global_locations,
                            global_locations,
                        )

            del prefiltered_groups
            gc.collect()

            # Pass 2 (fallback, memoized)
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
            )
            grouped_edges = group_edges_by_signature(
                edge_list=remaining_edges,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
            )
            del remaining_edges
            gc.collect()

            if grouped_edges:
                for (s_key, c_key, (cand_sup, cand_con)), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing global groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        s_key, c_key, cand_sup, cand_con
                    )
                    unc = (
                        agg_uncertainty
                        if agg_uncertainty is not None
                        else (
                            matched_cf_obj.get("uncertainty")
                            if matched_cf_obj
                            else None
                        )
                    )
                    if new_cf:
                        idx_buf = [(si, cj) for (si, cj) in edge_group]
                        _emit_group(
                            direction, dict(s_key), dict(c_key), idx_buf, new_cf, unc
                        )
                        del idx_buf
                    else:
                        self.logger.warning(
                            "Fallback CF could not be computed for supplier=%s, consumer=%s "
                            "with candidate suppliers=%s and consumers=%s",
                            s_key,
                            c_key,
                            cand_sup,
                            cand_con,
                        )

            del grouped_edges
            gc.collect()

            # Propagate per-direction stage allow (kept for symmetry / future stages)
            if direction == "biosphere-technosphere":
                stage_allow_bio |= stage_allow
            else:
                stage_allow_tec |= stage_allow

        # final progressive flush (if any)
        if positions_since_save:
            written = self._cache_save_new()
            if written:
                gc.collect()

        self._update_unprocessed_edges()

        written = self._cache_save_new()
        if written:
            self.logger.info(
                "Cache saved %d newly characterized positions (%s).", written, STAGE
            )
        else:
            self.logger.info("No new cache entries to save (%s).", STAGE)

        # Persist this stage's allowlist (for early pruning next run)
        if getattr(self, "_cache_glue", None):
            to_save = []
            for s, c in stage_allow_bio:
                to_save.append(("biosphere-technosphere", s, c))
            for s, c in stage_allow_tec:
                to_save.append(("technosphere-technosphere", s, c))
            saved = self._cache_glue.save_allow(self, STAGE, to_save)
            logger.info("ALLOW cache saved for %s: %d entries", STAGE, saved)

        # Expose for completeness (harmless if unused)
        self.eligible_edges_for_next_bio = stage_allow_bio
        self.eligible_edges_for_next_tech = stage_allow_tec

    def evaluate_cfs(self, scenario_idx: str | int = 0, scenario=None):
        """
        Evaluate the characterization factors (CFs) based on expressions, parameters, and uncertainty.

        This step computes the numeric CF values that will populate the characterization matrix.

        Depending on the method and configuration, it supports:
        - Symbolic CFs (e.g., "28 * (1 + 0.01 * (co2ppm - 410))")
        - Scenario-based parameter substitution
        - Uncertainty propagation via Monte Carlo simulation

        Parameters
        ----------
        scenario_idx : str or int, optional
            The scenario index (or year) for time/parameter-dependent evaluation. Defaults to 0.
        scenario : str, optional
            Name of the scenario to evaluate (overrides the default one set in `__init__`).

        Behavior
        --------
        - If `use_distributions=True` and `iterations > 1`, a 3D sparse matrix is created
          (i, j, k) where k indexes Monte Carlo iterations.
        - If symbolic expressions are present, they are resolved using the parameter set
          for the selected scenario and year.
        - If deterministic, builds a 2D matrix with direct values.

        Notes
        -----
        - Must be called before `lcia()` to populate the CF matrix.
        - Parameters are pulled from the method file or passed manually via `parameters`.


        Raises
        ------
        ValueError
            If the requested scenario is not found in the parameter dictionary.


        Updates
        -------
        - Sets `characterization_matrix`
        - Populates `scenario_cfs` with resolved CFs

        :return: None
        """

        if self.use_distributions and self.iterations > 1:
            coords_i, coords_j, coords_k = [], [], []
            data = []
            sample_cache = {}

            for cf in self.cfs_mapping:

                # Build a hashable key that uniquely identifies
                # the distribution definition
                key = make_distribution_key(cf)

                if key is None:
                    samples = sample_cf_distribution(
                        cf=cf,
                        n=self.iterations,
                        parameters=self.parameters,
                        random_state=self.random_state,  # can reuse global RNG
                        use_distributions=self.use_distributions,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                elif key in sample_cache:
                    samples = sample_cache[key]
                else:
                    rng = get_rng_for_key(key, self.random_seed)
                    samples = sample_cf_distribution(
                        cf=cf,
                        n=self.iterations,
                        parameters=self.parameters,
                        random_state=rng,
                        use_distributions=self.use_distributions,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                    sample_cache[key] = samples

                neg = (cf.get("uncertainty") or {}).get("negative", 0)
                if neg == 1:
                    samples = -samples

                for i, j in cf["positions"]:
                    for k in range(self.iterations):
                        coords_i.append(i)
                        coords_j.append(j)
                        coords_k.append(k)
                        data.append(samples[k])

            matrix_type = (
                "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
            )
            n_rows, n_cols = (
                self.lca.inventory.shape
                if matrix_type == "biosphere"
                else self.lca.technosphere_matrix.shape
            )

            # Sort all (i, j, k) indices to ensure consistent iteration ordering
            coords = np.array([coords_i, coords_j, coords_k])
            data = np.array(data)

            # Lexicographic sort by i, j, k
            order = np.lexsort((coords[2], coords[1], coords[0]))
            coords = coords[:, order]
            data = data[order]

            self.characterization_matrix = sparse.COO(
                coords=coords,
                data=data,
                shape=(n_rows, n_cols, self.iterations),
            )

            self.scenario_cfs = [{"positions": [], "value": 0}]  # dummy

        else:
            # Fallback to 2D
            self.scenario_cfs = []
            scenario_name = None

            if scenario is not None:
                scenario_name = scenario
            elif self.scenario is not None:
                scenario_name = self.scenario

            if scenario_name is None:
                if isinstance(self.parameters, dict):
                    if len(self.parameters) > 0:
                        scenario_name = list(self.parameters.keys())[0]

            resolved_params = self._resolve_parameters_for_scenario(
                scenario_idx, scenario_name
            )

            for cf in self.cfs_mapping:
                if isinstance(cf["value"], str):
                    try:
                        value = safe_eval_cached(
                            cf["value"],
                            parameters=resolved_params,
                            scenario_idx=scenario_idx,
                            SAFE_GLOBALS=self.SAFE_GLOBALS,
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to evaluate symbolic CF '{cf['value']}' with parameters {resolved_params}. Error: {e}"
                        )
                        value = 0
                else:
                    value = cf["value"]

                self.scenario_cfs.append(
                    {
                        "supplier": cf["supplier"],
                        "consumer": cf["consumer"],
                        "positions": cf["positions"],
                        "value": value,
                    }
                )

            matrix_type = (
                "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
            )
            self.characterization_matrix = initialize_lcia_matrix(
                self.lca, matrix_type=matrix_type
            )

            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    self.characterization_matrix[i, j] = cf["value"]

            self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia(self) -> None:
        """
        Perform the life cycle impact assessment (LCIA) using the evaluated characterization matrix.

        This method multiplies the inventory matrix with the CF matrix to produce a scalar score
        or a distribution of scores (for uncertainty propagation).


        Behavior
        --------
        - In deterministic mode: computes a single scalar LCIA score.
        - In uncertainty mode (3D matrix): computes a 1D array of LCIA scores across all iterations.


        Notes
        -----
        - Must be called after `evaluate_cfs()`.
        - Requires the inventory to be computed via `lci()`.
        - Technosphere or biosphere matrix is chosen based on exchange type.


        Updates
        -------
        - Sets `score` to the final impact value(s)
        - Stores `characterized_inventory` as a matrix or tensor

        If no exchanges are matched, the score defaults to 0.

        :return: None
        """

        # check that teh sum of processed biosphere and technosphere
        # edges is superior to zero, otherwise, we exit
        if (
            len(self.processed_biosphere_edges) + len(self.processed_technosphere_edges)
            == 0
        ):
            self.logger.warning(
                "No exchanges were matched or characterized. Score is set to 0."
            )

            self.score = 0
            return

        is_biosphere = len(self.biosphere_edges) > 0

        if self.use_distributions and self.iterations > 1:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )

            # Convert 2D inventory to sparse.COO
            inventory_coo = sparse.COO.from_scipy_sparse(inventory)

            # Broadcast inventory shape for multiplication
            inv_expanded = inventory_coo[:, :, None]  # (i, j, 1)

            # Element-wise multiply
            characterized = self.characterization_matrix * inv_expanded

            # Sum across dimensions i and j to get 1 value per iteration
            self.characterized_inventory = characterized
            self.score = characterized.sum(axis=(0, 1))

        else:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )
            self.characterized_inventory = self.characterization_matrix.multiply(
                inventory
            )
            self.score = self.characterized_inventory.sum()

    def statistics(self):
        """
        Print a summary table of method metadata and coverage statistics.

        This includes:
        - Demand activity name
        - Method name and data file
        - Unit (if available)
        - Total CFs in the method file
        - Number of CFs used (i.e., matched to exchanges)
        - Number of unique CF values applied
        - Number of characterized vs. uncharacterized exchanges
        - Ignored locations or CFs that could not be applied

        This is a useful diagnostic tool to assess method coverage and
        identify missing or unmatched data.

        Output
        ------
        - Prints a PrettyTable to the console
        - Does not return a value

        Notes
        -----
        - Can be used after `lcia()` to assess method completeness
        - Will reflect both direct and fallback-based characterizations
        """

        # build PrettyTable
        table = PrettyTable()
        table.header = False
        rows = []
        try:
            rows.append(
                [
                    "Activity",
                    fill(
                        list(self.lca.demand.keys())[0]["name"],
                        width=45,
                    ),
                ]
            )
        except TypeError:
            rows.append(
                [
                    "Activity",
                    fill(
                        bw2data.get_activity(id=list(self.lca.demand.keys())[0])[
                            "name"
                        ],
                        width=45,
                    ),
                ]
            )
        rows.append(["Method name", fill(str(self.method), width=45)])
        if "unit" in self.method_metadata:
            rows.append(["Unit", fill(self.method_metadata["unit"], width=45)])
        rows.append(["Data file", fill(self.filepath.stem, width=45)])
        rows.append(["CFs in method", self.cfs_number])
        rows.append(
            [
                "CFs used",
                len([x["value"] for x in self.cfs_mapping if len(x["positions"]) > 0]),
            ]
        )
        unique_cfs = set(
            [
                x["value"]
                for x in self.cfs_mapping
                if len(x["positions"]) > 0 and x["value"] is not None
            ]
        )
        rows.append(
            [
                "Unique CFs used",
                len(unique_cfs),
            ]
        )

        if self.ignored_method_exchanges:
            rows.append(
                ["CFs without eligible exc.", len(self.ignored_method_exchanges)]
            )

        if self.ignored_locations:
            rows.append(["Product system locations ignored", self.ignored_locations])

        if len(self.processed_biosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_biosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_biosphere_edges),
                ]
            )

        if len(self.processed_technosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_technosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_technosphere_edges),
                ]
            )

        for row in rows:
            table.add_row(row)

        print(table)

    def generate_cf_table(self, include_unmatched=False) -> pd.DataFrame:
        """
        Generate a detailed results table of characterized exchanges.

        Returns a pandas DataFrame with one row per characterized exchange,
        including the following fields:

        - Supplier and consumer activity name, reference product, and location
        - Flow amount
        - Characterization factor(s)
        - Characterized impact (CF × amount)

        Behavior
        --------
        - If uncertainty is enabled (`use_distributions=True`), the DataFrame contains:
          - Mean, std, percentiles, min/max for CFs and impact values
        - If deterministic: contains only point values for CF and impact

        Returns
        -------
        pd.DataFrame
            A table of all characterized exchanges with metadata and scores.

        Notes
        -----
        - Must be called after `evaluate_cfs()` and `lcia()`
        - Useful for debugging, reporting, or plotting contributions
        """

        if not self.scenario_cfs:
            self.logger.warning(
                "generate_cf_table() called before evaluate_cfs(). Returning empty DataFrame."
            )
            return pd.DataFrame()

        is_biosphere = True if self.technosphere_flow_matrix is None else False

        inventory = (
            self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
        )
        data = []

        if (
            self.use_distributions
            and hasattr(self, "characterization_matrix")
            and hasattr(self, "iterations")
        ):
            cm = self.characterization_matrix

            for i, j in zip(
                *cm.sum(axis=2).nonzero()
            ):  # Only loop over nonzero entries
                consumer = bw2data.get_activity(self.reversed_activity[j])
                supplier = (
                    bw2data.get_activity(self.reversed_biosphere[i])
                    if is_biosphere
                    else bw2data.get_activity(self.reversed_activity[i])
                )

                samples = np.array(cm[i, j, :].todense()).flatten().astype(float)
                amount = inventory[i, j]
                impact_samples = amount * samples

                # Percentiles
                cf_p = np.percentile(samples, [5, 25, 50, 75, 95])
                impact_p = np.percentile(impact_samples, [5, 25, 50, 75, 95])

                entry = {
                    "supplier name": supplier["name"],
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF (mean)": samples.mean(),
                    "CF (std)": samples.std(),
                    "CF (min)": samples.min(),
                    "CF (5th)": cf_p[0],
                    "CF (25th)": cf_p[1],
                    "CF (50th)": cf_p[2],
                    "CF (75th)": cf_p[3],
                    "CF (95th)": cf_p[4],
                    "CF (max)": samples.max(),
                    "impact (mean)": impact_samples.mean(),
                    "impact (std)": impact_samples.std(),
                    "impact (min)": impact_samples.min(),
                    "impact (5th)": impact_p[0],
                    "impact (25th)": impact_p[1],
                    "impact (50th)": impact_p[2],
                    "impact (75th)": impact_p[3],
                    "impact (95th)": impact_p[4],
                    "impact (max)": impact_samples.max(),
                }

                if is_biosphere:
                    entry["supplier categories"] = supplier.get("categories")
                else:
                    entry["supplier reference product"] = supplier.get(
                        "reference product"
                    )
                    entry["supplier location"] = supplier.get("location")

                data.append(entry)

        else:
            # Deterministic fallback
            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    consumer = bw2data.get_activity(self.reversed_activity[j])
                    supplier = (
                        bw2data.get_activity(self.reversed_biosphere[i])
                        if is_biosphere
                        else bw2data.get_activity(self.reversed_activity[i])
                    )

                    amount = inventory[i, j]
                    cf_value = cf["value"]
                    impact = amount * cf_value

                    entry = {
                        "supplier name": supplier["name"],
                        "consumer name": consumer["name"],
                        "consumer reference product": consumer.get("reference product"),
                        "consumer location": consumer.get("location"),
                        "amount": amount,
                        "CF": cf_value,
                        "impact": impact,
                    }

                    if is_biosphere:
                        entry["supplier categories"] = supplier.get("categories")
                    else:
                        entry["supplier reference product"] = supplier.get(
                            "reference product"
                        )
                        entry["supplier location"] = supplier.get("location")

                    data.append(entry)

        if include_unmatched is True:
            unprocess_exchanges = (
                self.unprocessed_biosphere_edges
                if is_biosphere is True
                else self.unprocessed_technosphere_edges
            )
            # Add unprocessed exchanges
            for i, j in unprocess_exchanges:
                if is_biosphere is True:
                    supplier = bw2data.get_activity(self.reversed_biosphere[i])
                else:
                    supplier = bw2data.get_activity(self.reversed_activity[i])
                consumer = bw2data.get_activity(self.reversed_activity[j])

                amount = inventory[i, j]
                cf_value = None
                impact = None

                entry = {
                    "supplier name": supplier["name"],
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF": cf_value,
                    "impact": impact,
                }

                if is_biosphere:
                    entry["supplier categories"] = supplier.get("categories")
                else:
                    entry["supplier reference product"] = supplier.get(
                        "reference product"
                    )
                    entry["supplier location"] = supplier.get("location")

                data.append(entry)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Order columns
        preferred_columns = [
            "supplier name",
            "supplier categories",
            "supplier reference product",
            "supplier location",
            "consumer name",
            "consumer reference product",
            "consumer location",
            "amount",
        ]

        # Add CF or CF summary columns
        if self.use_distributions:
            preferred_columns += [
                "CF (mean)",
                "CF (std)",
                "CF (min)",
                "CF (5th)",
                "CF (25th)",
                "CF (50th)",
                "CF (75th)",
                "CF (95th)",
                "CF (max)",
                "impact (mean)",
                "impact (std)",
                "impact (min)",
                "impact (5th)",
                "impact (25th)",
                "impact (50th)",
                "impact (75th)",
                "impact (95th)",
                "impact (max)",
            ]
        else:
            preferred_columns += ["CF", "impact"]

        df = df[[col for col in preferred_columns if col in df.columns]]

        return df

    @property
    def geo(self):
        """
        Get the GeoResolver instance for location containment checks.

        :return: GeoResolver object.
        """
        if getattr(self, "_geo", None) is None:
            self._geo = GeoResolver(self.weights)
        return self._geo
