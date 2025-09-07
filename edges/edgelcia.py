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


def add_cf_entry(
    cfs_mapping, supplier_info, consumer_info, direction, indices, value, uncertainty
):
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
    Cached version of _equality_supplier_signature, keyed by pre-hashable tuple.
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


def _collect_cf_prefixes_used_by_method(raw_cfs_data):
    """
    Return {scheme_lower: frozenset({prefixes})} of CF codes that will be queried.
    We only build prefix buckets that we will actually ask for.
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
    Build {scheme: {prefix: set(indices)}} but *only* for prefixes we will query.

    For each dataset code 'base', we generate all progressive prefixes of 'base'
    and, if a generated prefix is among required_prefixes[scheme], we add the index.
    This matches your startswith() semantics exactly.

    idx_to_norm_classes is like self.supplier_cls_bio etc.:
       {pos_idx: (("scheme", ("code1", "code2", ...)), ...)}
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
    From CF classifications (any allowed format), fetch the union of positions
    whose dataset codes start with any given CF code (per scheme), using the prefix index.
    Optionally intersect with current adjacency keys.
    """
    if not cf_classifications:
        return set()

    norm = _norm_cls(cf_classifications)  # (("SCHEME", ("code", ...)), ...)
    out = set()
    for scheme, codes in norm:
        sch = str(scheme).lower().strip()
        bucket = prefix_index_by_scheme.get(sch)
        if not bucket:
            continue
        for code in codes:
            pref = str(code).split(":", 1)[0].strip()
            hits = bucket.get(pref)
            if hits:
                out |= hits

    if adjacency_keys is not None:
        out &= adjacency_keys
    return out


def _norm_cls(x):
    """
    Normalize 'classifications' to a canonical, hashable form:
      (("SCHEME", ("code1","code2", ...)), ("SCHEME2", (...)), ...)
    Accepts:
      - dict: {"CPC": ["01","02"], "ISIC": ["A"]}
      - list/tuple of pairs: [("CPC","01"), ("CPC",["02","03"]), ("ISIC","A")]
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
    ):
        """
        Initialize an EdgeLCIA object for exchange-level life cycle impact assessment.

        Parameters
        ----------
        demand : dict
            A Brightway-style demand dictionary defining the functional unit.
        method : tuple, optional
            Method name as a tuple (e.g., ("AWARE", "2.0")), used to locate the CF JSON file.
        weight : str, optional
            Weighting variable used for region aggregation/disaggregation (e.g., "population", "gdp").
        parameters : dict, optional
            Dictionary of parameter values or scenarios for symbolic CF evaluation.
        scenario : str, optional
            Name of the default scenario (must match a key in `parameters`).
        filepath : str, optional
            Explicit path to the JSON method file; overrides `method` if provided.
        allowed_functions : dict, optional
            Additional safe functions available to CF evaluation expressions.
        use_distributions : bool, optional
            Whether to interpret CF uncertainty fields and perform Monte Carlo sampling.
        random_seed : int, optional
            Seed for reproducible uncertainty sampling.
        iterations : int, optional
            Number of Monte Carlo samples to draw if uncertainty is enabled.

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

    def _load_raw_lcia_data(self):
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
        Preprocess supplier and consumer flows into lookup dictionaries and
        materialized reversed lookups (dict per position) plus hot-field caches.

        Results:
          - self.supplier_lookup_bio / self.supplier_lookup_tech
          - self.reversed_supplier_lookup_bio / self.reversed_supplier_lookup_tech
          - self.supplier_loc_bio / self.supplier_loc_tech
          - self.supplier_cls_bio / self.supplier_cls_tech
          - self.consumer_lookup
          - self.reversed_consumer_lookup
          - self.consumer_loc / self.consumer_cls
          - (compat) self.supplier_lookup
        """

        # ---- What fields are required on the CONSUMER side (ignore control/meta fields)
        IGNORED_FIELDS = {"matrix", "operator", "weight", "classifications", "position"}
        self.required_consumer_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        # ---- Supplier lookups, per matrix
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

        # ---- Consumer lookup (always technosphere)
        self.consumer_lookup = preprocess_flows(
            flows_list=self.technosphere_flows,
            mandatory_fields=self.required_consumer_fields,
        )

        # ---- Helpers
        def _materialize_reversed(lookup: dict[int, list[int]]) -> dict[int, dict]:
            # map pos -> dict(key) so callers can use it directly (no dict(...) in hot loops)
            return {
                pos: dict(key) for key, positions in lookup.items() for pos in positions
            }

        # ---- Reversed lookups (materialized)
        self.reversed_supplier_lookup_bio = _materialize_reversed(
            self.supplier_lookup_bio
        )
        self.reversed_supplier_lookup_tech = _materialize_reversed(
            self.supplier_lookup_tech
        )
        self.reversed_consumer_lookup = _materialize_reversed(self.consumer_lookup)

        # 🔧 Enrich consumer reversed lookup with full metadata fields we may want to prefilter on.
        # In particular: bring 'classifications' from the actual activity dict.
        for idx, info in self.reversed_consumer_lookup.items():
            extra = self.position_to_technosphere_flows_lookup.get(idx, {})
            if "classifications" in extra and "classifications" not in info:
                info["classifications"] = extra["classifications"]

        # (Optional) Back-compat: a combined supplier_lookup for any legacy call sites
        # If all CFs are biosphere, expose the bio lookup; if all tech, expose tech; else merge.
        if self.supplier_lookup_bio and not self.supplier_lookup_tech:
            self.supplier_lookup = self.supplier_lookup_bio
        elif self.supplier_lookup_tech and not self.supplier_lookup_bio:
            self.supplier_lookup = self.supplier_lookup_tech
        else:
            # merged view (keys are hashable; positions lists are appended)
            merged = {}
            for src in (self.supplier_lookup_bio, self.supplier_lookup_tech):
                for k, v in src.items():
                    if k in merged:
                        merged[k].extend(v)
                    else:
                        merged[k] = list(v)
            self.supplier_lookup = merged

        # ---- Hot-field caches (avoid repeated dict lookups + allocations in tight loops)
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

        # --- Build classification prefix indexes (restricted to CF-used codes)
        self._cf_needed_prefixes = _collect_cf_prefixes_used_by_method(
            self.raw_cfs_data
        )

        # Suppliers
        self.cls_prefidx_supplier_bio = _build_prefix_index_restricted(
            self.supplier_cls_bio, self._cf_needed_prefixes
        )
        self.cls_prefidx_supplier_tech = _build_prefix_index_restricted(
            self.supplier_cls_tech, self._cf_needed_prefixes
        )
        # Consumers (always technosphere)
        self.cls_prefidx_consumer = _build_prefix_index_restricted(
            self.consumer_cls, self._cf_needed_prefixes
        )

    def _get_consumer_info(self, consumer_idx):
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
        """

        log = self.logger.getChild("map")  # edges.edgelcia.EdgeLCIA.map

        self._initialize_weights()
        self._preprocess_lookups()  # populates lookups and prefix indexes

        # ---- Build direction-specific bundles -----------------------------------
        DIR_BIO = "biosphere-technosphere"
        DIR_TECH = "technosphere-technosphere"

        # Adjacency + remaining edges per direction
        def build_adj(edges):
            ebs, ebc = defaultdict(set), defaultdict(set)
            rem = set(edges)
            for s, c in rem:
                ebs[s].add(c)
                ebc[c].add(s)
            return rem, ebs, ebc

        rem_bio, ebs_bio, ebc_bio = build_adj(self.biosphere_edges)
        rem_tec, ebs_tec, ebc_tec = build_adj(self.technosphere_edges)

        # Build indices once
        supplier_index_bio = build_index(
            self.supplier_lookup_bio, self.required_supplier_fields
        )
        supplier_index_tec = build_index(
            self.supplier_lookup_tech, self.required_supplier_fields
        )
        consumer_index = build_index(
            self.consumer_lookup, self.required_consumer_fields
        )

        # Allowlist for later steps (per direction)
        allow_bio = set()
        allow_tec = set()

        # Small helpers to select the right bundle per CF
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

        # --- helpers for concise logging -----------------------------------------
        def _short(d, limit=180):
            try:
                s = str(d)
            except Exception:
                s = repr(d)
            return s if len(s) <= limit else s[: limit - 1] + "…"

        def _count_none(x):
            return 0 if x is None else (len(x) if hasattr(x, "__len__") else 1)

        # High-level preamble
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

        # Bind hot locals (micro-optimization)
        consumer_lookup = self.consumer_lookup
        reversed_consumer_lookup = self.reversed_consumer_lookup

        # ---- Precompute required field tuples (no 'classifications') once
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

        # Iterate CFs
        for i, cf in enumerate(tqdm(self.raw_cfs_data, desc="Mapping exchanges")):
            s_crit = cf["supplier"]
            c_crit = cf["consumer"]

            # which direction are we in?
            dir_name, rem, ebs, ebc, s_index, s_lookup, s_reversed = get_dir_bundle(
                s_crit.get("matrix", "biosphere")
            )

            if not rem:
                # This direction already fully characterized
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

            s_matches_raw = list(s_out.matches)  # before adjacency & class refinement
            if s_class_hits is not None:
                s_cands = list(set(s_out.matches) & set(s_class_hits))
            else:
                s_cands = list(s_out.matches)
            # must still have consumers in adjacency
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

            c_matches_raw = list(c_out.matches)
            if c_class_hits is not None:
                c_cands = list(set(c_out.matches) & set(c_class_hits))
            else:
                c_cands = list(c_out.matches)
            c_cands = [c for c in c_cands if c in ebc]

            c_loc_only = set(c_out.location_only_rejects)
            if c_class_hits is not None:
                c_loc_only &= set(c_class_hits)
            c_loc_required = ("location" in c_crit) and (
                c_crit.get("location") is not None
            )

            # ---- DEBUG: explain empty candidate sets
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
                    # check adjacency pruning
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

            # ---------- Combine full matches using adjacency intersections ----------
            positions = []
            if s_cands and c_cands:
                cset = set(c_cands)
                for s in s_cands:
                    cs = ebs.get(s)
                    if not cs:
                        continue
                    for c in cs:
                        if c in cset:
                            positions.append((s, c))

            if positions:
                add_cf_entry(
                    cfs_mapping=self.cfs_mapping,
                    supplier_info=s_crit,
                    consumer_info=c_crit,
                    direction=dir_name,
                    indices=positions,
                    value=cf["value"],
                    uncertainty=cf.get("uncertainty"),
                )
                matched_positions_total += len(positions)
                log.debug(
                    "CF[%d] dir=%s MATCH | positions=%d | s_cands=%d c_cands=%d | s_loc_only=%d c_loc_only=%d",
                    i,
                    dir_name,
                    len(positions),
                    len(s_cands),
                    len(c_cands),
                    len(s_loc_only),
                    len(c_loc_only),
                )

                # prune matched edges from this direction
                for s, c in positions:
                    if (s, c) in rem:
                        rem.remove((s, c))
                        ebs[s].discard(c)
                        if not ebs[s]:
                            del ebs[s]
                        ebc[c].discard(s)
                        if not ebc[c]:
                            del ebc[c]
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

            # ---------- Build near-miss allowlists (location-only) ----------
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
                    log.debug(
                        "CF[%d] dir=%s allowlist add (supplier loc-only) | added=%d",
                        i,
                        dir_name,
                        added,
                    )

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
                    log.debug(
                        "CF[%d] dir=%s allowlist add (consumer loc-only) | added=%d",
                        i,
                        dir_name,
                        added,
                    )

            # both sides near-miss (rare but useful)
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
                    log.debug(
                        "CF[%d] dir=%s allowlist add (both loc-only) | added=%d",
                        i,
                        dir_name,
                        added,
                    )

        self._update_unprocessed_edges()

        # store per-direction allowlists for later passes
        self.eligible_edges_for_next_bio = allow_bio
        self.eligible_edges_for_next_tech = allow_tec

        log.debug(
            "END map_exchanges | matched_positions=%d | allow_bio=%d | allow_tec=%d | processed_bio=%d | processed_tech=%d | unprocessed_bio=%d | unprocessed_tech=%d",
            matched_positions_total,
            len(allow_bio),
            len(allow_tec),
            len(self.processed_biosphere_edges),
            len(self.processed_technosphere_edges),
            len(self.unprocessed_biosphere_edges),
            len(self.unprocessed_technosphere_edges),
        )

    def map_aggregate_locations(self) -> None:
        """
        Map unmatched exchanges using CFs from broader (aggregated) regions.

        This method resolves cases where a direct match was not found by using CFs
        defined at a higher aggregation level (e.g., region = "RER" instead of "FR").

        It computes weighted averages for aggregate CFs using a user-specified
        weighting variable (e.g., population, GDP, resource use) from the method metadata.

        Typical use case: national-level exchanges matched to region-level CFs
        when no country-specific CF is available.

        Notes
        -----
        - Weight values are extracted from the `weight` field in each CF.
        - Uses a two-pass matching strategy: fast signature-based prefiltering, then fallback.

        Preconditions
        -------------
        - `lci()` must be called
        - `map_exchanges()` must be called
        - Weight metadata must be available for aggregation

        Updates
        -------
        - Extends `cfs_mapping` with newly matched aggregate CFs.
        - Updates internal lists of `processed_*` and `unprocessed_*` edges.
        """

        self._initialize_weights()
        logger.info("Handling static regions…")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

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
            edges_index = defaultdict(list)

            # let's remove edges that have no chance of qualifying
            allowed = (
                self.eligible_edges_for_next_bio
                if direction == "biosphere-technosphere"
                else self.eligible_edges_for_next_tech
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

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

                if supplier_location is None:
                    candidate_suppliers_locations = [
                        "__ANY__",
                    ]
                else:
                    # 🔁 Use the shared utility function to get subregions
                    candidate_suppliers_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=supplier_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        supplier=True,
                    )

                if len(candidate_suppliers_locations) == 0:
                    candidate_suppliers_locations = [
                        supplier_location,
                    ]

                if consumer_location is None:
                    candidate_consumer_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_consumer_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=consumer_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        supplier=False,
                    )

                if len(candidate_consumer_locations) == 0:
                    candidate_consumer_locations = [
                        consumer_location,
                    ]

                if (
                    len(candidate_suppliers_locations) == 1
                    and len(candidate_consumer_locations) == 1
                ):
                    # neither the supplier or consumer locations are composite locations
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

            # Pass 1
            if len(prefiltered_groups) > 0:
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
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            _,
                            _,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                            f"with candidate suppliers={candidate_supplier_locations} and consumers={candidate_consumer_locations}"
                        )

            # Pass 2
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

            if len(grouped_edges) > 0:
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
                        for supplier_idx, consumer_idx in edge_group:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=dict(s_key),
                                consumer_info=dict(c_key),
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={s_key}, consumer={c_key} "
                            f"with candidate suppliers={candidate_suppliers} and consumers={candidate_consumers}"
                        )

        self._update_unprocessed_edges()

    def map_dynamic_locations(self) -> None:
        """
        Handle location-matching for dynamic or relative regions such as 'RoW' or 'RoE'.

        This method computes CFs for exchanges whose consumer location is a dynamic placeholder
        like "Rest of World" (RoW) by averaging over all regions **not** explicitly covered
        by the inventory.

        It uses the known supplier-consumer relationships in the inventory to identify
        excluded subregions, and builds CFs from the remaining regions using a weighted average.

        Typical use case: inventory exchanges with generic locations that need fallback handling
        (e.g., average CF for "RoW" that excludes countries already modeled explicitly).

        Notes
        -----
        - Technosphere exchange structure is analyzed to determine uncovered locations.
        - CFs are matched using exchange signatures and spatial exclusions.
        - Weighted averages are computed from the remaining eligible subregions.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - `weights` must be defined (e.g., population, GDP, etc.)
        - Suitable for methods with CFs that include relative or global coverage

        Updates
        -------
        - Adds dynamic-region CFs to `cfs_mapping`
        - Updates internal lists of processed and unprocessed exchanges
        """

        self._initialize_weights()
        logger.info("Handling dynamic regions…")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

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
            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            # let's remove edges that have no chance of qualifying
            allowed = (
                self.eligible_edges_for_next_bio
                if direction == "biosphere-technosphere"
                else self.eligible_edges_for_next_tech
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_info = self._get_consumer_info(consumer_idx)
                supplier_info = rev_sup[supplier_idx]

                supplier_loc = (
                    self.supplier_loc_bio.get(supplier_idx)
                    if direction == "biosphere-technosphere"
                    else self.supplier_loc_tech.get(supplier_idx)
                )
                consumer_loc = self.consumer_loc.get(consumer_idx)

                # Skip if neither side is dynamic
                if supplier_loc not in ["RoW", "RoE"] and consumer_loc not in [
                    "RoW",
                    "RoE",
                ]:
                    continue

                # Identify dynamic role
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
                    if supplier_loc is None:
                        candidate_suppliers_locs = [
                            "__ANY__",
                        ]
                    else:
                        candidate_suppliers_locs = [supplier_loc]

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
                    if consumer_loc is None:
                        candidate_consumers_locs = [
                            "__ANY__",
                        ]
                    else:
                        candidate_consumers_locs = [consumer_loc]

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

            # Pass 1
            if len(prefiltered_groups) > 0:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing dynamic groups (pass 1)",
                ):
                    rep_supplier = group_edges[0][2]
                    rep_consumer = group_edges[0][3]
                    candidate_supplier_locations = group_edges[0][-2]
                    candidate_consumer_locations = group_edges[0][-1]

                    new_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                        candidate_suppliers=candidate_supplier_locations,
                        candidate_consumers=candidate_consumer_locations,
                        supplier_info=rep_supplier,
                        consumer_info=rep_consumer,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        cf_index=self.cf_index,
                    )

                    if new_cf:
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            _,
                            _,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={rep_supplier}, consumer={rep_consumer} "
                            f"with candidate suppliers={candidate_supplier_locations} and consumers={candidate_consumer_locations}"
                        )

            # Pass 2
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

            if len(grouped_edges) > 0:
                for (
                    s_key,
                    c_key,
                    (candidate_supplier_locations, candidate_consumer_locations),
                ), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing dynamic groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        s_key,
                        c_key,
                        candidate_supplier_locations,
                        candidate_consumer_locations,
                    )

                    if new_cf:
                        for supplier_idx, consumer_idx in edge_group:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=dict(s_key),
                                consumer_info=dict(c_key),
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={s_key}, consumer={c_key} "
                            f"with candidate suppliers={candidate_supplier_locations} and consumers={candidate_consumer_locations}"
                        )

        self._update_unprocessed_edges()

    def map_contained_locations(self) -> None:
        """
        Resolve unmatched exchanges by assigning CFs from spatially containing regions.

        This method assigns a CF to an exchange based on a broader geographic area that
        contains the exchange's region. For example, if no CF exists for "Québec", but
        a CF exists for "Canada", that CF will be used.

        It is typically used when the method file contains national-level CFs but the
        inventory includes subnational or otherwise finer-grained locations.

        Notes
        -----
        - Uses a geographic containment hierarchy to resolve matches (e.g., geo aggregation trees).
        - Only uncharacterized exchanges are considered.
        - This is conceptually the inverse of `map_aggregate_locations()`.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - A geo containment structure must be defined or inferred

        Updates
        -------
        - Adds fallback CFs to `cfs_mapping`
        - Updates internal tracking of processed edges
        """

        self._initialize_weights()
        logger.info("Handling contained locations…")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

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
            edges_index = defaultdict(list)

            # let's remove edges that have no chance of qualifying
            allowed = (
                self.eligible_edges_for_next_bio
                if direction == "biosphere-technosphere"
                else self.eligible_edges_for_next_tech
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

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

                # 🔁 Use the shared utility function to get subregions
                if supplier_location is None:
                    candidate_suppliers_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_suppliers_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=supplier_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=False,
                        supplier=True,
                    )

                if consumer_location is None:
                    candidate_consumer_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_consumer_locations = resolve_candidate_locations(
                        geo=self.geo,
                        location=consumer_location,
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=False,
                        supplier=False,
                    )

                if (
                    len(candidate_suppliers_locations) == 0
                    and len(candidate_consumer_locations) == 0
                ):
                    # neither the supplier or consumer locations are composite locations
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

            # Pass 1
            if len(prefiltered_groups) > 0:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing contained groups (pass 1)",
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

                    if new_cf:
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            _,
                            _,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                            f"with candidate suppliers={candidate_supplier_locations} and consumers={candidate_consumer_locations}"
                        )

            # Pass 2
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

            if len(grouped_edges) > 0:
                for (
                    supplier_info,
                    consumer_info,
                    (candidate_suppliers, candidate_consumers),
                ), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing contained groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        supplier_info,
                        consumer_info,
                        candidate_suppliers,
                        candidate_consumers,
                    )
                    if new_cf:
                        for supplier_idx, consumer_idx in edge_group:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=dict(supplier_info),
                                consumer_info=dict(consumer_info),
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                            f"with candidate suppliers={candidate_suppliers} and consumers={candidate_consumers}"
                        )

        self._update_unprocessed_edges()

    def map_remaining_locations_to_global(self) -> None:
        """
        Assign global fallback CFs to exchanges that remain unmatched after all regional mapping steps.

        This method ensures that all eligible exchanges are characterized by assigning a CF
        from the global region ("GLO") when no direct, aggregate, dynamic, or containing region match
        has been found.

        It is the last step in the regional mapping cascade.

        Notes
        -----
        - Uses a weighted global average if multiple CFs exist for the same exchange type.
        - If no global CF exists for a given exchange, it remains uncharacterized.
        - This step guarantees that the system-wide score is computable unless coverage is zero.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - Should follow other mapping steps: `map_aggregate_locations()`, `map_dynamic_locations()`, etc.

        Updates
        -------
        - Adds fallback CFs to `cfs_mapping`
        - Marks remaining exchanges as processed
        """

        self._initialize_weights()
        logger.info("Handling remaining exchanges…")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # Resolve candidate locations for GLO once using utility
        global_locations = resolve_candidate_locations(
            geo=self.geo,
            location="GLO",
            weights=frozenset(k for k, v in self.weights.items()),
            containing=True,
        )

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
            edges_index = defaultdict(list)

            # let's remove edges that have no chance of qualifying
            allowed = (
                self.eligible_edges_for_next_bio
                if direction == "biosphere-technosphere"
                else self.eligible_edges_for_next_tech
            )
            if allowed:
                unprocessed_edges = [e for e in unprocessed_edges if e in allowed]

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

                if supplier_location is None:
                    candidate_suppliers_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_suppliers_locations = global_locations

                if consumer_location is None:
                    candidate_consumers_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_consumers_locations = global_locations

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

            # Pass 1
            if len(prefiltered_groups) > 0:
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
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            _,
                            _,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=unc,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                            f"with candidate suppliers={global_locations} and consumers={global_locations}"
                        )

            # Pass 2
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

            if len(grouped_edges) > 0:
                for (
                    supplier_info,
                    consumer_info,
                    (candidate_suppliers, candidate_consumers),
                ), edge_group in tqdm(
                    grouped_edges.items(), desc="Processing global groups (pass 2)"
                ):
                    new_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                        supplier_info,
                        consumer_info,
                        candidate_suppliers,
                        candidate_consumers,
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
                        for supplier_idx, consumer_idx in edge_group:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=dict(supplier_info),
                                consumer_info=dict(consumer_info),
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=unc,
                            )
                    else:
                        self.logger.warning(
                            f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                            f"with candidate suppliers={candidate_suppliers} and consumers={candidate_consumers}"
                        )

        self._update_unprocessed_edges()

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
        if getattr(self, "_geo", None) is None:
            self._geo = GeoResolver(self.weights)
        return self._geo
