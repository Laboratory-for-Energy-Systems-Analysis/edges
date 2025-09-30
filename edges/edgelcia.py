"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import math
import os
import sys
import platform
import scipy
import sparse as sp
import time
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


def _collect_cf_prefixes_used_by_method(raw_cfs_data):
    """
    Collect all classification prefixes that appear in a CF method.

    :param raw_cfs_data: Iterable of CF entries.
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

    :param idx_to_norm_classes: Mapping of activity index -> normalized classifications.
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
                base = str(code)
                if not base:
                    continue
                # generate progressive prefixes: '01.12' -> '0','01','01.','01.1','01.12'
                # (progressive is safest because your CF can be any prefix)
                for k in range(1, len(base) + 1):
                    pref = base[:k]
                    if pref in wanted:
                        out[sch][pref].add(idx)
    return out


def _norm_cls(x):
    def _san(c):
        # strip trailing ":..." and whitespace once
        return str(c).split(":", 1)[0].strip()

    if not x:
        return ()
    bag = {}
    if isinstance(x, dict):
        for scheme, codes in x.items():
            if codes is None:
                continue
            it = codes if isinstance(codes, (list, tuple, set)) else [codes]
            bag.setdefault(str(scheme), set()).update(_san(c) for c in it)
    elif isinstance(x, (list, tuple)):
        for item in x:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            scheme, codes = item
            if codes is None:
                continue
            it = codes if isinstance(codes, (list, tuple, set)) else [codes]
            bag.setdefault(str(scheme), set()).update(_san(c) for c in it)
    else:
        return ()
    return tuple((scheme, tuple(sorted(bag[scheme]))) for scheme in sorted(bag))


def make_coo_deterministic(coo: "sparse.COO"):
    """Return a COO with deterministically ordered coords and no duplicates.

    - Works for 2D and 3D COO.
    - No use of .sum() (avoids accidental scalar reduction).
    - If `coo` is not a pydata.sparse COO, just return it unchanged.
    """
    import numpy as _np
    import sparse as _sp

    # Pass through non-COO objects unchanged (e.g., scalar, ndarray)
    if not isinstance(coo, _sp.COO):
        return coo

    # Fast path: empty matrix
    if coo.nnz == 0:
        # Ensure the metadata flags are consistent
        return _sp.COO(
            coords=coo.coords,
            data=coo.data,
            shape=coo.shape,
            has_duplicates=False,
            sorted=True,
        )

    # 1) Compute a flattened linear index for each coordinate column
    #    This is stable across platforms and makes duplicate detection easy.
    lin = _np.ravel_multi_index(coo.coords, coo.shape)

    # 2) Sort by linear index (deterministic total ordering)
    order = _np.argsort(lin, kind="mergesort")  # stable sort
    lin_sorted = lin[order]
    coords_sorted = coo.coords[:, order]
    data_sorted = coo.data[order]

    # 3) Coalesce duplicates: sum data for identical linear indices
    uniq_lin, first_idx, counts = _np.unique(
        lin_sorted, return_index=True, return_counts=True
    )
    if _np.any(counts > 1):
        # Sum consecutive runs for duplicates
        summed_data = _np.add.reduceat(data_sorted, first_idx)
        uniq_coords = coords_sorted[:, first_idx]
    else:
        # No duplicates; keep sorted arrays
        summed_data = data_sorted
        uniq_coords = coords_sorted

    # 4) Rebuild a canonical COO (flags set so downstream ops know it's sorted)
    return _sp.COO(
        coords=uniq_coords,
        data=summed_data,
        shape=coo.shape,
        has_duplicates=False,
        sorted=True,
    )


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
        self.technosphere_flow_matrix = None
        self.technosphere_edges = set()
        self.biosphere_edges = set()
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
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
        self.log_platform()

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

        self._last_edges_snapshot_bio = set()
        self._last_edges_snapshot_tech = set()
        self._last_eval_scenario_name = None
        self._last_eval_scenario_idx = None
        self._failed_edges_tech: set[tuple[int, int]] = set()
        self._failed_edges_bio: set[tuple[int, int]] = set()
        self._last_nonempty_edges_snapshot_bio = set()
        self._last_nonempty_edges_snapshot_tech = set()
        self._ever_seen_edges_bio: set[tuple[int, int]] = set()
        self._ever_seen_edges_tech: set[tuple[int, int]] = set()
        self._flows_version = None
        self._cls_hits_cache = {}

    def log_platform(self):
        """
        Log versions of key dependencies and environment variables for debugging.
        """

        self.logger.info(
            "VERSIONS: python %s, numpy %s, scipy %s, sparse %s, platform %s",
            sys.version,
            np.__version__,
            scipy.__version__,
            sp.__version__,
            platform.platform(),
        )

        self.logger.info(
            "THREADS: %s",
            {
                k: os.environ.get(k)
                for k in [
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
        )

    def _load_raw_lcia_data(self):
        """
        Load and validate raw LCIA data for a given method.

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

        for cf in self.raw_cfs_data:
            cf["_norm_supplier_cls"] = _norm_cls(
                cf.get("supplier", {}).get("classifications")
            )
            cf["_norm_consumer_cls"] = _norm_cls(
                cf.get("consumer", {}).get("classifications")
            )

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

    def _cls_candidates_from_cf_cached(
        self, norm_cls, prefix_index_by_scheme, adjacency_keys=None
    ) -> set[int]:
        if not norm_cls:
            return set()
        cache_key = (id(prefix_index_by_scheme), norm_cls)
        base = self._cls_hits_cache.get(cache_key)
        if base is None:
            out = set()
            get_scheme = prefix_index_by_scheme.get
            for scheme, codes in norm_cls:
                bucket = get_scheme(str(scheme).lower().strip())
                if not bucket:
                    continue
                for code in codes:  # codes already sanitized
                    hits = bucket.get(code)  # exact prefix bucket
                    if hits:
                        out |= hits
            base = frozenset(out)  # cache as frozenset
            self._cls_hits_cache[cache_key] = base

        # No extra set() creations — let frozenset intersect in C
        return base if adjacency_keys is None else (base & adjacency_keys)

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

        # Convenience: available locations on each side in the method
        self.method_supplier_locs = {s for (s, _) in self.weights.keys()}
        self.method_consumer_locs = {c for (_, c) in self.weights.keys()}

        if hasattr(self, "geo") and getattr(self, "geo", None) is not None:
            getattr(
                self.geo, "_cached_lookup", lambda: None
            ) and self.geo._cached_lookup.cache_clear()

    def _ensure_filtered_lookups_for_current_edges(self) -> None:
        """Make sure filtered lookups + reversed maps exist for the current edge sets."""
        have = (
            isinstance(getattr(self, "reversed_consumer_lookup", None), dict)
            and isinstance(getattr(self, "reversed_supplier_lookup_bio", None), dict)
            and isinstance(getattr(self, "reversed_supplier_lookup_tech", None), dict)
        )
        if have:
            return

        # If needed, make sure inventory/edges exist
        if (self.biosphere_edges is None and self.technosphere_edges is None) or (
            not self.biosphere_edges and not self.technosphere_edges
        ):
            self.lci()

        restrict_sup_bio = {s for s, _ in (self.biosphere_edges or [])} or None
        restrict_sup_tec = {s for s, _ in (self.technosphere_edges or [])} or None
        restrict_con = (
            {c for _, c in (self.biosphere_edges or [])}
            | {c for _, c in (self.technosphere_edges or [])}
        ) or None

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio,
            restrict_supplier_positions_tech=restrict_sup_tec,
            restrict_consumer_positions=restrict_con,
        )

    def _get_candidate_supplier_keys(self):
        """
        Get possible supplier activity keys matching a CF entry.

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
        Resolve symbolic parameters for a given scenario, without spamming warnings.
        - If scenario_name is None, fall back to self.scenario, then first available key.
        - Warn only if a *provided* scenario_name is missing from parameters.
        """
        # Determine effective scenario name
        effective_name = (
            scenario_name
            if scenario_name is not None
            else (self.scenario if self.scenario is not None else None)
        )

        if effective_name is None:
            # No scenario chosen; if params exist, we can still evaluate constants or
            # expressions that don't rely on scenario keys. Return empty silently.
            return {}

        # If we have parameters but the requested name is missing
        if isinstance(self.parameters, dict) and effective_name not in self.parameters:
            # Warn only when user explicitly asked for this scenario
            if scenario_name is not None:
                self.logger.warning(
                    f"No parameter set found for scenario '{effective_name}'. Using empty defaults."
                )
            return {}

        param_set = (
            self.parameters.get(effective_name)
            if isinstance(self.parameters, dict)
            else None
        )
        if not param_set:
            return {}

        # Resolve index-aware values
        resolved = {}
        for k, v in param_set.items():
            if isinstance(v, dict):
                resolved[k] = v.get(str(scenario_idx), list(v.values())[-1])
            else:
                resolved[k] = v
        return resolved

    def _update_unprocessed_edges(self):
        """
        Add new edges to the list of unprocessed edges.

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

    def _preprocess_lookups(
        self,
        restrict_supplier_positions_bio: set[int] | None = None,
        restrict_supplier_positions_tech: set[int] | None = None,
        restrict_consumer_positions: set[int] | None = None,
    ):
        """
        Preprocess supplier and consumer flows into lookup dictionaries and
        materialized reversed lookups (dict per position) plus hot-field caches.

        This version caches *base* lookups built from all flows, then constructs
        filtered, tiny lookups for just the positions in `restrict_*` for each run.

        Results populated on self:
          - supplier lookups (filtered):
              self.supplier_lookup_bio / self.supplier_lookup_tech
          - reversed (position -> key dict):
              self.reversed_supplier_lookup_bio / self.reversed_supplier_lookup_tech
              self.reversed_consumer_lookup
          - hot caches:
              self.supplier_loc_bio / self.supplier_loc_tech
              self.supplier_cls_bio / self.supplier_cls_tech
              self.consumer_loc / self.consumer_cls
          - combined supplier_lookup (back-compat):
              self.supplier_lookup
          - prefix indexes (restricted to CF-used codes):
              self.cls_prefidx_supplier_bio / self.cls_prefidx_supplier_tech
              self.cls_prefidx_consumer
        """

        # ---- Figure out required CONSUMER fields once (ignore control/meta fields)
        IGNORED_FIELDS = {"matrix", "operator", "weight", "classifications", "position"}
        if not hasattr(self, "required_consumer_fields"):
            self.required_consumer_fields = {
                k
                for cf in self.raw_cfs_data
                for k in cf["consumer"].keys()
                if k not in IGNORED_FIELDS
            }

        if getattr(self, "_base_supplier_lookup_bio", None) is None:
            if self.biosphere_flows:
                self._base_supplier_lookup_bio = preprocess_flows(
                    flows_list=self.biosphere_flows,
                    mandatory_fields=self.required_supplier_fields,
                )
            else:
                self._base_supplier_lookup_bio = {}

        if getattr(self, "_base_supplier_lookup_tech", None) is None:
            if self.technosphere_flows:
                self._base_supplier_lookup_tech = preprocess_flows(
                    flows_list=self.technosphere_flows,
                    mandatory_fields=self.required_supplier_fields,
                )
            else:
                self._base_supplier_lookup_tech = {}

        if getattr(self, "_base_consumer_lookup", None) is None:
            self._base_consumer_lookup = preprocess_flows(
                flows_list=self.technosphere_flows or [],
                mandatory_fields=self.required_consumer_fields,
            )

        base_bio = self._base_supplier_lookup_bio
        base_tech = self._base_supplier_lookup_tech
        base_con = self._base_consumer_lookup

        # ---- Filter lookups down to the positions we will actually touch ----------
        def _filter_lookup(
            base: dict[tuple, list[int]], allowed: set[int] | None
        ) -> dict[tuple, list[int]]:
            if not base:
                return {}
            if allowed is None:
                # No restriction requested
                return base
            if not allowed:
                # Explicitly restrict to empty: return empty
                return {}
            out: dict[tuple, list[int]] = {}
            # Membership test is O(1) with set
            _allowed = allowed
            for key, positions in base.items():
                # positions is a list[int]; keep only those in allowed
                kept = [p for p in positions if p in _allowed]
                if kept:
                    out[key] = kept
            return out

        self.supplier_lookup_bio = _filter_lookup(
            base_bio, restrict_supplier_positions_bio
        )
        self.supplier_lookup_tech = _filter_lookup(
            base_tech, restrict_supplier_positions_tech
        )
        self.consumer_lookup = _filter_lookup(base_con, restrict_consumer_positions)

        # ---- Reversed lookups (materialized) for filtered sets --------------------
        # Map each *position* back to the (hashable) key dict used in the lookup
        def _materialize_reversed(lookup: dict[tuple, list[int]]) -> dict[int, dict]:
            # dict(key) avoids allocations during hot loops elsewhere
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

        self.logger.debug(
            "lookups: sup_tech=%d sup_bio=%d con=%d",
            len(self.reversed_supplier_lookup_tech),
            len(self.reversed_supplier_lookup_bio),
            len(self.reversed_consumer_lookup),
        )
        # show a couple of concrete rows if any
        for i, (pos, info) in enumerate(
            list(self.reversed_consumer_lookup.items())[:5]
        ):
            self.logger.debug("consumer[%d] -> %s", pos, info)
        for i, (pos, info) in enumerate(
            list(self.reversed_supplier_lookup_tech.items())[:5]
        ):
            self.logger.debug("supplier_tech[%d] -> %s", pos, info)

        # ---- Enrich consumer reversed lookup with activity metadata (classifications) ----
        # Bring 'classifications' from the activity map if missing (used by class filters)
        if self.position_to_technosphere_flows_lookup:
            for idx, info in self.reversed_consumer_lookup.items():
                extra = self.position_to_technosphere_flows_lookup.get(idx, {})
                if "location" not in info and "location" in extra:
                    info["location"] = extra["location"]
                if "classifications" in extra and "classifications" not in info:
                    info["classifications"] = extra["classifications"]

            for idx, info in self.reversed_supplier_lookup_tech.items():
                extra = self.position_to_technosphere_flows_lookup.get(idx, {})

                if "location" in extra and "location" not in info:
                    info["location"] = extra["location"]

                if "classifications" in extra and "classifications" not in info:
                    info["classifications"] = extra["classifications"]

        # ---- Back-compat: merged supplier_lookup view if needed -------------------
        if self.supplier_lookup_bio and not self.supplier_lookup_tech:
            self.supplier_lookup = self.supplier_lookup_bio
        elif self.supplier_lookup_tech and not self.supplier_lookup_bio:
            self.supplier_lookup = self.supplier_lookup_tech
        else:
            merged: dict[tuple, list[int]] = {}
            for src in (self.supplier_lookup_bio, self.supplier_lookup_tech):
                for k, v in src.items():
                    if k in merged:
                        merged[k].extend(v)
                    else:
                        merged[k] = list(v)
            self.supplier_lookup = merged

        # ---- Hot-field caches (avoid dict lookups in tight loops) -----------------
        self.supplier_loc_bio = {
            i: d.get("location") for i, d in self.reversed_supplier_lookup_bio.items()
        }
        self.supplier_loc_tech = {
            i: d.get("location") for i, d in self.reversed_supplier_lookup_tech.items()
        }
        self.consumer_loc = {
            i: d.get("location") for i, d in self.reversed_consumer_lookup.items()
        }

        self.logger.debug(
            "hot-caches: supplier_loc_tech=%s | consumer_loc=%s",
            {k: self.supplier_loc_tech[k] for k in list(self.supplier_loc_tech)[:5]},
            {k: self.consumer_loc[k] for k in list(self.consumer_loc)[:5]},
        )

        # Only log method keys if weights exist (they are set by _initialize_weights later)
        if self.weights:
            sup_locs = sorted({s for (s, _) in self.weights})
            con_locs = sorted({c for (_, c) in self.weights})
        else:
            sup_locs = con_locs = []
        self.logger.debug(
            "method keys: supplier_locs=%s | consumer_locs=%s", sup_locs, con_locs
        )

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

        # ---- CF-needed classification prefixes (compute once per method) ----------
        if not hasattr(self, "_cf_needed_prefixes"):
            self._cf_needed_prefixes = _collect_cf_prefixes_used_by_method(
                self.raw_cfs_data
            )

        # ---- Build prefix indexes from the *filtered* caches ----------------------
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
        self._cls_hits_cache.clear()

    def _get_supplier_info(self, supplier_idx: int, direction: str) -> dict:
        """
        Robustly fetch supplier info for a row index in either direction.
        Uses filtered reversed lookups first; falls back to the full activity map.

        Ensures we also keep the hot caches (loc/cls) coherent when we fill from fallback.
        """
        if direction == "biosphere-technosphere":
            info = self.reversed_supplier_lookup_bio.get(supplier_idx)
            if info is not None:
                return info

            # Fallback for biosphere suppliers: project the dataset to the
            # *method-required* supplier fields (method-agnostic).
            try:
                ds = bw2data.get_activity(self.reversed_biosphere[supplier_idx])
            except Exception:
                ds = {}

            info = self._project_dataset_to_required_fields(
                ds=ds,
                required_fields=self.required_supplier_fields,
            )

            # Optional: lightweight debug if the projection missed any required keys
            missing = [
                k
                for k in self.required_supplier_fields
                if k not in info
                and k not in {"matrix", "operator", "weight", "position", "excludes"}
            ]
            if missing:
                self.logger.debug(
                    "biosphere-fallback: missing required supplier keys %s for idx=%s",
                    missing,
                    supplier_idx,
                )
            return info

        # --- technosphere-technosphere
        info = self.reversed_supplier_lookup_tech.get(supplier_idx)
        if info is not None:
            return info

        # Fallback to full activity metadata for this position
        act = self.position_to_technosphere_flows_lookup.get(supplier_idx, {})
        info = dict(act) if act else {}

        # Normalize optional bits to help later class/location logic
        if "classifications" in info:
            self.supplier_cls_tech[supplier_idx] = _norm_cls(info["classifications"])
        if "location" in info:
            self.supplier_loc_tech[supplier_idx] = info["location"]

        if not info or (("location" not in info) and ("classifications" not in info)):
            act = self.position_to_technosphere_flows_lookup.get(supplier_idx, {})
            info = dict(act) if act else {}
            # keep hot caches coherent
            if "classifications" in info:
                self.supplier_cls_tech[supplier_idx] = _norm_cls(
                    info["classifications"]
                )
            if "location" in info:
                self.supplier_loc_tech[supplier_idx] = info["location"]

        return info

    def _project_dataset_to_required_fields(
        self, ds: dict, required_fields: set[str]
    ) -> dict:
        """
        Method-agnostic projection: given a BW2 dataset and the method’s
        required supplier fields, pull values from reasonable source keys.
        - Does not assume a particular LCIA method.
        - Normalizes simple container types where sensible.
        """
        out: dict = {}

        # Where to pull each logical field from (in order of preference).
        # Safe, generic mappings that work across many methods.
        FIELD_SOURCES: dict[str, tuple[str, ...]] = {
            "name": ("name",),
            "reference product": ("reference product", "reference_product"),
            "unit": ("unit",),
            "location": ("location",),
            "categories": ("categories",),  # EcoSpold-style list/tuple
            "classifications": (
                "classifications",
                "categories",
            ),  # Some methods key on either
            # For any other field, we'll try the same key name directly.
        }

        for f in required_fields or ():
            if f in {"matrix", "operator", "weight", "position", "excludes"}:
                continue
            candidates = FIELD_SOURCES.get(f, (f,))
            val = None
            for src in candidates:
                if isinstance(ds, dict) and src in ds:
                    val = ds.get(src)
                    break
            if val is None:
                continue

            # Light normalization
            if f == "categories" and isinstance(val, (list, tuple)):
                out[f] = tuple(val)
            else:
                out[f] = val

        # If the method didn’t explicitly require classifications but they are present,
        # keep them as a free bonus (helps other methods without hurting matching).
        if "classifications" not in out:
            cls = ds.get("classifications")
            if cls is not None:
                out["classifications"] = cls
        return out

    def _get_consumer_info(self, consumer_idx):
        """
        Extract consumer information from an exchange.

        :param consumer_idx: Index of the consumer flow.
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
            if decomposed_exclusions.get(loc):
                excluded_subregions.extend(decomposed_exclusions[loc])
            else:
                excluded_subregions.append(loc)

        self.logger.debug(
            "exclusions[%d]: name=%s | refprod=%s | raw=%s | excluded=%s",
            idx,
            name,
            reference_product,
            sorted(exclusions),
            sorted(excluded_subregions),
        )

        return frozenset(excluded_subregions)

    def _dbg_edge_label(self, idx: int) -> str:
        """Human-friendly label for a technosphere position."""
        a = self.position_to_technosphere_flows_lookup.get(idx, {})
        return f"{a.get('name', '?')}[{a.get('reference product')}]@{a.get('location')}#{idx}"

    def _dbg_geo_dump(self, where: str, loc: str | None):
        """Log what GeoResolver knows about a location."""
        if not loc:
            self.logger.debug("geo[%s]: loc=None", where)
            return
        try:
            kids = self.geo.batch(locations=[loc], containing=True) or {}
            pars = self.geo.batch(locations=[loc], containing=False) or {}
            self.logger.debug(
                "geo[%s]: %s | parents=%s | children=%s",
                where,
                loc,
                pars.get(loc, []),
                kids.get(loc, []),
            )
        except Exception as e:
            self.logger.debug("geo[%s]: %s | ERROR: %s", where, loc, e)

    def _dbg_cf_call(
        self,
        tag: str,
        s_info: dict,
        c_info: dict,
        cand_sup: list[str],
        cand_con: list[str],
    ):
        """Uniform pre/post log around CF computations."""
        self.logger.debug(
            "CF-CALL[%s]: s=%s c=%s | cand_sup=%s | cand_con=%s | req_sup=%s | req_con=%s",
            tag,
            {k: s_info.get(k) for k in sorted(self.required_supplier_fields)},
            {k: c_info.get(k) for k in sorted(self.required_consumer_fields)},
            cand_sup,
            cand_con,
            sorted(self.required_supplier_fields),
            sorted(self.required_consumer_fields),
        )

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

        self.lca.lci(factorize=True)

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

        new_version = (
            len(self.biosphere_flows) if self.biosphere_flows else 0,
            len(self.technosphere_flows) if self.technosphere_flows else 0,
        )
        if getattr(self, "_flows_version", None) != new_version:
            self._base_supplier_lookup_bio = None
            self._base_supplier_lookup_tech = None
            self._base_consumer_lookup = None
            self._flows_version = new_version

    def map_exchanges(self):
        """
        Direction-aware matching with per-direction adjacency, indices, and allowlists.
        Uses pivoted set intersections (iterate on the smaller side) and batch pruning.
        Leaves near-misses due to 'location' for later geo steps.
        """

        self._ensure_filtered_lookups_for_current_edges()

        log = self.logger.getChild("map")
        debug = log.isEnabledFor(logging.DEBUG)
        self._initialize_weights()

        DIR_BIO = "biosphere-technosphere"
        DIR_TECH = "technosphere-technosphere"

        # ---- Build adjacency once ---------------------------------------------------
        def build_adj(edges):
            ebs, ebc = defaultdict(set), defaultdict(set)
            rem = set(edges)
            for s, c in rem:
                ebs[s].add(c)
                ebc[c].add(s)
            return rem, ebs, ebc

        rem_bio, ebs_bio, ebc_bio = build_adj(self.biosphere_edges)
        rem_tec, ebs_tec, ebc_tec = build_adj(self.technosphere_edges)

        if not rem_bio and not rem_tec:
            self.eligible_edges_for_next_bio = set()
            self.eligible_edges_for_next_tech = set()
            self._update_unprocessed_edges()
            return

        # Restrict lookups to positions we might touch (cheap, one-time)
        restrict_sup_bio = set(ebs_bio.keys())
        restrict_sup_tec = set(ebs_tec.keys())
        restrict_con = set(ebc_bio.keys()) | set(ebc_tec.keys())

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio,
            restrict_supplier_positions_tech=restrict_sup_tec,
            restrict_consumer_positions=restrict_con,
        )

        # Build per-direction indexes (filtered view)
        supplier_index_bio = (
            build_index(self.supplier_lookup_bio, self.required_supplier_fields)
            if self.supplier_lookup_bio
            else {}
        )
        supplier_index_tec = (
            build_index(self.supplier_lookup_tech, self.required_supplier_fields)
            if self.supplier_lookup_tech
            else {}
        )
        consumer_index = (
            build_index(self.consumer_lookup, self.required_consumer_fields)
            if self.consumer_lookup
            else {}
        )

        allow_bio, allow_tec = set(), set()

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

        # Hot locals
        consumer_lookup = self.consumer_lookup
        reversed_consumer_lookup = self.reversed_consumer_lookup

        # Precompute required field lists (no 'classifications')
        if getattr(self, "_req_sup_nc", None) is None:
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

        if debug:
            log.debug(
                "START map_exchanges | bio_edges=%d | tech_edges=%d | CFs=%d | req_supplier=%s | req_consumer=%s",
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

        # Iterate CFs
        for i, cf in enumerate(tqdm(self.raw_cfs_data, desc="Mapping exchanges")):
            # Early exit if everything got characterized in both directions
            if not rem_bio and not rem_tec:
                break

            s_crit = cf["supplier"]
            c_crit = cf["consumer"]

            # Direction bundle
            dir_name, rem, ebs, ebc, s_index, s_lookup, s_reversed = get_dir_bundle(
                s_crit.get("matrix", "biosphere")
            )
            if not rem:
                if debug:
                    log.debug("CF[%d] dir=%s skipped: no remaining edges.", i, dir_name)
                continue

            # ---------- SUPPLIER side ----------
            norm_s = cf.get("_norm_supplier_cls")  # pre-normalized & sanitized once
            s_class_hits = (
                self._cls_candidates_from_cf_cached(
                    norm_s,
                    (
                        self.cls_prefidx_supplier_bio
                        if dir_name == DIR_BIO
                        else self.cls_prefidx_supplier_tech
                    ),
                    adjacency_keys=None,  # get base frozenset (no allocation)
                )
                if norm_s
                else None
            )

            # Prepare indexed match (cache key includes index context)
            cached_match_with_index.index = s_index
            cached_match_with_index.lookup_mapping = s_lookup
            cached_match_with_index.reversed_lookup = s_reversed
            s_ctx_id = (id(s_index), id(s_lookup), id(s_reversed))

            # Hashable flow minus classifications (precompute this if you want to squeeze more)
            # --- before cached_match_with_index(...) for supplier ---
            s_nonclass = {k: v for k, v in s_crit.items() if k != "classifications"}

            # If supplier criteria are empty (ignoring 'matrix'), treat as wildcard:
            _supplier_keys_wo_matrix = [k for k in s_nonclass.keys() if k != "matrix"]
            if not _supplier_keys_wo_matrix and all(
                k in {"matrix", "classifications"} for k in s_crit.keys()
            ):
                # use ALL suppliers present in this direction (restricted by adjacency)
                s_cands = set(
                    ebs.keys()
                )  # all supplier positions that have outgoing edges
                s_loc_only = set()
                s_loc_required = False
            else:
                # current behavior
                s_out = cached_match_with_index(
                    make_hashable(s_nonclass), req_sup_nc, s_ctx_id
                )
                s_cands = set(s_out.matches)
                if s_class_hits is not None:
                    s_cands &= s_class_hits
                s_loc_only = set(s_out.location_only_rejects)
                if s_class_hits is not None:
                    s_loc_only &= s_class_hits
                s_loc_required = ("location" in s_crit) and (
                    s_crit.get("location") is not None
                )

            # ---------- CONSUMER side ----------
            norm_c = cf.get("_norm_consumer_cls")
            c_class_hits = (
                self._cls_candidates_from_cf_cached(
                    norm_c, self.cls_prefidx_consumer, adjacency_keys=None
                )
                if norm_c
                else None
            )

            cached_match_with_index.index = consumer_index
            cached_match_with_index.lookup_mapping = consumer_lookup
            cached_match_with_index.reversed_lookup = reversed_consumer_lookup
            c_ctx_id = (
                id(consumer_index),
                id(consumer_lookup),
                id(reversed_consumer_lookup),
            )

            c_nonclass = {k: v for k, v in c_crit.items() if k != "classifications"}
            c_out = cached_match_with_index(
                make_hashable(c_nonclass), req_con_nc, c_ctx_id
            )
            c_cands = set(c_out.matches)
            if c_class_hits is not None:
                c_cands &= c_class_hits

            c_loc_only = set(c_out.location_only_rejects)
            if c_class_hits is not None:
                c_loc_only &= c_class_hits
            c_loc_required = ("location" in c_crit) and (
                c_crit.get("location") is not None
            )

            # ---- DEBUG diagnostics (only if needed) ----------------------------------
            if debug and not s_cands:
                s_matches_raw = list(s_out.matches)
                reason = []
                if not s_matches_raw:
                    reason.append("no-index-match")
                else:
                    reason.append(f"raw-matches={len(s_matches_raw)}")
                    if s_class_hits is not None and not (
                        set(s_matches_raw) & s_class_hits
                    ):
                        reason.append("class-filtered-out")
                    if s_class_hits is None:
                        reason.append("no-class-filter")
                log.debug(
                    "CF[%d] dir=%s supplier candidates empty | reasons=%s | s_crit=%s | raw=%d",
                    i,
                    dir_name,
                    ",".join(reason),
                    s_crit,
                    len(s_matches_raw),
                )

            if debug and not c_cands:
                c_matches_raw = list(c_out.matches)
                reason = []
                if not c_matches_raw:
                    reason.append("no-index-match")
                else:
                    reason.append(f"raw-matches={len(c_matches_raw)}")
                    if c_class_hits is not None and not (
                        set(c_matches_raw) & c_class_hits
                    ):
                        reason.append("class-filtered-out")
                    if c_class_hits is None:
                        reason.append("no-class-filter")
                log.debug(
                    "CF[%d] dir=%s consumer candidates empty | reasons=%s | c_crit=%s | raw=%d",
                    i,
                    dir_name,
                    ",".join(reason),
                    c_crit,
                    len(c_matches_raw),
                )

            # ---------- Combine full matches using set intersections ----------
            positions = []
            if s_cands and c_cands:
                sset = s_cands
                cset = c_cands
                ebs_get = ebs.get
                ebc_get = ebc.get

                # Pick the cheaper side to iterate
                iterate_suppliers = len(sset) <= len(cset)

                if iterate_suppliers:
                    # suppliers → consumers
                    for s in list(sset):
                        cs = ebs_get(s)
                        if not cs:
                            continue
                        hit = cs & cset
                        if not hit:
                            continue

                        positions.extend((s, c) for c in hit)

                        # prune rem, ebs, ebc with minimal lookups
                        rem.difference_update((s, c) for c in hit)

                        # supplier out-neigh
                        cs.difference_update(hit)
                        if not cs:
                            # keep empty buckets to avoid costly deletes if you prefer;
                            # but if you want to delete, do it once:
                            del ebs[s]

                        # consumer in-neigh
                        for c in hit:
                            bucket = ebc_get(c)
                            if bucket:
                                bucket.discard(s)
                                if not bucket:
                                    del ebc[c]
                else:
                    # consumers → suppliers
                    for c in list(cset):
                        ss = ebc_get(c)
                        if not ss:
                            continue
                        hit = ss & sset
                        if not hit:
                            continue

                        positions.extend((s, c) for s in hit)

                        rem.difference_update((s, c) for s in hit)

                        # consumer in-neigh
                        ss.difference_update(hit)
                        if not ss:
                            del ebc[c]

                        # supplier out-neigh
                        for s in hit:
                            nb = ebs_get(s)
                            if nb:
                                nb.discard(c)
                                if not nb:
                                    del ebs[s]

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
                if debug:
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
            elif debug:
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

            # ---------- Near-miss allowlists (location-only) --------------------------
            if s_loc_required and s_loc_only and c_cands:
                cset = c_cands
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for s in list(s_loc_only):
                    cs = ebs.get(s)
                    if not cs:
                        continue
                    hit = cs & cset
                    if not hit:
                        continue
                    for c in hit:
                        if (s, c) in rem:
                            bucket.add((s, c))
                            added += 1
                if added:
                    if debug:
                        log.debug(
                            "CF[%d] dir=%s allowlist add (supplier loc-only) | added=%d",
                            i,
                            dir_name,
                            added,
                        )

            if c_loc_required and c_loc_only and s_cands:
                sset = s_cands
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for c in list(c_loc_only):
                    ss = ebc.get(c)
                    if not ss:
                        continue
                    hit = ss & sset
                    if not hit:
                        continue
                    for s in hit:
                        if (s, c) in rem:
                            bucket.add((s, c))
                            added += 1
                if added:
                    if debug:
                        log.debug(
                            "CF[%d] dir=%s allowlist add (consumer loc-only) | added=%d",
                            i,
                            dir_name,
                            added,
                        )

                        for s in list(s_loc_only or []):
                            pass  # (no-op, just anchor for breakpoint)
                        sampled = list(bucket)[:5]
                        self.logger.debug(
                            "allowlist[%s]: added=%d examples=%s",
                            dir_name,
                            added,
                            [
                                (
                                    f"{self._dbg_edge_label(s)} → {self._dbg_edge_label(c)}"
                                    if False
                                    else (s, c)
                                )
                                for (s, c) in sampled
                            ],  # keep indices to avoid any side effect
                        )

            if s_loc_required and c_loc_required and s_loc_only and c_loc_only:
                cset = set(c_loc_only)
                bucket = allow_bio if dir_name == DIR_BIO else allow_tec
                added = 0
                for s in list(s_loc_only):
                    cs = ebs.get(s)
                    if not cs:
                        continue
                    hit = cs & cset
                    if not hit:
                        continue
                    for c in hit:
                        if (s, c) in rem:
                            bucket.add((s, c))
                            added += 1
                if added:
                    if debug:
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

        if debug:
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

        :return: None
        """

        import json, itertools

        _edge_seq = itertools.count(1)

        def _edge_fate(
            phase: str,
            direction: str,
            supplier_idx: int,
            consumer_idx: int,
            supplier_loc,
            consumer_loc,
            reason: str,
            extra: dict | None = None,
        ):
            """Log a structured, single-line fate for this edge."""
            payload = {
                "seq": next(_edge_seq),
                "phase": phase,  # INDEX, GROUP, PASS1, PASS2, END
                "dir": direction,  # biosphere-technosphere / technosphere-technosphere
                "supplier_idx": supplier_idx,
                "consumer_idx": consumer_idx,
                "supplier_loc": supplier_loc,
                "consumer_loc": consumer_loc,
                "reason": reason,
            }
            if extra:
                payload.update(extra)

            # JSON helper: make numpy/pandas/sets serializable
            def _json_default(o):
                import numpy as _np
                import pandas as _pd
                from pathlib import Path as _Path

                if isinstance(o, _np.generic):  # np.int32, np.float64, etc.
                    return o.item()
                if isinstance(o, _np.ndarray):
                    return o.tolist()
                if isinstance(o, (_pd.Timestamp,)):
                    return o.isoformat()
                if isinstance(o, (set, frozenset)):
                    # sort for determinism
                    return sorted(list(o))
                if isinstance(o, _Path):
                    return str(o)
                # last resort: string form
                return str(o)

            tag = " [RER]" if str(consumer_loc).upper() == "RER" else ""
            self.logger.debug(
                "AGG%s %s",
                tag,
                json.dumps(
                    payload, sort_keys=True, ensure_ascii=False, default=_json_default
                ),
            )

        self._ensure_filtered_lookups_for_current_edges()

        # IMPORTANT: rebuild filtered lookups to cover the (current) unprocessed edges
        restrict_sup_bio = {s for s, _ in self.unprocessed_biosphere_edges}
        restrict_sup_tec = {s for s, _ in self.unprocessed_technosphere_edges}
        restrict_con = {c for _, c in self.unprocessed_biosphere_edges} | {
            c for _, c in self.unprocessed_technosphere_edges
        }

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio or None,
            restrict_supplier_positions_tech=restrict_sup_tec or None,
            restrict_consumer_positions=restrict_con or None,
        )

        self._initialize_weights()
        logger.info("Handling static regions…")

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

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
                    _edge_fate(
                        "INDEX",
                        direction,
                        supplier_idx,
                        consumer_idx,
                        supplier_loc=None,
                        consumer_loc=None,
                        reason="ALREADY_PROCESSED",
                    )
                    continue

                consumer_loc = self.consumer_loc.get(consumer_idx)

                if not consumer_loc:
                    _edge_fate(
                        "INDEX",
                        direction,
                        supplier_idx,
                        consumer_idx,
                        supplier_loc=None,
                        consumer_loc=consumer_loc,
                        reason="NO_CONSUMER_LOCATION",
                    )
                    raise ValueError(
                        f"Consumer flow {consumer_idx} has no 'location' field. "
                        "Ensure all consumer flows have a valid location."
                    )

                # Get supplier info first (tech/bio aware) before reading its location
                supplier_info = self._get_supplier_info(supplier_idx, direction)

                if not supplier_info:
                    # nothing usable for this supplier; skip defensively
                    _edge_fate(
                        "INDEX",
                        direction,
                        supplier_idx,
                        consumer_idx,
                        supplier_loc=None,
                        consumer_loc=consumer_loc,
                        reason="NO_SUPPLIER_INFO",
                    )

                    continue
                supplier_loc = supplier_info.get("location")

                edges_index[(consumer_loc, supplier_loc)].append(
                    (supplier_idx, consumer_idx)
                )
                _edge_fate(
                    "INDEX",
                    direction,
                    supplier_idx,
                    consumer_idx,
                    supplier_loc=supplier_loc,
                    consumer_loc=consumer_loc,
                    reason="QUEUED",
                )

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for (consumer_location, supplier_location), edges in edges_index.items():
                if any(
                    x in ("RoW", "RoE") for x in (consumer_location, supplier_location)
                ):
                    for si, ci in edges:
                        _edge_fate(
                            phase="GROUP",
                            direction=direction,
                            supplier_idx=si,
                            consumer_idx=ci,
                            supplier_loc=supplier_location,
                            consumer_loc=consumer_location,
                            reason="SKIP_ROW_ROE",
                        )
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
                    for si, ci in edges:
                        _edge_fate(
                            "GROUP",
                            direction,
                            si,
                            ci,
                            supplier_loc=supplier_location,
                            consumer_loc=consumer_location,
                            reason="SINGLETON_CANDIDATES",
                            extra={
                                "cand_sup": candidate_suppliers_locations,
                                "cand_con": candidate_consumer_locations,
                            },
                        )
                    continue

                self.logger.debug(
                    "aggregate: (con=%s, sup=%s) → cand_sup=%s | cand_con=%s | edges=%d",
                    consumer_location,
                    supplier_location,
                    candidate_suppliers_locations,
                    candidate_consumer_locations,
                    len(edges),
                )
                self._dbg_geo_dump("aggregate/consumer", consumer_location)
                self._dbg_geo_dump("aggregate/supplier", supplier_location)

                for supplier_idx, consumer_idx in edges:

                    supplier_info = self._get_supplier_info(supplier_idx, direction)
                    if not supplier_info:
                        # Nothing useful we can use: skip this edge defensively
                        # (or log at DEBUG)
                        continue

                    consumer_info = self._get_consumer_info(consumer_idx)

                    include_cls_in_sig = any(
                        "classifications" in cf.get("supplier", {})
                        for cf in self.raw_cfs_data
                    )
                    sig_fields = set(self.required_supplier_fields)
                    if include_cls_in_sig:
                        sig_fields.add("classifications")

                    _proj = {
                        k: supplier_info.get(k)
                        for k in sig_fields
                        if supplier_info.get(k) is not None
                    }
                    sig = _equality_supplier_signature_cached(make_hashable(_proj))

                    if sig in self._cached_supplier_keys:
                        _edge_fate(
                            "GROUP",
                            direction,
                            supplier_idx,
                            consumer_idx,
                            supplier_loc=supplier_location,
                            consumer_loc=consumer_location,
                            reason="PREFILTERED",
                            extra={
                                "sig_fields": sorted(sig_fields),
                                "cand_sup": candidate_suppliers_locations[:10],
                                "cand_con": candidate_consumer_locations[:10],
                            },
                        )
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
                        _edge_fate(
                            "GROUP",
                            direction,
                            supplier_idx,
                            consumer_idx,
                            supplier_loc=supplier_location,
                            consumer_loc=consumer_location,
                            reason="REMAINING",
                            extra={
                                "sig_fields": sorted(sig_fields),
                                "cand_sup": candidate_suppliers_locations[:10],
                                "cand_con": candidate_consumer_locations[:10],
                            },
                        )
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

                    self._dbg_cf_call(
                        "AGG-PASS1",
                        supplier_info,
                        consumer_info,
                        candidate_supplier_locations,
                        candidate_consumer_locations,
                    )

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
                            _edge_fate(
                                "PASS1",
                                direction,
                                supplier_idx,
                                consumer_idx,
                                supplier_loc=supplier_info.get("location"),
                                consumer_loc=consumer_info.get("location"),
                                reason="PASS1_APPLIED",
                                extra={"new_cf": str(new_cf)},
                            )
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
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            _,
                            _,
                        ) in group_edges:
                            _edge_fate(
                                "PASS1",
                                direction,
                                supplier_idx,
                                consumer_idx,
                                supplier_loc=supplier_info.get("location"),
                                consumer_loc=consumer_info.get("location"),
                                reason="PASS1_FAIL",
                            )
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

                    self._dbg_cf_call(
                        "AGG-PASS2",
                        dict(s_key),
                        dict(c_key),
                        candidate_suppliers,
                        candidate_consumers,
                    )

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
                            _edge_fate(
                                "PASS2",
                                direction,
                                supplier_idx,
                                consumer_idx,
                                supplier_loc=dict(s_key).get("location"),
                                consumer_loc=dict(c_key).get("location"),
                                reason="PASS2_APPLIED",
                                extra={"new_cf": str(new_cf)},
                            )
                    else:
                        for supplier_idx, consumer_idx in edge_group:
                            _edge_fate(
                                "PASS2",
                                direction,
                                supplier_idx,
                                consumer_idx,
                                supplier_loc=dict(s_key).get("location"),
                                consumer_loc=dict(c_key).get("location"),
                                reason="PASS2_FAIL",
                            )
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

        :return: None
        """

        self._ensure_filtered_lookups_for_current_edges()

        # IMPORTANT: rebuild filtered lookups to cover the (current) unprocessed edges
        restrict_sup_bio = {s for s, _ in self.unprocessed_biosphere_edges}
        restrict_sup_tec = {s for s, _ in self.unprocessed_technosphere_edges}
        restrict_con = {c for _, c in self.unprocessed_biosphere_edges} | {
            c for _, c in self.unprocessed_technosphere_edges
        }

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio or None,
            restrict_supplier_positions_tech=restrict_sup_tec or None,
            restrict_consumer_positions=restrict_con or None,
        )

        self._initialize_weights()
        logger.info("Handling dynamic regions…")

        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        raw_exclusion_locs = {
            loc
            for locs in self.technosphere_flows_lookup.values()
            for loc in locs
            if str(loc).upper() not in {"ROW", "ROE"}
        }
        decomposed_exclusions = self.geo.batch(
            locations=list(raw_exclusion_locs), containing=True
        )
        decomposed_exclusions = frozenset(
            (k, tuple(v)) for k, v in decomposed_exclusions.items()
        )

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

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
                supplier_info = self._get_supplier_info(supplier_idx, direction)
                if not supplier_info:
                    # Nothing useful we can use: skip this edge defensively
                    # (or log at DEBUG)
                    continue

                supplier_loc = supplier_info.get("location")
                consumer_loc = self.consumer_loc.get(consumer_idx)

                # Skip if neither side is dynamic
                # Identify dynamic role
                _is_dyn = lambda x: isinstance(x, str) and x.upper() in {"ROW", "ROE"}
                if not (_is_dyn(supplier_loc) or _is_dyn(consumer_loc)):
                    continue

                dynamic_supplier = _is_dyn(supplier_loc)
                dynamic_consumer = _is_dyn(consumer_loc)

                # Resolve fallback candidate locations
                if dynamic_supplier:
                    suppliers_excluded_subregions = self._extract_excluded_subregions(
                        supplier_idx, decomposed_exclusions
                    )
                    candidate_suppliers_locs = resolve_candidate_locations(
                        geo=self.geo,
                        location="GLO",
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        exceptions=suppliers_excluded_subregions,
                        supplier=True,
                    )
                    candidate_suppliers_locs = [
                        loc for loc in candidate_suppliers_locs if loc != "GLO"
                    ]
                    candidate_suppliers_locs = sorted(set(candidate_suppliers_locs))
                else:
                    if supplier_loc is None:
                        candidate_suppliers_locs = [
                            "__ANY__",
                        ]
                    else:
                        candidate_suppliers_locs = [supplier_loc]

                if dynamic_consumer:
                    consumers_excluded_subregions = self._extract_excluded_subregions(
                        consumer_idx, decomposed_exclusions
                    )
                    candidate_consumers_locs = resolve_candidate_locations(
                        geo=self.geo,
                        location="GLO",
                        weights=frozenset(k for k, v in self.weights.items()),
                        containing=True,
                        exceptions=consumers_excluded_subregions,
                        supplier=False,
                    )
                    candidate_consumers_locs = [
                        loc for loc in candidate_consumers_locs if loc != "GLO"
                    ]
                    candidate_consumers_locs = sorted(set(candidate_consumers_locs))

                    self.logger.debug(
                        "dynamic-cands: consumer=RoW | candidates=%d (e.g. %s...) | excluded=%d",
                        len(candidate_consumers_locs),
                        candidate_consumers_locs[:20],
                        len(consumers_excluded_subregions),
                    )

                else:
                    if consumer_loc is None:
                        candidate_consumers_locs = [
                            "__ANY__",
                        ]
                    else:
                        candidate_consumers_locs = [consumer_loc]

                if dynamic_consumer and not candidate_consumers_locs:
                    self.logger.debug(
                        "dynamic: RoW consumer collapsed to empty set after exclusions; deferring to global pass"
                    )
                    continue

                # project supplier info to the required fields (+classifications) before hashing
                include_cls_in_sig = any(
                    "classifications" in cf.get("supplier", {})
                    for cf in self.raw_cfs_data
                )
                sig_fields = set(self.required_supplier_fields)
                if include_cls_in_sig:
                    sig_fields.add("classifications")

                _proj = {
                    k: supplier_info.get(k)
                    for k in sig_fields
                    if supplier_info.get(k) is not None
                }
                sig = _equality_supplier_signature_cached(make_hashable(_proj))

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

            # Pass 1 (corrected): compute per unique (cand_sup, cand_con, consumer_sig)
            if len(prefiltered_groups) > 0:
                for sig, group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing dynamic groups (pass 1)",
                ):
                    # Build a small memo to avoid recomputing identical combos in this group
                    memo = {}

                    def _consumer_sig(consumer_info: dict) -> tuple:
                        """Hashable, filtered consumer signature (only required fields + classifications if used)."""
                        fields = set(self.required_consumer_fields)
                        if any(
                            "classifications" in cf.get("consumer", {})
                            for cf in self.raw_cfs_data
                        ):
                            fields.add("classifications")
                        proj = {
                            k: consumer_info.get(k)
                            for k in fields
                            if consumer_info.get(k) is not None
                        }
                        return make_hashable(proj)

                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        cand_sup,
                        cand_con,
                    ) in group_edges:

                        # Deterministic candidate lists (avoid order-dependent averaging)
                        cand_sup_s = tuple(sorted(set(cand_sup)))
                        cand_con_s = tuple(sorted(set(cand_con)))

                        c_sig = _consumer_sig(consumer_info)
                        memo_key = (cand_sup_s, cand_con_s, c_sig)

                        if memo_key not in memo:
                            # Optional debug hook
                            self._dbg_cf_call(
                                "DYN-PASS1",
                                supplier_info,
                                consumer_info,
                                list(cand_sup_s),
                                list(cand_con_s),
                            )

                            new_cf, matched_cf_obj, agg_uncertainty = (
                                compute_average_cf(
                                    candidate_suppliers=list(cand_sup_s),
                                    candidate_consumers=list(cand_con_s),
                                    supplier_info=supplier_info,
                                    consumer_info=consumer_info,
                                    required_supplier_fields=self.required_supplier_fields,
                                    required_consumer_fields=self.required_consumer_fields,
                                    cf_index=self.cf_index,
                                )
                            )
                            memo[memo_key] = (new_cf, matched_cf_obj, agg_uncertainty)

                        new_cf, matched_cf_obj, agg_uncertainty = memo[memo_key]

                        if new_cf:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=agg_uncertainty,
                            )
                            self.logger.debug(
                                "dynamic-check: supplier name=%s | supplier refprod=%s | cons name=%s | cons refprod=%s | new_cf=%s | supplier_cands=%s | consumer_cands=%s",
                                supplier_info.get("name"),
                                supplier_info.get("reference product"),
                                consumer_info.get("name"),
                                consumer_info.get("reference product"),
                                new_cf,
                                list(cand_sup_s)[:10],
                                list(cand_con_s)[:10],
                            )
                        else:
                            self.logger.warning(
                                "Fallback CF could not be computed for supplier=%s, consumer=%s "
                                "with candidate suppliers=%s and consumers=%s",
                                supplier_info.get("location"),
                                consumer_info.get("location"),
                                list(cand_sup_s),
                                list(cand_con_s),
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
                    self._dbg_cf_call(
                        "DYN-PASS2",
                        dict(s_key),
                        dict(c_key),
                        candidate_supplier_locations,
                        candidate_consumer_locations,
                    )

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

        :return: None
        """

        self._ensure_filtered_lookups_for_current_edges()

        # IMPORTANT: rebuild filtered lookups to cover the (current) unprocessed edges
        restrict_sup_bio = {s for s, _ in self.unprocessed_biosphere_edges}
        restrict_sup_tec = {s for s, _ in self.unprocessed_technosphere_edges}
        restrict_con = {c for _, c in self.unprocessed_biosphere_edges} | {
            c for _, c in self.unprocessed_technosphere_edges
        }

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio or None,
            restrict_supplier_positions_tech=restrict_sup_tec or None,
            restrict_consumer_positions=restrict_con or None,
        )

        self._initialize_weights()
        logger.info("Handling contained locations…")

        def _geo_contains(container: str, member: str) -> bool:
            """Return True if `container` geographically contains `member`, robust to API flag semantics and case."""
            if not container or not member:
                return False

            C_raw = str(container).strip()
            M_raw = str(member).strip()

            def N(x):  # normalize for comparisons
                return str(x).strip().upper()

            def q(loc: str, containing: bool) -> list[str]:
                try:
                    mp = self.geo.batch(locations=[loc], containing=containing) or {}
                    return mp.get(loc, []) or []
                except Exception as e:
                    self.logger.debug(
                        "geo-contains: batch error loc=%s containing=%s err=%s",
                        loc,
                        containing,
                        e,
                    )
                    return []

            # Collect both interpretations, then normalize once
            containers_of_member = set(map(N, q(M_raw, True))) | set(
                map(N, q(M_raw, False))
            )
            children_of_container = set(map(N, q(C_raw, False))) | set(
                map(N, q(C_raw, True))
            )

            C = N(C_raw)
            M = N(M_raw)

            res = (C in containers_of_member) or (M in children_of_container)

            return res

        # Respect wildcard suppliers in method keys (e.g., ('__ANY__','RER'))
        supplier_wildcard = any(k[0] == "__ANY__" for k in self.weights.keys())
        available_consumer_locs = sorted({loc for _, loc in self.weights.keys()})

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

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

                # Get supplier info first (tech/bio aware) before reading its location
                supplier_info = self._get_supplier_info(supplier_idx, direction)

                if not supplier_info:
                    continue
                supplier_loc = supplier_info.get("location")

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

                # Supplier: if CFs use wildcard on supplier side, stick to '__ANY__'.
                # Otherwise, keep the supplier's own location (no up/down traversal here).

                if supplier_wildcard:
                    candidate_suppliers_locations = ["__ANY__"]
                elif supplier_location is None:
                    candidate_suppliers_locations = ["__ANY__"]
                else:
                    candidate_suppliers_locations = [supplier_location]

                # Consumer: climb to the nearest containing region (e.g., IT → RER),
                # limited to locations actually present on the consumer side of method keys.

                # Consumer: climb to the nearest containing region present in the method (prefer non-GLO)
                if consumer_location is None:
                    candidate_consumer_locations = ["__ANY__"]
                else:
                    # Consider only method regions on the consumer side (exclude __ANY__/GLO at first)
                    available_non_global = [
                        loc
                        for loc in available_consumer_locs
                        if loc not in {"__ANY__", "GLO"}
                    ]

                    # Try to find the nearest method region that contains the inventory region
                    # Prioritize a stable order that tends to pick the smallest sensible container;
                    # if you want to strongly prefer RER when present, keep it first.
                    ordered = sorted(
                        available_non_global, key=lambda x: (x != "RER", x)
                    )

                    self._dbg_geo_dump("contained/consumer", consumer_location)
                    # Probe each candidate method region (non-GLO) once so we see what children they claim
                    for _cand in [
                        loc
                        for loc in available_consumer_locs
                        if loc not in {"__ANY__", "GLO"}
                    ][:10]:
                        self._dbg_geo_dump(f"contained/method-cand[{_cand}]", _cand)

                    nearest = next(
                        (
                            cand
                            for cand in ordered
                            if _geo_contains(cand, consumer_location)
                        ),
                        None,
                    )

                    self.logger.debug(
                        "contained: consumer %s -> nearest method container %s (ordered candidates=%s)",
                        consumer_location,
                        nearest,
                        ordered,
                    )

                    if nearest is not None:
                        candidate_consumer_locations = [nearest]
                    else:
                        # Nothing but GLO contains this region (or geo data is missing). Leave empty here
                        # so this pass skips; the global pass will handle it.
                        candidate_consumer_locations = []

                # If we couldn't find any suitable consumer region to climb to, skip.

                if not candidate_consumer_locations:
                    continue

                for supplier_idx, consumer_idx in edges:
                    supplier_info = self._get_supplier_info(supplier_idx, direction)
                    if not supplier_info:
                        # Nothing useful we can use: skip this edge defensively
                        # (or log at DEBUG)
                        continue
                    consumer_info = self._get_consumer_info(consumer_idx)

                    include_cls_in_sig = any(
                        "classifications" in cf.get("supplier", {})
                        for cf in self.raw_cfs_data
                    )
                    sig_fields = set(self.required_supplier_fields)
                    if include_cls_in_sig:
                        sig_fields.add("classifications")

                    _proj = {
                        k: supplier_info.get(k)
                        for k in sig_fields
                        if supplier_info.get(k) is not None
                    }
                    sig = _equality_supplier_signature_cached(make_hashable(_proj))

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

                    self._dbg_cf_call(
                        "CONT-PASS1",
                        supplier_info,
                        consumer_info,
                        candidate_supplier_locations,
                        candidate_consumer_locations,
                    )

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

                    self._dbg_cf_call(
                        "CONT-PASS2",
                        supplier_info,
                        consumer_info,
                        candidate_suppliers,
                        candidate_consumers,
                    )

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

        :return: None
        """

        self._ensure_filtered_lookups_for_current_edges()

        # IMPORTANT: rebuild filtered lookups to cover the (current) unprocessed edges
        restrict_sup_bio = {s for s, _ in self.unprocessed_biosphere_edges}
        restrict_sup_tec = {s for s, _ in self.unprocessed_technosphere_edges}
        restrict_con = {c for _, c in self.unprocessed_biosphere_edges} | {
            c for _, c in self.unprocessed_technosphere_edges
        }

        self._preprocess_lookups(
            restrict_supplier_positions_bio=restrict_sup_bio or None,
            restrict_supplier_positions_tech=restrict_sup_tec or None,
            restrict_consumer_positions=restrict_con or None,
        )

        self._initialize_weights()
        logger.info("Handling remaining exchanges…")

        # Resolve candidate locations for GLO once using utility
        # NOTE: containing=False → return contained regions of GLO (i.e., the world)
        global_supplier_locs = resolve_candidate_locations(
            geo=self.geo,
            location="GLO",
            weights=frozenset(k for k, v in self.weights.items()),
            containing=False,
            supplier=True,
        )
        global_consumer_locs = resolve_candidate_locations(
            geo=self.geo,
            location="GLO",
            weights=frozenset(k for k, v in self.weights.items()),
            containing=False,
            supplier=False,
        )

        # If the consumer side is allowed to use a CF at GLO directly, make sure "GLO" is included
        if "GLO" not in global_consumer_locs:
            global_consumer_locs = ["GLO"] + list(global_consumer_locs)

        # If supplier side is wildcard-only, keep that wildcard as the candidate
        if not global_supplier_locs:
            sup_keys = {k[0] for k in self.weights.keys()}
            if "__ANY__" in sup_keys:
                global_supplier_locs = ["__ANY__"]

        supplier_wildcard = any(k[0] == "__ANY__" for k in self.weights.keys())

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:

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

                # Get supplier info first (tech/bio aware) before reading its location
                supplier_info = self._get_supplier_info(supplier_idx, direction)

                if not supplier_info:
                    continue
                supplier_loc = supplier_info.get("location")

                edges_index[(consumer_loc, supplier_loc)].append(
                    (supplier_idx, consumer_idx)
                )

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for (consumer_location, supplier_location), edges in edges_index.items():

                if supplier_wildcard:
                    candidate_suppliers_locations = ["__ANY__"]
                elif supplier_location is None:
                    candidate_suppliers_locations = ["__ANY__"]
                else:
                    candidate_suppliers_locations = global_supplier_locs

                if consumer_location is None:
                    candidate_consumers_locations = [
                        "__ANY__",
                    ]
                else:
                    candidate_consumers_locations = global_consumer_locs

                for supplier_idx, consumer_idx in edges:

                    supplier_info = self._get_supplier_info(supplier_idx, direction)
                    if not supplier_info:
                        # Nothing useful we can use: skip this edge defensively
                        # (or log at DEBUG)
                        continue
                    consumer_info = self._get_consumer_info(consumer_idx)

                    include_cls_in_sig = any(
                        "classifications" in cf.get("supplier", {})
                        for cf in self.raw_cfs_data
                    )
                    sig_fields = set(self.required_supplier_fields)
                    if include_cls_in_sig:
                        sig_fields.add("classifications")

                    _proj = {
                        k: supplier_info.get(k)
                        for k in sig_fields
                        if supplier_info.get(k) is not None
                    }
                    sig = _equality_supplier_signature_cached(make_hashable(_proj))

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

                # ---- Pass 1 (prefiltered_groups) ----
                if len(prefiltered_groups) > 0:
                    for sig, group_edges in tqdm(
                        prefiltered_groups.items(),
                        desc="Processing global groups (pass 1)",
                    ):
                        supplier_info = group_edges[0][2]
                        consumer_info = group_edges[0][3]

                        # 1) Try DIRECT GLO on the CONSUMER side
                        #    Supplier candidates: keep as-is if present, else "__ANY__"
                        if supplier_wildcard:
                            direct_sup_candidates = ["__ANY__"]
                        else:
                            sup_loc = supplier_info.get("location")
                            direct_sup_candidates = (
                                [sup_loc] if sup_loc is not None else []
                            )
                        direct_con_candidates = ["GLO"]

                        self._dbg_cf_call(
                            "GLO-PASS1",
                            supplier_info,
                            consumer_info,
                            direct_sup_candidates,
                            direct_con_candidates,
                        )

                        # compute_average_cf already ignores fields not present in CFs,
                        # so if supplier 'location' isn't in CF schema, it won't block matches.
                        glo_cf, matched_cf_obj, glo_unc = compute_average_cf(
                            candidate_suppliers=direct_sup_candidates,
                            candidate_consumers=direct_con_candidates,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            required_supplier_fields=self.required_supplier_fields,
                            required_consumer_fields=self.required_consumer_fields,
                            cf_index=self.cf_index,
                        )

                        if glo_cf != 0:
                            for supplier_idx, consumer_idx, _, _, _, _ in group_edges:
                                add_cf_entry(
                                    cfs_mapping=self.cfs_mapping,
                                    supplier_info=supplier_info,
                                    consumer_info=consumer_info,
                                    direction=direction,
                                    indices=[(supplier_idx, consumer_idx)],
                                    value=glo_cf,
                                    uncertainty=(
                                        glo_unc
                                        if glo_unc is not None
                                        else (
                                            matched_cf_obj.get("uncertainty")
                                            if matched_cf_obj
                                            else None
                                        )
                                    ),
                                )
                            continue  # done with this group

                        # 2) Fallback: weighted average over GLOBAL LOCATIONS
                        self._dbg_cf_call(
                            "GLO-AVG",
                            supplier_info,
                            consumer_info,
                            (
                                ["__ANY__"]
                                if supplier_wildcard
                                else global_supplier_locs
                            ),
                            global_consumer_locs,
                        )
                        avg_cf, matched_cf_obj, agg_uncertainty = compute_average_cf(
                            candidate_suppliers=(
                                ["__ANY__"]
                                if supplier_wildcard
                                else global_supplier_locs
                            ),
                            candidate_consumers=global_consumer_locs,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            required_supplier_fields=self.required_supplier_fields,
                            required_consumer_fields=self.required_consumer_fields,
                            cf_index=self.cf_index,
                        )

                        if avg_cf != 0:
                            for supplier_idx, consumer_idx, _, _, _, _ in group_edges:
                                add_cf_entry(
                                    cfs_mapping=self.cfs_mapping,
                                    supplier_info=supplier_info,
                                    consumer_info=consumer_info,
                                    direction=direction,
                                    indices=[(supplier_idx, consumer_idx)],
                                    value=avg_cf,
                                    uncertainty=agg_uncertainty,
                                )
                        else:
                            self.logger.warning(
                                f"Fallback CF could not be computed for supplier={supplier_info}, consumer={consumer_info} "
                                f"with candidate suppliers={ (['__ANY__'] if supplier_wildcard else global_supplier_locs) } and consumers={global_consumer_locs}"
                            )

                # ---- Pass 2 (grouped_edges) ----
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
                        grouped_edges.items(), desc="Processing global groups (pass 2)"
                    ):
                        # 1) Try DIRECT GLO (consumer)
                        if supplier_wildcard:
                            direct_sup_candidates = ["__ANY__"]
                        else:
                            sup_loc = s_key.get("location")
                            direct_sup_candidates = (
                                [sup_loc] if sup_loc is not None else []
                            )
                        direct_con_candidates = ["GLO"]

                        glo_cf, matched_cf_obj, glo_unc = compute_cf_memoized(
                            s_key, c_key, direct_sup_candidates, direct_con_candidates
                        )

                        if glo_cf != 0:
                            for supplier_idx, consumer_idx in edge_group:
                                add_cf_entry(
                                    cfs_mapping=self.cfs_mapping,
                                    supplier_info=dict(s_key),
                                    consumer_info=dict(c_key),
                                    direction=direction,
                                    indices=[(supplier_idx, consumer_idx)],
                                    value=glo_cf,
                                    uncertainty=(
                                        glo_unc
                                        if glo_unc is not None
                                        else (
                                            matched_cf_obj.get("uncertainty")
                                            if matched_cf_obj
                                            else None
                                        )
                                    ),
                                )
                            continue

                        # 2) Fallback to weighted average across global locations
                        avg_cf, matched_cf_obj, agg_uncertainty = compute_cf_memoized(
                            s_key,
                            c_key,
                            (
                                ["__ANY__"]
                                if supplier_wildcard
                                else global_supplier_locs
                            ),
                            global_consumer_locs,
                        )

                        if avg_cf != 0:
                            for supplier_idx, consumer_idx in edge_group:
                                add_cf_entry(
                                    cfs_mapping=self.cfs_mapping,
                                    supplier_info=dict(s_key),
                                    consumer_info=dict(c_key),
                                    direction=direction,
                                    indices=[(supplier_idx, consumer_idx)],
                                    value=avg_cf,
                                    uncertainty=(
                                        agg_uncertainty
                                        if agg_uncertainty is not None
                                        else (
                                            matched_cf_obj.get("uncertainty")
                                            if matched_cf_obj
                                            else None
                                        )
                                    ),
                                )
                        else:
                            self.logger.warning(
                                f"Fallback CF could not be computed for supplier={s_key}, consumer={c_key} "
                                f"with candidate suppliers={ (['__ANY__'] if supplier_wildcard else global_supplier_locs) } and consumers={global_consumer_locs}"
                            )

        self._update_unprocessed_edges()

    def apply_strategies(self, strategies: list[str] | None = None) -> None:
        """
        Execute mapping strategies (strings only) in order.

        If `strategies` is None, read from:
          self.method_metadata["strategies"] (must be a list of strings).

        Valid names:
          - "map_exchanges"
          - "map_aggregate_locations"
          - "map_dynamic_locations"
          - "map_contained_locations"
          - "map_remaining_locations_to_global"

        :params strategies: list of strategy names to apply in order, or None to read from metadata.
        :return: None

        """

        # ---- discover strategies from metadata if not provided
        if strategies is None:
            md = getattr(self, "method_metadata", None) or {}
            strategies = md.get("strategies")

        if strategies is None:
            self.logger.info("No 'strategies' found; nothing to apply.")
            return self

        if not isinstance(strategies, (list, tuple)) or not all(
            isinstance(s, str) for s in strategies
        ):
            raise TypeError("'strategies' must be a list/tuple of strings")

        # ---- dispatch table
        dispatch = {
            "map_exchanges": getattr(self, "map_exchanges", None),
            "map_aggregate_locations": getattr(self, "map_aggregate_locations", None),
            "map_dynamic_locations": getattr(self, "map_dynamic_locations", None),
            "map_contained_locations": getattr(self, "map_contained_locations", None),
            "map_remaining_locations_to_global": getattr(
                self, "map_remaining_locations_to_global", None
            ),
        }

        # ---- validate names
        for name in strategies:
            if name not in dispatch or not callable(dispatch[name]):
                raise AttributeError(f"Unknown or unavailable strategy '{name}'.")

        # ---- ensure inventory is ready
        edges_ready = not (
            (self.biosphere_edges is None and self.technosphere_edges is None)
            or (not self.biosphere_edges and not self.technosphere_edges)
        )
        if not edges_ready:
            self.lci()

        # ---- execute

        self.logger.info("Applying strategies: %s", strategies)

        for name in strategies:
            fn = dispatch[name]
            t0 = time.perf_counter()
            self.logger.info("Running %s()", name)
            fn()
            self.logger.info("Finished %s in %.3fs", name, time.perf_counter() - t0)

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
            self.characterization_matrix = make_coo_deterministic(
                self.characterization_matrix
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

            self._last_eval_scenario_name = scenario_name
            self._last_eval_scenario_idx = scenario_idx

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
                        "positions": sorted(cf["positions"]),
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

        # Decide matrix type from the method (stable across runs), not from transient edge sets
        only_tech = all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        )
        is_biosphere = not only_tech

        # Pick inventory once
        inventory = (
            self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
        )
        if inventory is None:
            raise RuntimeError(
                f"Inventory matrix for {'biosphere' if is_biosphere else 'technosphere'} is None. "
                "Ensure lci() was called and that matrix-type detection does not rely on edge sets."
            )

        if self.use_distributions and self.iterations > 1:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )

            inventory_coo = sparse.COO.from_scipy_sparse(inventory)
            inventory_coo = make_coo_deterministic(inventory_coo)
            inv_expanded = inventory_coo[:, :, None]

            # Element-wise multiply
            characterized = self.characterization_matrix * inv_expanded

            # Sum across dimensions i and j to get 1 value per iteration
            self.characterized_inventory = characterized
            self.score = characterized.sum(axis=(0, 1), dtype=np.float64)

        else:
            # --- Deterministic path with a small guard against rare NotImplemented
            cm = self.characterization_matrix.tocsr()
            inv = inventory.tocsr()  # ensure CSR–CSR
            prod = cm.multiply(inv)
            if prod is NotImplemented:  # very rare, but just in case
                prod = inv.multiply(cm)
            self.characterized_inventory = prod
            self.score = prod.sum(dtype=np.float64)

    # --- Add these helpers inside EdgeLCIA -----------------------------------
    def _covered_positions_from_characterization(self) -> set[tuple[int, int]]:
        """
        Return the set of (i, j) positions that already have CF values
        in the current characterization matrix.
        Works for both 2D SciPy CSR and 3D sparse.COO matrices.
        """
        if self.characterization_matrix is None:
            return set()

        # Uncertainty mode: 3D (i, j, k) COO
        if isinstance(self.characterization_matrix, sparse.COO):
            # coords shape: (3, N); take unique (i, j)
            if self.characterization_matrix.coords.size == 0:
                return set()
            i = self.characterization_matrix.coords[0]
            j = self.characterization_matrix.coords[1]
            return set(zip(map(int, i), map(int, j)))

        # Deterministic mode: 2D SciPy sparse
        ii, jj = self.characterization_matrix.nonzero()
        return set(zip(ii.tolist(), jj.tolist()))

    def _evaluate_cf_value_for_redo(self, cf: dict, scenario_idx, scenario_name):
        """
        Deterministic path: evaluate a single CF value for the redo.
        Mirrors the logic in evaluate_cfs() for a single entry.
        """
        if isinstance(cf["value"], str):
            try:
                params = self._resolve_parameters_for_scenario(
                    scenario_idx, scenario_name
                )
                return float(
                    safe_eval_cached(
                        cf["value"],
                        parameters=params,
                        scenario_idx=scenario_idx,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate symbolic CF '{cf['value']}'. Error: {e}"
                )
                return 0.0
        else:
            return float(cf["value"])

    def redo_lcia(
        self,
        demand: dict | None = None,
        *,
        scenario_idx: int | str | None = None,
        scenario: str | None = None,
        run_aggregate: bool = True,
        run_dynamic: bool = True,
        run_contained: bool = True,
        run_global: bool = True,
        recompute_score: bool = True,
    ) -> None:
        """
        Re-run LCI, preserve the existing characterization_matrix, and only map
        CFs for *new* exchanges that don't already have CF coverage.

        Typical usage after you’ve already done:
            lci(); map_exchanges(); (other mapping); evaluate_cfs(); lcia()

        Parameters
        ----------
        scenario_idx : int|str, optional
            Scenario index/year to use if we need to evaluate numeric CFs
            for newly mapped exchanges (deterministic mode).
            Defaults to the last-used one if available, otherwise 0 or method default.
        scenario : str, optional
            Scenario name to use for evaluating symbolic CFs (deterministic mode).
            Defaults to the last-used one or the class default.
        run_aggregate, run_dynamic, run_contained, run_global : bool
            Which fallback mapping passes to run for the *new* edges.
        recompute_score : bool
            If True, recompute the LCIA score using the updated inventory.

        Behavior
        --------
        - Keeps self.characterization_matrix as-is and adds entries for newly mapped edges.
        - In deterministic mode, also extends self.scenario_cfs with the new entries
          so downstream reporting stays consistent.
        - In uncertainty mode, samples new CFs consistently using the same seeding
          scheme used in evaluate_cfs() and appends them into the 3D COO.

        Notes
        -----
        - This method will NOT remove CFs for edges that disappeared from the inventory;
          it only adds CFs for the new edges. If you want a “full refresh”, call
          the usual pipeline again.
        """

        if self.characterization_matrix is None:
            raise RuntimeError(
                "redo_lcia() requires an existing characterization_matrix. "
                "Run the normal pipeline (map/evaluate) once before calling this."
            )

        # --- Diagnostics: starting nnz
        if isinstance(self.characterization_matrix, sparse.COO):
            start_nnz = len(self.characterization_matrix.data)
        else:
            start_nnz = self.characterization_matrix.nnz
        self.logger.info(f"Starting characterization_matrix nnz = {start_nnz}")

        # 0) Update demand vector if user passed one
        if demand is not None:
            self.lca.demand.clear()
            self.lca.demand.update(demand)

        # Decide direction (tech-only vs bio) from CFs (doesn't require lci)
        only_tech = all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        )

        # 2) Recompute inventory & edges for the *new* demand
        self.lca.redo_lci(demand=demand)  # updates matrices

        only_tech = all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        )

        # Recompute CURRENT edges from fresh matrices
        if only_tech:
            # refresh helper & edges
            self.technosphere_flow_matrix = build_technosphere_edges_matrix(
                self.lca.technosphere_matrix, self.lca.supply_array
            )
            current_edges = set(zip(*self.technosphere_flow_matrix.nonzero()))
        else:
            current_edges = set(zip(*self.lca.inventory.nonzero()))

        # Edges that already have CF coverage in the existing characterization matrix
        covered = self._covered_positions_from_characterization()
        # Persistently failed edges (don’t thrash on them)
        failed = self._failed_edges_tech if only_tech else self._failed_edges_bio

        # --- Use cumulative "ever seen" edges to avoid rescanning after tiny runs
        if only_tech:
            ever_seen = self._ever_seen_edges_tech
        else:
            ever_seen = self._ever_seen_edges_bio

        # Seed ever_seen the first time with the best baseline we have
        if not ever_seen:
            baseline_seed = set()
            if only_tech:
                if self._last_edges_snapshot_tech:
                    baseline_seed = set(self._last_edges_snapshot_tech)
                elif self._last_nonempty_edges_snapshot_tech:
                    baseline_seed = set(self._last_nonempty_edges_snapshot_tech)
                elif self.technosphere_flow_matrix is not None:
                    baseline_seed = set(zip(*self.technosphere_flow_matrix.nonzero()))
            else:
                if self._last_edges_snapshot_bio:
                    baseline_seed = set(self._last_edges_snapshot_bio)
                elif self._last_nonempty_edges_snapshot_bio:
                    baseline_seed = set(self._last_nonempty_edges_snapshot_bio)
                elif getattr(self.lca, "inventory", None) is not None:
                    baseline_seed = set(zip(*self.lca.inventory.nonzero()))
            ever_seen |= baseline_seed

        # Compute new edges strictly as (current − covered − failed − ever_seen)
        new_edges = current_edges - covered - failed - ever_seen

        # --- Restrict mapping to *only* the newly discovered edges
        if only_tech:
            self.biosphere_edges = set()
            self.technosphere_edges = set(new_edges)
        else:
            self.technosphere_edges = set()
            self.biosphere_edges = set(new_edges)

        # Persist the CURRENT snapshot. Also update the "non-empty" snapshot only when non-empty.
        if only_tech:
            self._last_edges_snapshot_tech = set(current_edges)
            if current_edges:
                self._last_nonempty_edges_snapshot_tech = set(current_edges)
        else:
            self._last_edges_snapshot_bio = set(current_edges)
            if current_edges:
                self._last_nonempty_edges_snapshot_bio = set(current_edges)

        # Extend the cumulative history so future runs won't rescan these
        if only_tech:
            self._ever_seen_edges_tech |= new_edges
        else:
            self._ever_seen_edges_bio |= new_edges

        self.logger.info(
            f"Identified {len(new_edges)} new edges to map "
            f"(current={len(current_edges)}, covered={len(covered)}, ever_seen={len(ever_seen)}, failed={len(failed)})"
        )

        if not new_edges:
            self.logger.info("redo_lcia(): No new exchanges to map.")
            if recompute_score:
                self.lcia()
            return

        # 3) Map only the new edges: snapshot cfs_mapping length to capture the delta later
        baseline_len = len(self.cfs_mapping)

        # Primary mapping on the restricted edge set
        self.map_exchanges()

        # Optional fallback passes (these operate only on unprocessed edges, which we’ve
        # already restricted to the new edges in step 2)
        if run_aggregate:
            self.map_aggregate_locations()
        if run_dynamic:
            self.map_dynamic_locations()
        if run_contained:
            self.map_contained_locations()
        if run_global:
            self.map_remaining_locations_to_global()

        # Identify the CF entries created in this redo
        new_cf_entries = self.cfs_mapping[baseline_len:]

        self.logger.info(f"Mapping produced {len(new_cf_entries)} new CF entries")

        if not new_cf_entries:
            self.logger.info("redo_lcia(): Mapping produced no applicable CFs.")
            # These 'new_edges' were attempted and still have no CF — remember them as failed
            if only_tech:
                self._failed_edges_tech |= set(new_edges)
            else:
                self._failed_edges_bio |= set(new_edges)
            if recompute_score:
                self.lcia()
            return

        # 4) Apply those *new* CFs into the existing characterization_matrix
        if self.use_distributions and self.iterations > 1:
            # Uncertainty mode: append (i, j, k) samples to 3D COO
            cm = self.characterization_matrix
            assert isinstance(
                cm, sparse.COO
            ), "Expected sparse.COO in uncertainty mode."

            # Collect coords/data to append
            coords_i, coords_j, coords_k, data = [], [], [], []
            sample_cache = {}

            for cf in new_cf_entries:
                # Draw (or reuse) samples for this distribution/spec
                key = make_distribution_key(cf)
                if key is None:
                    samples = sample_cf_distribution(
                        cf=cf,
                        n=self.iterations,
                        parameters=self.parameters,
                        random_state=self.random_state,
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

            if data:
                # Concatenate to existing COO
                new_coords = np.array([coords_i, coords_j, coords_k])
                new_data = np.array(data)
                # Merge
                merged_coords = np.concatenate([cm.coords, new_coords], axis=1)
                merged_data = np.concatenate([cm.data, new_data])
                self.characterization_matrix = sparse.COO(
                    coords=merged_coords, data=merged_data, shape=cm.shape
                )
                self.characterization_matrix = make_coo_deterministic(
                    self.characterization_matrix
                )

        else:
            # Deterministic mode: set values directly in the existing 2D matrix
            cm = self.characterization_matrix  # SciPy CSR
            # Decide scenario context (use last known if possible)
            # Decide scenario context (prefer explicit args, then last-used, then class default, then first available key, else None)
            scenario_name = (
                scenario
                if scenario is not None
                else (
                    self._last_eval_scenario_name
                    if getattr(self, "_last_eval_scenario_name", None) is not None
                    else (
                        self.scenario
                        if self.scenario is not None
                        else (
                            next(iter(self.parameters), None)
                            if isinstance(self.parameters, dict) and self.parameters
                            else None
                        )
                    )
                )
            )

            if scenario_idx is None:
                scenario_idx = (
                    self._last_eval_scenario_idx
                    if getattr(self, "_last_eval_scenario_idx", None) is not None
                    else 0
                )

            # Also extend scenario_cfs so reporting includes new rows
            if self.scenario_cfs is None:
                self.scenario_cfs = []

            for cf in new_cf_entries:
                val = self._evaluate_cf_value_for_redo(
                    cf, scenario_idx=scenario_idx, scenario_name=scenario_name
                )
                if val == 0:
                    continue
                for i, j in cf["positions"]:
                    cm[i, j] = val
                # Keep reporting structures in sync
                self.scenario_cfs.append(
                    {
                        "supplier": cf["supplier"],
                        "consumer": cf["consumer"],
                        "positions": sorted(cf["positions"]),
                        "value": val,
                    }
                )
            # Ensure efficient structure
            self.characterization_matrix = self.characterization_matrix.tocsr()

        # --- Diagnostics: ending nnz
        if isinstance(self.characterization_matrix, sparse.COO):
            end_nnz = len(self.characterization_matrix.data)
        else:
            end_nnz = self.characterization_matrix.nnz
        self.logger.info(f"Ending characterization_matrix nnz = {end_nnz}")

        # 5) Update processed/unprocessed tracking and optionally recompute score
        self._update_unprocessed_edges()

        # Remember last evaluation context (so redo_lcia can be called repeatedly without args)
        if scenario is not None:
            self._last_eval_scenario_name = scenario
        elif getattr(self, "_last_eval_scenario_name", None) is None:
            self._last_eval_scenario_name = self.scenario

        if scenario_idx is not None:
            self._last_eval_scenario_idx = scenario_idx
        elif getattr(self, "_last_eval_scenario_idx", None) is None:
            self._last_eval_scenario_idx = 0

        if recompute_score:
            self.lcia()

        # Save the CURRENT inventory edges as the baseline for the next redo
        if only_tech:
            self._last_edges_snapshot_tech = current_edges
        else:
            self._last_edges_snapshot_bio = current_edges

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
