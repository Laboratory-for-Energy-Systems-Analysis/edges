from collections import defaultdict
from functools import lru_cache
import numpy as np
from copy import deepcopy
import json, time
from typing import NamedTuple, List, Optional

from edges.utils import make_hashable, _short_cf, _head


import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def preprocess_cfs(cf_list, by="consumer"):
    """
    Group CFs by location from either 'consumer', 'supplier', or both.

    :param cf_list: List of characterization factors (CFs)
    :param by: One of 'consumer', 'supplier', or 'both'
    :return: defaultdict of location -> list of CFs
    """
    assert by in {
        "consumer",
        "supplier",
        "both",
    }, "'by' must be 'consumer', 'supplier', or 'both'"

    lookup = defaultdict(list)

    for cf in cf_list:
        consumer_loc = cf.get("consumer", {}).get("location")
        supplier_loc = cf.get("supplier", {}).get("location")

        if by == "consumer":
            if consumer_loc:
                lookup[consumer_loc].append(cf)

        elif by == "supplier":
            if supplier_loc:
                lookup[supplier_loc].append(cf)

        elif by == "both":
            if consumer_loc:
                lookup[consumer_loc].append(cf)
            elif supplier_loc:
                lookup[supplier_loc].append(cf)

    return lookup


def process_cf_list(
    cf_list: list,
    filtered_supplier: dict,
    filtered_consumer: dict,
) -> list:
    """
    Select the best-matching CF from a candidate list given supplier/consumer filters.

    :param cf_list: List of candidate CF dictionaries.
    :param filtered_supplier: Supplier-side fields to match against.
    :param filtered_consumer: Consumer-side fields to match against.
    :return: List with the single best CF (or empty if none matched).
    """
    results = []
    best_score = -1
    best_cf = None

    for cf in cf_list:
        supplier_cf = cf.get("supplier", {})
        consumer_cf = cf.get("consumer", {})

        supplier_match = match_flow(
            flow=filtered_supplier,
            criteria=supplier_cf,
        )

        if not supplier_match:
            continue

        consumer_match = match_flow(
            flow=filtered_consumer,
            criteria=consumer_cf,
        )

        if not consumer_match:
            continue

        match_score = 0
        cf_class = supplier_cf.get("classifications")
        ds_class = filtered_supplier.get("classifications")
        if cf_class and ds_class and matches_classifications(cf_class, ds_class):
            match_score += 1

        cf_cons_class = consumer_cf.get("classifications")
        ds_cons_class = filtered_consumer.get("classifications")
        if (
            cf_cons_class
            and ds_cons_class
            and matches_classifications(cf_cons_class, ds_cons_class)
        ):
            match_score += 1

        if match_score > best_score:
            best_score = match_score
            best_cf = cf
            if best_score == 2:
                break

    if best_cf:
        results.append(best_cf)
    else:
        logger.debug(
            "No matching CF found for supplier %s and consumer %s.",
            filtered_supplier,
            filtered_consumer,
        )

    return results


def matches_classifications(cf_classifications, dataset_classifications):
    """
    Check if CF classification codes match dataset classifications (prefix logic).

    :param cf_classifications: CF-side classifications (dict or list/tuple).
    :param dataset_classifications: Dataset classifications as list/tuple pairs.
    :return: True if at least one scheme/code pair matches by prefix, else False.
    """

    if isinstance(cf_classifications, dict):
        cf_classifications = [
            (scheme, code)
            for scheme, codes in cf_classifications.items()
            for code in codes
        ]
    elif isinstance(cf_classifications, (list, tuple)):
        if all(
            isinstance(x, tuple) and isinstance(x[1], (list, tuple))
            for x in cf_classifications
        ):
            # Convert from tuple of tuples like (('cpc', ('01.1',)),) -> [('cpc', '01.1')]
            cf_classifications = [
                (scheme, code) for scheme, codes in cf_classifications for code in codes
            ]

    dataset_codes = [
        (scheme, str(c).split(":")[0].strip())
        for scheme, codes in dataset_classifications
        for c in (codes if isinstance(codes, (list, tuple, set)) else [codes])
    ]

    for scheme, code in dataset_codes:
        if any(
            code.startswith(cf_code)
            and scheme.lower().strip() == cf_scheme.lower().strip()
            for cf_scheme, cf_code in cf_classifications
        ):
            return True
    return False


def match_flow(flow: dict, criteria: dict) -> bool:
    """
    Match a flow dictionary against criteria with operator and exclude support.

    :param flow: Flow metadata to test.
    :param criteria: Matching criteria (fields, operator, excludes, classifications).
    :return: True if all non-special fields match, else False.
    """

    operator = criteria.get("operator", "equals")
    excludes = criteria.get("excludes", [])

    # Handle excludes
    if excludes:
        for val in flow.values():
            if isinstance(val, str) and any(
                term.lower() in val.lower() for term in excludes
            ):
                return False
            elif isinstance(val, tuple):
                if any(
                    term.lower() in str(v).lower() for v in val for term in excludes
                ):
                    return False
            elif isinstance(val, list):
                if any(
                    term.lower() in str(v)
                    for v in val for term in excludes
                ):
                    return False

    # Handle standard field matching
    for key, target in criteria.items():
        if key in {
            "matrix",
            "operator",
            "weight",
            "position",
            "excludes",
            "classifications",
        }:
            continue

        value = flow.get(key)

        if target == "__ANY__":
            continue

        if value is None or not match_operator(value, target, operator):
            return False
    return True


@lru_cache(maxsize=None)
def match_operator(value: str, target: str, operator: str) -> bool:
    """
    Implements matching for three operator types:
      - "equals": value == target
      - "startswith": value starts with target (if both are strings)
      - "contains": target is contained in value (if both are strings)

    :param value: The flow's value.
    :param target: The lookup's candidate value.
    :param operator: The operator type ("equals", "startswith", "contains").
    :return: True if the condition is met, False otherwise.
    """
    if target == "__ANY__":
        return True

    if operator == "equals":
        return value == target
    elif operator == "startswith":
        if isinstance(value, str):
            return value.startswith(target)
        if isinstance(value, tuple):
            return value[0].startswith(target)
    elif operator == "contains":
        return target in value
    return False


def normalize_classification_entries(cf_list: list[dict]) -> list[dict]:
    """
    Normalize supplier-side 'classifications' to a flat tuple of (scheme, code).

    :param cf_list: List of CF dictionaries to normalize in-place.
    :return: The same list with normalized supplier classifications.
    """
    for cf in cf_list:
        supplier = cf.get("supplier", {})
        classifications = supplier.get("classifications")
        if isinstance(classifications, dict):
            # Normalize from dict
            supplier["classifications"] = tuple(
                (scheme, val)
                for scheme, values in sorted(classifications.items())
                for val in values
            )
        elif isinstance(classifications, list):
            # Already list of (scheme, code), just ensure it's a tuple
            supplier["classifications"] = tuple(classifications)
        elif isinstance(classifications, tuple):
            # Handle legacy format like: (('cpc', ('01.1',)),)
            new_classifications = []
            for scheme, maybe_codes in classifications:
                if isinstance(maybe_codes, (tuple, list)):
                    for code in maybe_codes:
                        new_classifications.append((scheme, code))
                else:
                    new_classifications.append((scheme, maybe_codes))
            supplier["classifications"] = tuple(new_classifications)
    return cf_list


def build_cf_index(raw_cfs: list[dict]) -> dict:
    """
    Build a CF index keyed by (supplier_location, consumer_location).

    :param raw_cfs: List of CF dictionaries.
    :return: Dict mapping (supplier_loc, consumer_loc) -> list of CFs.
    """
    index = defaultdict(list)

    for cf in raw_cfs:
        supplier_loc = cf.get("supplier", {}).get("location", "__ANY__")
        consumer_loc = cf.get("consumer", {}).get("location", "__ANY__")

        index[(supplier_loc, consumer_loc)].append(cf)

    return index


@lru_cache(maxsize=None)
def cached_match_with_index(flow_to_match_hashable, required_fields_tuple):
    flow_to_match = dict(flow_to_match_hashable)
    required_fields = set(required_fields_tuple)
    # the contexts live on the function as attributes
    return match_with_index(
        flow_to_match,
        cached_match_with_index.index,
        cached_match_with_index.lookup_mapping,
        required_fields,
        cached_match_with_index.reversed_lookup,
    )


def preprocess_flows(flows_list: list, mandatory_fields: set) -> dict:
    """
    Preprocess flows into a lookup dict keyed by selected metadata fields.

    :param flows_list: Iterable of flow dicts with at least a 'position' key.
    :param mandatory_fields: Set of fields to include in the key (may be empty).
    :return: Dict where key is a tuple of (field, value) and value is list of positions.
    """
    lookup = {}

    for flow in flows_list:

        def make_value_hashable(v):
            if isinstance(v, list):
                return tuple(v)
            if isinstance(v, dict):
                return tuple(
                    sorted((k, make_value_hashable(val)) for k, val in v.items())
                )
            return v

        if mandatory_fields:
            # Build a hashable key from mandatory fields
            key_elements = [
                (k, make_value_hashable(flow[k]))
                for k in mandatory_fields
                if k in flow and flow[k] is not None
            ]
            key = tuple(sorted(key_elements))
        else:
            # 🔁 NEW: universal key for empty criteria
            key = ()

        lookup.setdefault(key, []).append(flow["position"])

    return lookup


def build_index(lookup: dict, required_fields: set) -> dict:
    """
    Build an inverted index from the lookup dictionary.
    The index maps each required field to a dict, whose keys are the values
    from the lookup entries and whose values are lists of tuples:
    (lookup_key, positions), where lookup_key is the original key from lookup.

    :param lookup: The original lookup dictionary.
    :param required_fields: The fields to index.
    :return: A dictionary index.
    """
    index = {field: {} for field in required_fields}
    for key, positions in lookup.items():
        # Each key is assumed to be an iterable of (field, value) pairs.
        for k, v in key:
            if k in required_fields:
                index[k].setdefault(v, []).append((key, positions))
    return index


class MatchResult(NamedTuple):
    """Result container for indexed matching.

    :var matches: List of matched positions.
    :var location_only_rejects: Map of position -> reason ("location").
    """

    matches: List[int]
    location_only_rejects: dict[int, str]


def match_with_index(
    flow_to_match: dict,
    index: dict,
    lookup_mapping: dict,
    required_fields: set,
    reversed_lookup: dict,
) -> MatchResult:
    """
    Match a flow to positions using a per-field inverted index and full criteria.
    """
    SPECIAL = {"excludes", "operator", "matrix"}
    nonloc_fields = [f for f in required_fields if f not in SPECIAL and f != "location"]
    has_location_constraint = ("location" in required_fields) and (
        "location" in flow_to_match
    )
    op = flow_to_match.get("operator", "equals")

    allowed_keys = getattr(cached_match_with_index, "allowed_keys", None)

    def field_candidates(field, target, operator_value):
        field_index = index.get(field, {})
        out = set()
        if operator_value == "equals":
            if target == "__ANY__":
                for _, cand_list in field_index.items():
                    for key_only, _ in cand_list:
                        if (allowed_keys is None) or (key_only in allowed_keys):
                            out.add(key_only)
            else:
                for key_only, _ in field_index.get(target, []):
                    if (allowed_keys is None) or (key_only in allowed_keys):
                        out.add(key_only)
        else:
            # startswith / contains
            if target == "__ANY__":
                for _, cand_list in field_index.items():
                    for key_only, _ in cand_list:
                        if (allowed_keys is None) or (key_only in allowed_keys):
                            out.add(key_only)
            else:
                for candidate_value, cand_list in field_index.items():
                    if match_operator(
                        value=candidate_value, target=target, operator=operator_value
                    ):
                        for key_only, _ in cand_list:
                            if (allowed_keys is None) or (key_only in allowed_keys):
                                out.add(key_only)
        return out

    def gather_positions(keys, ft_for_matchflow):
        if not keys:
            return []
        out = []
        # Fast path: no excludes -> everything in these keys already matches
        excludes = ft_for_matchflow.get("excludes")
        if not excludes:
            for key in keys:
                # lookup_mapping[key] is the list of positions for this composite key
                bucket = lookup_mapping.get(key)
                if bucket:
                    out.extend(bucket)
            return out

        # Slow path: excludes present -> filter per-record once
        # Normalize excludes for faster checks
        ex = tuple(e.lower() for e in (excludes or ()))
        for key in keys:
            bucket = lookup_mapping.get(key)
            if not bucket:
                continue
            for pos in bucket:
                raw = reversed_lookup[pos]
                flow = dict(raw) if isinstance(raw, tuple) else raw
                # Only scan string fields; short-circuit early
                if any(
                    isinstance(v, str) and any(e in v.lower() for e in ex)
                    for v in flow.values()
                ):
                    continue
                out.append(pos)
        return out

    def intersect_smallest_first(sets_iterable):
        sets_list = [s for s in sets_iterable if s is not None]
        if not sets_list:
            return set()
        acc = min(sets_list, key=len).copy()
        for s in sorted((x for x in sets_list if x is not acc), key=len):
            acc &= s
            if not acc:
                break
        return acc

    # --- SPECIAL CASE: only 'location' is required ---
    if not nonloc_fields and has_location_constraint:
        all_keys = set(lookup_mapping.keys())

        # passes when ignoring location (still honors excludes/operator)
        ft_no_loc = dict(flow_to_match)
        ft_no_loc.pop("location", None)
        noloc_positions = gather_positions(all_keys, ft_no_loc)

        # full matches with location
        loc_keys = field_candidates("location", flow_to_match.get("location"), op)
        full_matches = gather_positions(loc_keys, flow_to_match)

        # everything that passed without location but failed with it
        loc_only = set(noloc_positions) - set(full_matches)

        return MatchResult(
            matches=full_matches,
            location_only_rejects={pos: "location" for pos in loc_only},
        )

    # --- NORMAL PATH: there are non-location required fields ---
    if nonloc_fields:
        # Build candidate key sets per non-location field
        per_field_sets = []
        for field in nonloc_fields:
            cand = field_candidates(field, flow_to_match.get(field), op)
            if not cand:
                # Any empty set means no matches possible
                return MatchResult(matches=[], location_only_rejects={})
            per_field_sets.append(cand)

        # Intersect smallest-first for speed
        pre_location_keys = intersect_smallest_first(per_field_sets)
        if not pre_location_keys:
            return MatchResult(matches=[], location_only_rejects={})
    else:
        # no required fields at all → start from all keys
        pre_location_keys = set(lookup_mapping.keys())

    # Apply location as an extra filter (kept separate to preserve location-only diagnostics)
    candidate_keys = pre_location_keys
    if has_location_constraint:
        loc_cand = field_candidates("location", flow_to_match.get("location"), op)
        # Intersect with location last (fast set op on already reduced key-space)
        candidate_keys = pre_location_keys & loc_cand

    # noloc matches (for diagnosing location-only)
    ft_no_loc = dict(flow_to_match)
    ft_no_loc.pop("location", None)
    noloc_matches = gather_positions(pre_location_keys, ft_no_loc)

    # full matches
    full_matches = gather_positions(candidate_keys, flow_to_match)

    loc_only = (
        (set(noloc_matches) - set(full_matches)) if has_location_constraint else set()
    )

    return MatchResult(
        matches=full_matches,
        location_only_rejects={pos: "location" for pos in loc_only},
    )


def compute_cf_memoized_factory(
    cf_index, required_supplier_fields, required_consumer_fields
):
    """
    Factory for a memoized compute_average_cf over signature/location candidates.

    :param cf_index: CF index keyed by (supplier_loc, consumer_loc).
    :param required_supplier_fields: Required fields for supplier signature.
    :param required_consumer_fields: Required fields for consumer signature.
    :return: Cached function(s_key, c_key, supplier_candidates, consumer_candidates) -> tuple.
    """

    @lru_cache(maxsize=None)
    def compute_cf(s_key, c_key, supplier_candidates, consumer_candidates):
        return compute_average_cf(
            candidate_suppliers=list(supplier_candidates),
            candidate_consumers=list(consumer_candidates),
            supplier_info=dict(s_key),
            consumer_info=dict(c_key),
            cf_index=cf_index,
            required_supplier_fields=required_supplier_fields,
            required_consumer_fields=required_consumer_fields,
        )

    return compute_cf


def normalize_signature_data(info_dict, required_fields):
    """
    Filter and normalize a dict to required fields for signature hashing.

    :param info_dict: Original supplier/consumer info dict.
    :param required_fields: Required field names to keep.
    :return: Filtered dict with normalized 'classifications' if present.
    """

    filtered = {k: info_dict[k] for k in required_fields if k in info_dict}

    # Normalize classifications
    if "classifications" in filtered:
        c = filtered["classifications"]
        if isinstance(c, dict):
            # From dict of lists -> tuple of (scheme, code)
            filtered["classifications"] = tuple(
                (scheme, code) for scheme, codes in c.items() for code in codes
            )
        elif isinstance(c, list):
            # Ensure it's a list of 2-tuples
            filtered["classifications"] = tuple(
                (scheme, code) for scheme, code in c if isinstance(scheme, str)
            )
        elif isinstance(c, tuple):
            # Possibly already normalized — validate structure
            if all(isinstance(e, tuple) and len(e) == 2 for e in c):
                filtered["classifications"] = c
            else:
                # Convert from legacy format
                new_classifications = []
                for scheme, maybe_codes in c:
                    if isinstance(maybe_codes, (tuple, list)):
                        for code in maybe_codes:
                            new_classifications.append((scheme, code))
                    else:
                        new_classifications.append((scheme, maybe_codes))
                filtered["classifications"] = tuple(new_classifications)

    return filtered


@lru_cache(maxsize=4096)
def _available_locs_from_weights(weights_key_tuple: tuple, supplier: bool) -> tuple:
    """
    Project available locations from a stable weights key.
    weights_key_tuple is a tuple of (supplier_loc, consumer_loc) pairs.
    Returns a sorted, de-duplicated tuple of allowed codes for the given side.
    """
    if supplier:
        vals = {w[0] for w in weights_key_tuple}
    else:
        vals = {w[1] for w in weights_key_tuple}
    # Keep deterministic order; don't special-case __ANY__ here
    return tuple(sorted(vals))


@lru_cache(maxsize=200_000)
def resolve_candidate_locations(
    *,
    geo,
    location: str,
    weights: tuple,
    containing: bool = False,
    exceptions: tuple | None = None,  # <— changed: tuple for caching
    supplier: bool = True,
) -> tuple:
    """
    Cached candidate resolver:
    - derives available locations once per weights_key_tuple + side
    - filters inside (including dropping 'GLO' when expanding GLO) to avoid extra list comps in hot loops
    - returns a tuple (hashable, deterministic)
    """
    try:
        exceptions = list(exceptions) if exceptions else []
        candidates = geo.resolve(
            location=location, containing=containing, exceptions=exceptions
        )
    except KeyError:
        return tuple()

    # When expanding GLO to its contained regions, drop 'GLO' itself here
    if containing and isinstance(location, str) and location == "GLO":
        candidates = [c for c in candidates if c != "GLO"]

    avail = _available_locs_from_weights(weights, supplier=supplier)

    # If wildcard is allowed on this side, we don't filter candidates by availability
    if "__ANY__" in avail:
        pool = candidates
    else:
        # avail is small; convert to set once for O(1) membership
        a = set(avail)
        pool = [loc for loc in candidates if loc in a]

    # Deterministic ordering across platforms
    # If you still want 'GLO' first (we dropped it above for GLO-expansion),
    # keep the same policy for non-GLO locations
    return tuple(sorted(set(pool)))


def group_edges_by_signature(
    edge_list, required_supplier_fields, required_consumer_fields
):
    """
    Group edges by (supplier signature, consumer signature, candidate locations).

    :param edge_list: Iterable of (s_idx, c_idx, s_info, c_info, s_cands, c_cands).
    :param required_supplier_fields: Supplier fields required for signature.
    :param required_consumer_fields: Consumer fields required for signature.
    :return: Dict[(s_key, c_key, (s_cands, c_cands))] -> list of (s_idx, c_idx).
    """
    grouped = defaultdict(list)

    for (
        supplier_idx,
        consumer_idx,
        supplier_info,
        consumer_info,
        supplier_candidate_locations,
        consumer_candidate_locations,
    ) in edge_list:
        s_filtered = normalize_signature_data(supplier_info, required_supplier_fields)
        c_filtered = normalize_signature_data(consumer_info, required_consumer_fields)

        s_key = make_hashable(s_filtered)
        c_key = make_hashable(c_filtered)

        loc_key = (
            tuple(make_hashable(c) for c in supplier_candidate_locations),
            tuple(make_hashable(c) for c in consumer_candidate_locations),
        )

        grouped[(s_key, c_key, loc_key)].append((supplier_idx, consumer_idx))

        for _k in grouped:
            grouped[_k].sort()

    return grouped


def compute_average_cf(
    candidate_suppliers: list | tuple,
    candidate_consumers: list | tuple,
    supplier_info: dict,
    consumer_info: dict,
    cf_index: dict,
    required_supplier_fields: set = None,
    required_consumer_fields: set = None,
) -> tuple[str | float, Optional[dict], Optional[dict]]:
    """
    Compute a weighted CF expression and aggregated uncertainty for composite regions.
    Deterministic across platforms without deep freezing: we sort by (s_loc, c_loc, cf_signature),
    where cf_signature is a compact, shallow tuple of stable fields.
    """
    _t0 = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None

    # ---- compact, shallow signatures (no deep recursion) ----
    # Keep only a few stable fields that define semantics; fall back to repr for odd types.
    def _cf_signature(cf: dict) -> tuple:
        # Pull once to locals (avoid many dict.get calls)
        # Choose a small set of fields that make equal CFs sort adjacent/stably
        v = cf.get("value")
        w = cf.get("weight")
        u = cf.get("unit")
        sym = cf.get("symbolic")  # expression or None
        # If there is an explicit identifier, prefer it for stability
        cfid = cf.get("id") or cf.get("code") or None
        # Normalize numerics; avoid touching nested dicts/lists
        try:
            v_norm = float(v) if isinstance(v, (int, float)) else repr(v)
        except Exception:
            v_norm = repr(v)
        try:
            w_norm = (
                float(w)
                if isinstance(w, (int, float))
                else (0.0 if w in (None, "", False) else 0.0)
            )
        except Exception:
            w_norm = 0.0
        return (cfid, v_norm, u or "", bool(sym))

    def _unc_signature(unc: dict | None) -> tuple:
        if not unc:
            return ("",)
        dist = unc.get("distribution", "")
        neg = unc.get("negative", None)
        # Shallow, order-stable snapshot of top-level parameters only
        params = unc.get("parameters")
        if isinstance(params, dict):
            # Only sort top-level keys; values kept as-is (repr) to avoid deep cost
            par_sig = tuple(sorted((k, repr(params[k])) for k in params.keys()))
        else:
            par_sig = repr(params)
        return (
            dist,
            1 if neg in (1, True) else 0 if neg in (0, False) else -1,
            par_sig,
        )

    # ---------- 1) Canonicalize candidate pools (once) ----------
    if not isinstance(candidate_suppliers, tuple):
        candidate_suppliers = tuple(set(candidate_suppliers))
    if not isinstance(candidate_consumers, tuple):
        candidate_consumers = tuple(set(candidate_consumers))

    if not candidate_suppliers and not candidate_consumers:
        logger.warning(
            "CF-AVG: no candidate locations provided | supplier_cands=%s | consumer_cands=%s",
            candidate_suppliers,
            candidate_consumers,
        )
        return 0, None, None

    S = candidate_suppliers
    C = candidate_consumers
    setS, setC = set(S), set(C)

    # ---------- 2) Efficient valid (s,c) pair discovery ----------
    idx_keys = cf_index.keys()
    prod_size = len(S) * len(C)
    if prod_size and prod_size <= len(idx_keys):
        valid_location_pairs = [(s, c) for s in S for c in C if (s, c) in cf_index]
        # S and C are already sorted; this is lexicographically ordered
    else:
        valid_location_pairs = [k for k in idx_keys if k[0] in setS and k[1] in setC]
        valid_location_pairs.sort()

    if not valid_location_pairs:
        if logger.isEnabledFor(logging.DEBUG):
            some_keys = _head(idx_keys, 10)
            logger.debug(
                "CF-AVG: no (supplier,consumer) keys in cf_index for candidates "
                "| suppliers=%s | consumers=%s | sample_index_keys=%s",
                _head(S),
                _head(C),
                some_keys,
            )
        return 0, None, None

    # ---------- 3) Base, field-filtered views (exclude 'location' here) ----------
    required_supplier_fields = required_supplier_fields or set()
    required_consumer_fields = required_consumer_fields or set()

    base_supplier = {
        k: supplier_info[k]
        for k in required_supplier_fields
        if k in supplier_info and k != "location"
    }
    base_consumer = {
        k: consumer_info[k]
        for k in required_consumer_fields
        if k in consumer_info and k != "location"
    }

    # ---------- 4) Field/operator/classification match ----------
    matched: list[tuple[str, str, dict]] = []
    total_candidates_seen = 0

    for s_loc, c_loc in valid_location_pairs:
        cands = cf_index.get((s_loc, c_loc)) or []
        total_candidates_seen += len(cands)

        fs = {**base_supplier, "location": s_loc}
        fc = {**base_consumer, "location": c_loc}

        got = process_cf_list(cands, fs, fc)
        if got and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CF-AVG: matched %d/%d CFs @ (%s,%s); example=%s",
                len(got),
                len(cands),
                s_loc,
                c_loc,
                _short_cf(got[0]),
            )
        for cf in got:
            matched.append((s_loc, c_loc, cf))

    if not matched:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CF-AVG: 0 CFs matched after field/classification checks "
                "| supplier_info=%s | consumer_info=%s | pairs=%s | total_candidates_seen=%d",
                supplier_info,
                consumer_info,
                _head(valid_location_pairs, 10),
                total_candidates_seen,
            )
        return 0, None, None

    # ---------- 5) Deterministic ordering without deep freezing ----------
    matched.sort(key=lambda t: (t[0], t[1], _cf_signature(t[2])))

    # ---------- 6) Build and normalize weights ----------
    # Pull weights once; avoid repeated cf.get in loops
    weights = []
    for _s, _c, cf in matched:
        w = cf.get("weight", 0.0)
        try:
            w = float(w)
        except Exception:
            w = 0.0
        if not np.isfinite(w) or w < 0.0:
            w = 0.0
        weights.append(w)

    w_arr = np.asarray(weights, dtype=np.float64)
    w_sum = float(w_arr.sum(dtype=np.float64))
    n_m = len(matched)

    if w_sum <= 0.0:
        shares = np.full(n_m, 1.0 / n_m, dtype=np.float64)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CF-AVG: weights all zero/missing → using equal shares | matched=%d | example=%s",
                n_m,
                _short_cf(matched[0][2]) if matched else None,
            )
    else:
        shares = w_arr / w_sum
        # prune tiny contributions to stabilize representation
        shares = np.where(shares < 1e-4, 0.0, shares)
        ssum = float(shares.sum(dtype=np.float64))
        shares = (
            (shares / ssum) if ssum > 0.0 else np.full(n_m, 1.0 / n_m, dtype=np.float64)
        )

    # ---------- 7) Expression assembly (uses matched order) ----------
    # Use shallow value access (no deep repr/formatting)
    expressions = []
    for (_s, _c, cf), sh in zip(matched, shares):
        if sh > 0.0:
            expressions.append(f"({sh:.4f} * ({cf.get('value')}))")
    expr = " + ".join(expressions)

    # ---------- 8) Single CF shortcut ----------
    if len(matched) == 1:
        single_cf = matched[0][2]
        agg_uncertainty = single_cf.get("uncertainty")
        if logger.isEnabledFor(logging.DEBUG):
            dt = (time.perf_counter() - _t0) if _t0 else None
            logger.debug(
                "CF-AVG: single CF path | expr=%s | has_unc=%s | dt=%.3f ms",
                expr,
                bool(agg_uncertainty),
                (dt * 1000.0) if dt else -1.0,
            )
        return (expr, single_cf, agg_uncertainty)

    # ---------- 9) Aggregate uncertainty (deterministic, shallow) ----------
    def _cf_sign(cf_obj) -> int | None:
        unc = cf_obj.get("uncertainty")
        neg = None if unc is None else unc.get("negative", None)
        if neg in (0, 1):
            return -1 if neg == 1 else +1
        v = cf_obj.get("value")
        if isinstance(v, (int, float)):
            return -1 if v < 0 else (+1 if v > 0 else None)
        return None

    cf_signs = [_cf_sign(cf) for (_s, _c, cf) in matched]
    cf_signs = [s for s in cf_signs if s is not None]
    agg_sign = (
        cf_signs[0] if (cf_signs and all(s == cf_signs[0] for s in cf_signs)) else None
    )

    child_values, child_weights = [], []
    for (_s, _c, cf), sh in zip(matched, shares):
        if sh <= 0.0:
            continue
        unc = cf.get("uncertainty")
        if unc is not None:
            # Shallow copy of top-level only (keeps nested as-is)
            child_unc = {
                k: (dict(v) if isinstance(v, dict) else v) for k, v in unc.items()
            }
            child_unc["negative"] = 0
        else:
            v = cf.get("value")
            if isinstance(v, (int, float)):
                child_unc = {
                    "distribution": "discrete_empirical",
                    "parameters": {"values": [abs(float(v))], "weights": [1.0]},
                    "negative": 0,
                }
            else:
                # symbolic without uncertainty: cannot aggregate deterministically
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "CF-AVG: skip agg-unc (symbolic child without unc) | child=%s",
                        _short_cf(cf),
                    )
                return expr, None, None
        child_values.append(child_unc)
        child_weights.append(float(sh))

    if not child_values:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CF-AVG: filtered children empty after cleanup.")
        return 0, None, None

    w = np.asarray(child_weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    wsum = float(w.sum(dtype=np.float64))
    w = (w / wsum) if wsum > 0.0 else np.full_like(w, 1.0 / len(w), dtype=np.float64)

    # Deterministic order of child uncertainties via shallow signature only
    order = sorted(
        range(len(child_values)), key=lambda i: _unc_signature(child_values[i])
    )
    child_values = [child_values[i] for i in order]
    child_weights = [float(w[i]) for i in order]

    # Final cleanup
    filtered = [
        (v, wt)
        for v, wt in zip(child_values, child_weights)
        if wt > 0.0 and v is not None
    ]
    if not filtered:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CF-AVG: filtered children empty after cleanup (post-sort).")
        return 0, None, None

    child_values, child_weights = zip(*filtered)

    agg_uncertainty = {
        "distribution": "discrete_empirical",
        "parameters": {"values": list(child_values), "weights": list(child_weights)},
    }
    if agg_sign is not None:
        agg_uncertainty["negative"] = 1 if agg_sign == -1 else 0

    if logger.isEnabledFor(logging.DEBUG):
        dt = (time.perf_counter() - _t0) if _t0 else None
        logger.debug(
            "CF-AVG: success | children=%d | expr_len=%d | agg_sign=%s | dt=%.3f ms | expr=%s",
            len(child_values),
            len(expr),
            agg_sign,
            (dt * 1000.0) if dt else -1.0,
            expr,
        )

    return expr, None, agg_uncertainty
