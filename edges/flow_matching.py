from collections import defaultdict
from functools import lru_cache
import numpy as np
from copy import deepcopy
import json, time
from typing import NamedTuple, List, Optional

from .utils import make_hashable, _short_cf, _head


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

        if supplier_match is False:
            continue

        consumer_match = match_flow(
            flow=filtered_consumer,
            criteria=consumer_cf,
        )

        if consumer_match is False:
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

    if best_cf:
        logger.debug("Best matching CF selected with score %d: %s", best_score, best_cf)
        results.append(best_cf)
    else:
        logger.debug(
            "No matching CF found for supplier %s and consumer %s.",
            filtered_supplier,
            filtered_consumer,
        )

    return results


def matches_classifications(cf_classifications, dataset_classifications):
    """Match CF classification codes to dataset classifications."""

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
    Build a nested CF index:
        cf_index[(supplier_loc, consumer_loc)] → list of CFs
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
    return match_with_index(
        flow_to_match,
        cached_match_with_index.index,
        cached_match_with_index.lookup_mapping,
        required_fields,
        cached_match_with_index.reversed_lookup,
    )


def preprocess_flows(flows_list: list, mandatory_fields: set) -> dict:
    """
    Preprocess flows into a lookup dictionary.
    Each flow is keyed by a tuple of selected metadata fields.
    If no fields are present, falls back to a single universal key ().
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
    matches: List[int]
    location_only_rejects: dict[int, str]


def match_with_index(
    flow_to_match: dict,
    index: dict,
    lookup_mapping: dict,
    required_fields: set,
    reversed_lookup: dict,
) -> MatchResult:
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
        for key in keys:
            for pos in lookup_mapping.get(key, []):
                raw = reversed_lookup[pos]
                flow = dict(raw) if isinstance(raw, tuple) else raw
                if flow and match_flow(flow, ft_for_matchflow):
                    out.append(pos)
        return out

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
        pre_location_keys = None
        for field in nonloc_fields:
            cand = field_candidates(field, flow_to_match.get(field), op)
            pre_location_keys = (
                cand if pre_location_keys is None else (pre_location_keys & cand)
            )
            if not pre_location_keys:
                return MatchResult(matches=[], location_only_rejects={})
    else:
        # no required fields at all
        pre_location_keys = set(lookup_mapping.keys())

    # apply location last
    candidate_keys = pre_location_keys
    if has_location_constraint:
        loc_cand = field_candidates("location", flow_to_match.get("location"), op)
        candidate_keys = pre_location_keys & loc_cand

    # noloc matches (for diagnosing location-only)
    ft_no_loc = dict(flow_to_match)
    ft_no_loc.pop("location", None)
    noloc_matches = gather_positions(pre_location_keys, ft_no_loc)

    # full matches
    full_matches = gather_positions(candidate_keys, flow_to_match)

    loc_only = (
        set(noloc_matches) - set(full_matches) if has_location_constraint else set()
    )

    return MatchResult(
        matches=full_matches,
        location_only_rejects={pos: "location" for pos in loc_only},
    )


def compute_cf_memoized_factory(
    cf_index, required_supplier_fields, required_consumer_fields, weights
):
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


@lru_cache(maxsize=None)
def resolve_candidate_locations(
    *,
    geo,
    location: str,
    weights: frozenset,
    containing: bool = False,
    exceptions: set = None,
    supplier: bool = True,
) -> list:
    """
    Resolve candidate consumer locations from a base location.

    Parameters:
    - geo: GeoResolver instance
    - location: base location string (e.g., "GLO", "CH")
    - weights: valid weight region codes
    - containing: if True, return regions containing the location;
                  if False, return regions contained by the location
    - exceptions: list of regions to exclude (used with GLO fallback)

    Returns:
    - list of valid candidate location codes
    """
    try:
        candidates = geo.resolve(
            location=location,
            containing=containing,
            exceptions=exceptions or [],
        )
    except KeyError:
        return []

    if supplier is True:
        available_locs = [loc[0] for loc in weights]
    else:
        available_locs = [loc[1] for loc in weights]
    return [loc for loc in candidates if loc in available_locs]


def group_edges_by_signature(
    edge_list, required_supplier_fields, required_consumer_fields
):
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

    return grouped


def compute_average_cf(
    candidate_suppliers: list,
    candidate_consumers: list,
    supplier_info: dict,
    consumer_info: dict,
    cf_index: dict,
    required_supplier_fields: set = None,
    required_consumer_fields: set = None,
) -> tuple[str | float, Optional[dict], Optional[dict]]:
    """
    Compute weighted CF and a canonical aggregated uncertainty for composite regions.
    Returns: (expr_or_value, matched_cf_obj|None, agg_uncertainty|None)
    """
    # Optional timing (only if DEBUG)
    _t0 = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None

    if not candidate_suppliers and not candidate_consumers:
        logger.warning(
            "CF-AVG: no candidate locations provided | supplier_cands=%s | consumer_cands=%s",
            candidate_suppliers,
            candidate_consumers,
        )
        return 0, None, None

    # -------- Gate 1: location-key presence in cf_index --------
    valid_location_pairs = [
        (s, c)
        for s in candidate_suppliers
        for c in candidate_consumers
        if cf_index.get((s, c))
    ]

    if not valid_location_pairs:
        if logger.isEnabledFor(logging.DEBUG):
            # show small sample of what keys do exist for quick diagnosis
            some_keys = _head(cf_index.keys(), 10)
            logger.debug(
                "CF-AVG: no (supplier,consumer) keys in cf_index for candidates "
                "| suppliers=%s | consumers=%s | sample_index_keys=%s",
                _head(candidate_suppliers),
                _head(candidate_consumers),
                some_keys,
            )
        return 0, None, None
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "CF-AVG: %d valid (s,c) keys found (showing up to 10): %s",
                len(valid_location_pairs),
                _head(valid_location_pairs, 10),
            )

    # Build field-filtered views (exclude location; added per-loop)
    filtered_supplier = {
        k: supplier_info[k]
        for k in (required_supplier_fields or ())
        if k in supplier_info and k != "location"
    }
    filtered_consumer = {
        k: consumer_info[k]
        for k in (required_consumer_fields or ())
        if k in consumer_info and k != "location"
    }

    # -------- Gate 2: field/operator/classification match --------
    matched = []
    total_candidates_seen = 0

    for s_loc, c_loc in valid_location_pairs:
        cands = cf_index.get((s_loc, c_loc)) or []
        total_candidates_seen += len(cands)

        filtered_supplier["location"] = s_loc
        filtered_consumer["location"] = c_loc

        got = process_cf_list(cands, filtered_supplier, filtered_consumer)
        if logger.isEnabledFor(logging.DEBUG) and got:
            logger.debug(
                "CF-AVG: matched %d/%d CFs @ (%s,%s); example=%s",
                len(got),
                len(cands),
                s_loc,
                c_loc,
                _short_cf(got[0]),
            )
        matched.extend(got)

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

    # Weights
    total_w = sum(cf.get("weight", 0.0) for cf in matched)
    if total_w == 0:
        logger.warning(
            "CF-AVG: weights all zero/missing → using equal shares | matched=%d | example=%s",
            len(matched),
            _short_cf(matched[0]) if matched else None,
        )
        matched_cfs = [(cf, 1.0 / len(matched)) for cf in matched]
    else:
        matched_cfs = [(cf, cf.get("weight", 0.0) / total_w) for cf in matched]

    # Safety check on weights; log before assert explodes
    share_sum = sum(s for _, s in matched_cfs)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "CF-AVG: matched=%d | sum_shares=%.6f | example=%s",
            len(matched_cfs),
            share_sum,
            _short_cf(matched_cfs[0][0]) if matched_cfs else None,
        )

    assert np.isclose(share_sum, 1.0), f"Total shares must equal 1. Got: {share_sum}"

    # Build deterministic expression (string)
    expressions = [f"({share:.3f} * ({cf['value']}))" for cf, share in matched_cfs]
    expr = " + ".join(expressions)

    # Single CF shortcut (pass-through uncertainty)
    if len(matched_cfs) == 1:
        single_cf = matched_cfs[0][0]
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

    # Multi-CF aggregated uncertainty
    def _cf_sign(cf_obj) -> int | None:
        neg = (cf_obj.get("uncertainty") or {}).get("negative", None)
        if neg in (0, 1):
            return -1 if neg == 1 else +1
        v = cf_obj.get("value")
        if isinstance(v, (int, float)):
            return -1 if v < 0 else (+1 if v > 0 else None)
        return None

    cf_signs = [s for (cf, _sh) in matched_cfs if (s := _cf_sign(cf)) is not None]
    agg_sign = (
        cf_signs[0] if (cf_signs and all(s == cf_signs[0] for s in cf_signs)) else None
    )

    child_values, child_weights = [], []
    for cf, share in matched_cfs:
        if share <= 0:
            continue
        if cf.get("uncertainty") is not None:
            u = deepcopy(cf["uncertainty"])
            u["negative"] = 0
            child_unc = u
        else:
            v = cf.get("value")
            if isinstance(v, (int, float)):
                child_unc = {
                    "distribution": "discrete_empirical",
                    "parameters": {"values": [abs(v)], "weights": [1.0]},
                    "negative": 0,
                }
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "CF-AVG: skip agg-unc (symbolic child without unc) | child=%s",
                        _short_cf(cf),
                    )
                return (expr, None, None)
        child_values.append(child_unc)
        child_weights.append(float(share))

    wsum = sum(child_weights) or 1.0
    child_weights = [w / wsum for w in child_weights]

    ordering = sorted(
        range(len(child_values)),
        key=lambda i: json.dumps(child_values[i], sort_keys=True),
    )
    child_values = [child_values[i] for i in ordering]
    child_weights = [child_weights[i] for i in ordering]

    filtered = [
        (v, w) for v, w in zip(child_values, child_weights) if w > 0 and v is not None
    ]
    if not filtered:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CF-AVG: filtered children empty after cleanup.")
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
            "CF-AVG: success | children=%d | expr_len=%d | agg_sign=%s | dt=%.3f ms",
            len(child_values),
            len(expr),
            agg_sign,
            (dt * 1000.0) if dt else -1.0,
        )

    return (expr, None, agg_uncertainty)
