import logging
from collections import defaultdict
from functools import cache, lru_cache
from typing import Optional
import numpy as np

from .utils import make_hashable, get_shares


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
        results.append(best_cf)

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
        (scheme, code.split(":")[0].strip()) for scheme, code in dataset_classifications
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


@cache
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


def build_cf_index(raw_cfs: list[dict], required_supplier_fields: set) -> dict:
    """
    Build a nested CF index:
        cf_index[consumer_location][supplier_signature] → list of CFs
    """
    index = defaultdict(list)

    for cf in raw_cfs:
        supplier_loc = cf.get("supplier", {}).get("location", "__ANY__")
        consumer_loc = cf.get("consumer", {}).get("location", "__ANY__")

        index[(supplier_loc, consumer_loc)].append(cf)

    return index


@cache
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


def match_with_index(
    flow_to_match: dict,
    index: dict,
    lookup_mapping: dict,
    required_fields: set,
    reversed_lookup: dict,
) -> list:
    """
    Match a flow against the lookup using the inverted index.
    Supports "equals", "startswith", and "contains" operators.

    :param flow_to_match: The flow to match.
    :param index: The inverted index produced by build_index().
    :param lookup_mapping: The original lookup dictionary mapping keys to positions.
    :param required_fields: The required fields for matching.
    :return: A list of matching positions.
    """
    candidate_keys = None

    if not required_fields:
        # Match all flows if no fields to match
        return [pos for positions in lookup_mapping.values() for pos in positions]

    for field in required_fields:

        if field in ("excludes", "operator", "matrix"):
            continue

        match_target = flow_to_match.get(field)

        operator_value = flow_to_match.get("operator", "equals")
        field_index = index.get(field, {})
        field_candidates = set()

        if operator_value == "equals":
            # Fast direct lookup.
            if match_target == "__ANY__":
                for candidate_value, candidate_list in field_index.items():
                    for candidate in candidate_list:
                        candidate_key, _ = candidate
                        field_candidates.add(candidate_key)
            else:
                for candidate in field_index.get(match_target, []):
                    candidate_key, _ = candidate
                    field_candidates.add(candidate_key)
        else:
            # For "startswith" or "contains", we iterate over all candidate values.
            if match_target == "__ANY__":
                for candidate_value, candidate_list in field_index.items():
                    for candidate in candidate_list:
                        candidate_key, _ = candidate
                        field_candidates.add(candidate_key)
            else:
                for candidate_value, candidate_list in field_index.items():
                    if match_operator(
                        value=candidate_value,
                        target=match_target,
                        operator=operator_value,
                    ):
                        for candidate in candidate_list:
                            candidate_key, _ = candidate
                            field_candidates.add(candidate_key)

        # Initialize or intersect candidate sets.
        if candidate_keys is None:
            candidate_keys = field_candidates
        else:
            candidate_keys &= field_candidates

        # Early exit if no candidates remain.
        if not candidate_keys:
            return []

    # Gather positions from the original lookup mapping for all candidate keys.
    matches = []
    for key in candidate_keys:
        for pos in lookup_mapping[key]:
            raw = reversed_lookup[pos]
            flow = dict(raw) if isinstance(raw, tuple) else raw
            if flow and match_flow(flow, flow_to_match):
                matches.append(pos)

    if "classifications" in flow_to_match:
        cf_classifications_by_scheme = flow_to_match["classifications"]

        if isinstance(cf_classifications_by_scheme, tuple):
            cf_classifications_by_scheme = dict(cf_classifications_by_scheme)

        classified_matches = []

        for pos in matches:
            flow = reversed_lookup.get(pos)
            flow = dict(flow)
            if not flow:
                continue
            dataset_classifications = flow.get("classifications", [])

            if dataset_classifications:
                for scheme, cf_codes in cf_classifications_by_scheme.items():
                    relevant_codes = [
                        code.split(":")[0].strip()
                        for s, code in dataset_classifications
                        if s.lower() == scheme.lower()
                    ]
                    if any(
                        code.startswith(prefix)
                        for prefix in cf_codes
                        for code in relevant_codes
                    ):
                        classified_matches.append(pos)
                        break

        if classified_matches:
            return classified_matches

    return matches


def compute_cf_memoized_factory(
    cf_index, required_supplier_fields, required_consumer_fields, weights, logger
):
    @lru_cache(maxsize=None)
    def compute_cf(s_key, c_key, supplier_candidates, consumer_candidates):
        return compute_average_cf(
            candidate_suppliers=list(supplier_candidates),
            candidate_consumers=list(consumer_candidates),
            supplier_info=dict(s_key),
            consumer_info=dict(c_key),
            weight=weights,
            cf_index=cf_index,
            required_supplier_fields=required_supplier_fields,
            required_consumer_fields=required_consumer_fields,
            logger=logger,
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
    weight: dict,
    cf_index: dict,
    required_supplier_fields: set = None,
    required_consumer_fields: set = None,
    logger=None,
) -> tuple[str | float, Optional[dict]]:
    """
    Compute the weighted average characterization factor (CF) for a given supplier-consumer pair.
    Supports disaggregated regional matching on both supplier and consumer sides.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not candidate_suppliers and not candidate_consumers:
        return 0, None

    # calculate all permutations
    valid_location_pairs = [
        (s, c)
        for s in candidate_suppliers
        for c in candidate_consumers
        if cf_index.get((s, c)) or cf_index.get(("__ANY__", "__ANY__"))
    ]

    if len(valid_location_pairs) == 0:
        return 0, None

    filtered_supplier = {
        k: supplier_info[k]
        for k in required_supplier_fields
        if k in supplier_info and k != "location"
    }
    filtered_consumer = {
        k: consumer_info[k]
        for k in required_consumer_fields
        if k in consumer_info and k != "location"
    }

    matched_cfs = []

    for (
        candidate_supplier_location,
        candidate_consumer_location,
    ) in valid_location_pairs:
        candidate_cfs = cf_index.get(
            (candidate_supplier_location, candidate_consumer_location)
        )

        if not candidate_cfs:
            continue

        filtered_supplier["location"] = candidate_supplier_location
        filtered_consumer["location"] = candidate_consumer_location

        matched_cfs.extend(
            process_cf_list(
                candidate_cfs,
                filtered_supplier,
                filtered_consumer,
            )
        )

    if not matched_cfs:
        return 0, None

    # normalize weights into shares
    sum_weights = sum(c.get("weight") for c in matched_cfs)
    if sum_weights == 0:
        logger.warning(
            f"No valid weights found for supplier {supplier_info} and consumer {consumer_info}. "
            "Using equal shares."
        )
        matched_cfs = [(cf, (1.0 / len(matched_cfs))) for cf in matched_cfs]
    else:
        matched_cfs = [
            (cf, (cf.get("weight", 0) / sum_weights if sum_weights else 1.0))
            for cf in matched_cfs
        ]

    assert np.isclose(
        sum(share for _, share in matched_cfs), 1
    ), f"Total shares must equal 1. Got: {sum(share for _, share in matched_cfs)}"

    expressions = [f"({share:.3f} * ({cf['value']}))" for cf, share in matched_cfs]
    expr = " + ".join(expressions)

    return (expr, matched_cfs[0][0]) if len(matched_cfs) == 1 else (expr, None)
