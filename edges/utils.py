"""
Utility functions for the LCIA methods implementation.
"""

import os
import logging
from typing import Any

import yaml
import numpy as np

from functools import reduce, cache
import operator
import hashlib
import json
import math

from bw2data import __version__ as bw2data_version

if isinstance(bw2data_version, str):
    bw2data_version = tuple(map(int, bw2data_version.split(".")))

if bw2data_version >= (4, 0, 0):
    from bw2data.backends import ActivityDataset as AD
    from bw2data.subclass_mapping import NODE_PROCESS_CLASS_MAPPING
else:
    from bw2data.backends.peewee import ActivityDataset as AD

    NODE_PROCESS_CLASS_MAPPING = None

from bw2data import databases
import numbers

from .filesystem_constants import DATA_DIR


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_eval_cache = {}


def format_method_name(name: str) -> tuple:
    """
    Format the name of the method.
    :param name: The name of the method.
    :return: A tuple with the name of the method.
    """
    return tuple(name.split("_"))


def get_available_methods() -> list:
    """
    Display the available impact assessment methods by reading
     file names under `data` directory
    that ends with ".json" extension.
    :return: A list of available impact assessment methods.
    """
    return sorted(
        [
            format_method_name(f.replace(".json", ""))
            for f in os.listdir(DATA_DIR)
            if f.endswith(".json")
        ]
    )


def check_presence_of_required_fields(data: list):
    """
    Check if the required fields are present in the data.
    :param data: The data to check.
    :return: True if the required fields are present, False otherwise.
    """

    assert len(data) > 0, "No data provided."

    for cf in data:
        assert all(
            x in cf for x in ["supplier", "consumer"]
        ), f"Missing supplier or consumer in {cf}."
        assert any(x in cf for x in ["value", "formula"])
        assert "matrix" in cf["supplier"], f"Missing matrix fields in {cf['supplier']}."
        assert "matrix" in cf["consumer"], f"Missing matrix fields in {cf['consumer']}."
        assert any(
            x.get("operator", "equals") in ["equals", "contains", "startswith"]
            for x in [cf["supplier"], cf["consumer"]]
        ), f"Invalid operator in {cf}."


def format_data(data: dict, weight: str) -> tuple[list, dict[Any, Any]]:
    """
    Format the data for the LCIA method.
    :param data: The data for the LCIA method.
    :param weight: The type of weight to include.
    :return: The formatted data for the LCIA method.
    """

    assert all(
        x in data for x in ("name", "version", "unit", "exchanges")
    ), "Missing required fields in data."

    # Extract and attach scenario-specific parameters if present
    scenario_parameters = data.get("parameters", {})

    for cf in data["exchanges"]:
        for category in ["supplier", "consumer"]:
            for field, value in cf.get(category, {}).items():
                if field == "categories":
                    cf[category][field] = tuple(value)

    check_presence_of_required_fields(data["exchanges"])

    formatted_exchanges = add_population_and_gdp_data(
        data=data["exchanges"], weight=weight
    )

    metadata = {k: v for k, v in data.items() if k != "exchanges"}
    if scenario_parameters:
        metadata["parameters"] = scenario_parameters

    return formatted_exchanges, metadata


def add_population_and_gdp_data(data: list, weight: str) -> list:
    """
    Add population and GDP data to the LCIA method.
    :param data: The data for the LCIA method.
    :param weight: the type of weight to include.
    :return: The data for the LCIA method with population and GDP data.
    """
    # load population data from data/population.yaml

    if weight == "population":
        path = DATA_DIR / "metadata" / "population.yaml"
        try:
            with open(path, "r", encoding="utf-8") as f:
                weighting_data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Population metadata file not found at %s", path)
            raise

    # load GDP data from data/gdp.yaml
    if weight == "gdp":
        path = DATA_DIR / "metadata" / "gdp.yaml"
        try:
            with open(path, "r", encoding="utf-8") as f:
                weighting_data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("GDP metadata file not found at %s", path)
            raise

    # add to the data dictionary
    missing = 0
    for cf in data:
        for category in ["consumer", "supplier"]:
            if "location" in cf[category]:
                if "weight" not in cf:
                    k = cf[category]["location"]
                    w = weighting_data.get(k, 0)
                    if not w:
                        missing += 1
                    cf["weight"] = w
    if missing:
        logger.warning(
            "Added weights with %d missing entries (defaulted to 0) for weight='%s'",
            missing,
            weight,
        )

    return data


def normalize_flow(flow):
    """
    Return a dictionary view of a flow object.

    For current bw2data (>= 4.0.0), flow is already dict‑like.
    For older versions, try to extract the underlying data from either:
      - flow._data (if available)
      - flow.data (if available)
    and return it as a dict.
    """
    # Current version: already dict‑like.
    if hasattr(flow, "get"):
        try:
            # Sometimes even if .get exists, the object might not be a pure dict.
            # Test if iterating over it works.
            iter(flow)
            return flow
        except TypeError:
            pass
    # Older version: check for _data attribute.
    if hasattr(flow, "_data"):
        data = flow._data
        if isinstance(data, dict):
            return data
        try:
            return dict(data)
        except Exception:
            pass
    # Sometimes the underlying document holds the data.
    if hasattr(flow, "data"):
        data = flow.data
        if isinstance(data, dict):
            return data
        try:
            return dict(data)
        except Exception:
            pass
    raise TypeError("Flow object does not support dict-like access.")


def get_flow_matrix_positions(mapping: dict) -> list:
    """
    Retrieve information about the flows in the given matrix.

    This function works for both current and anterior bw2data versions.
    It uses bw2data.get_activities() to batch query the flows, then builds
    a lookup using normalized flow data. For flows from older versions, the data
    is obtained from the _data attribute.

    :param mapping: A dict mapping flow identifiers (either (database, code) tuples
                    or integer IDs) to their positions.
    :return: A list of dictionaries with flow information and their positions.
    """
    # Batch retrieve flows using get_activities() (assumed available in bw2data)
    keys = list(mapping.keys())
    flows_objs = get_activities(keys)
    logger.debug("Resolved %d flow objects for %d keys", len(flows_objs), len(keys))

    # Build a lookup mapping both the numeric ID (if available) and (database, code)
    # tuple to the original flow object.
    lookup = {}
    for flow in flows_objs:
        data = normalize_flow(flow)
        if "id" in data:
            lookup[data["id"]] = flow
        if "database" in data and "code" in data:
            lookup[(data["database"], data["code"])] = flow

    result = []
    for k, pos in mapping.items():
        flow = lookup.get(k)
        if flow is None and isinstance(k, tuple) and len(k) == 2:
            # Fallback: try to find a match manually.
            for f in flows_objs:
                data = normalize_flow(f)
                if data.get("database") == k[0] and data.get("code") == k[1]:
                    flow = f
                    break
        if flow is None:
            logger.error("Flow with key %s not found in fetched objects", k)
            raise KeyError(f"Flow with key {k} not found.")
        data = normalize_flow(flow)
        result.append(
            {
                "name": data.get("name"),
                "reference product": data.get("reference product"),
                "categories": data.get("categories"),
                "unit": data.get("unit"),
                "location": data.get("location"),
                "classifications": data.get("classifications"),
                "type": data.get("type"),
                "position": pos,
            }
        )
    return result


def get_activities(keys, **kwargs):
    """
    Retrieve multiple activity objects in a single SQL query.

    Args:
        keys: An iterable of keys, each being either a tuple (database, code)
              or an integer (the activity id).
        **kwargs: Additional filtering criteria.

    Returns:
        A list of activity objects. For bw2data >= 4.0.0 they are wrapped via
        NODE_PROCESS_CLASS_MAPPING, and for earlier versions the raw objects are returned.
    """

    keys = list(keys)
    qs = AD.select()

    # If keys are tuples, group by database and use an IN clause on code.
    if all(isinstance(k, tuple) for k in keys):
        groups = {}
        for db, code in keys:
            groups.setdefault(db, []).append(code)
        conditions = []
        for db, codes in groups.items():
            conditions.append((AD.database == db) & (AD.code.in_(codes)))
        qs = qs.where(reduce(operator.or_, conditions))
    # If keys are integers, assume they are activity ids.
    elif all(isinstance(k, numbers.Integral) for k in keys):
        qs = qs.where(AD.id.in_(keys))
    else:
        raise TypeError(
            "All keys must be either tuples (database, code) or integers (ids)."
        )

    # Apply additional filtering from kwargs.
    field_mapping = {
        "id": AD.id,
        "code": AD.code,
        "database": AD.database,
        "location": AD.location,
        "name": AD.name,
        "product": AD.product,
        "type": AD.type,
    }
    for key, value in kwargs.items():
        if key in field_mapping:
            qs = qs.where(field_mapping[key] == value)

    nodes = []
    for obj in qs:
        if NODE_PROCESS_CLASS_MAPPING is not None:
            backend = databases[obj.database].get("backend", "sqlite")
            cls = NODE_PROCESS_CLASS_MAPPING.get(backend, lambda x: x)
            nodes.append(cls(obj))
        else:
            nodes.append(obj)

    if len(nodes) != len(keys):
        logger.error(
            "Requested %d activities but found %d. Keys (sample): %s",
            len(keys),
            len(nodes),
            keys[:5],
        )
        raise Exception("Not all requested activity objects were found.")

    return nodes


def load_missing_geographies():
    """
    Load missing geographies from the YAML file.
    """
    with open(
        DATA_DIR / "metadata" / "missing_geographies.yaml", "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


def get_str(loc):
    if isinstance(loc, tuple):
        return loc[1]
    return str(loc)


def safe_eval(expr, parameters, SAFE_GLOBALS=None, scenario_idx: int | str = 0):
    """
    Evaluate a mathematical expression safely.
    :param expr: The expression to evaluate.
    :param parameters: A dictionary of parameters to use in the evaluation.
    :param SAFE_GLOBALS: A dictionary of global variables to use in the evaluation.
    :param scenario_idx: The index of the scenario to use in the evaluation.
    :return: The result of the evaluation.
    """
    if isinstance(expr, (int, float)):
        return float(expr)  # directly return numeric values

    # If expr is a string, evaluate it
    eval_params = {
        k: (v[scenario_idx] if isinstance(v, (list, tuple, np.ndarray)) else v)
        for k, v in parameters.items()
    }

    try:
        return eval(expr, SAFE_GLOBALS, eval_params)
    except NameError as e:
        missing_param = str(e).split("'")[1]
        logger.error(f"Missing parameter '{missing_param}' in expression '{expr}'")
        raise KeyError(
            f"Missing parameter '{missing_param}' in parameters dictionary."
        ) from None
    except Exception as e:
        logger.error(f"Error evaluating '{expr}': {e}")
        raise ValueError(f"Invalid expression '{expr}': {e}")


def safe_eval_cached(
    expr: str, parameters: dict, scenario_idx: str | int, SAFE_GLOBALS: dict
):
    # Convert parameters into a hashable string key
    key = (
        expr,
        scenario_idx,
        json.dumps(parameters, sort_keys=True),  # string representation
    )
    cache_key = hashlib.md5(str(key).encode()).hexdigest()

    if cache_key in _eval_cache:
        return _eval_cache[cache_key]

    result = safe_eval(
        expr, parameters, SAFE_GLOBALS=SAFE_GLOBALS, scenario_idx=scenario_idx
    )
    _eval_cache[cache_key] = result
    return result


def validate_parameter_lengths(parameters):
    lengths = {
        len(v) for v in parameters.values() if isinstance(v, (list, tuple, np.ndarray))
    }

    if not lengths:
        return 1  # Single scenario if no arrays

    if len(lengths) > 1:
        raise ValueError(f"Inconsistent lengths in parameter arrays: {lengths}")

    return lengths.pop()


def make_hashable(value):
    def convert(v):
        if isinstance(v, list):
            return tuple(convert(i) for i in v)
        if isinstance(v, dict):
            return tuple(sorted((k, convert(val)) for k, val in v.items()))
        return v

    return convert(value)


def assert_no_nans_in_cf_list(cf_list: list[dict], file_source: str = "<input>"):
    for i, cf in enumerate(cf_list):
        for side in ("supplier", "consumer"):
            entry = cf.get(side, {})
            for k, v in entry.items():
                if isinstance(v, float) and math.isnan(v):
                    raise ValueError(
                        f"NaN detected in {side} field '{k}' of CF at index {i} "
                        f"in {file_source}: {entry}. This field must be removed or filled."
                    )


def _head(seq, n=8):
    try:
        seq = list(seq)
        return seq[:n] + (["…"] if len(seq) > n else [])
    except Exception:
        return seq


def _short_cf(cf: dict, maxlen=160):
    """Compact view of a CF for logs."""
    try:
        core = {
            "value": cf.get("value"),
            "weight": cf.get("weight"),
            "supplier_loc": cf.get("supplier", {}).get("location"),
            "consumer_loc": cf.get("consumer", {}).get("location"),
        }
        s = json.dumps(core, sort_keys=True)
        return (s[: maxlen - 1] + "…") if len(s) > maxlen else s
    except Exception:
        return str(cf)[:maxlen]
