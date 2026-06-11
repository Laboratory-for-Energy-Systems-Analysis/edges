"""
Utility functions for the LCIA methods implementation.
"""

import ast
import os
import logging
import sqlite3
from typing import Any

import yaml
import numpy as np

from functools import cache
import hashlib
import json
import math

from bw2data import __version__ as bw2data_version
from packaging.version import Version

if isinstance(bw2data_version, tuple):
    bw2data_version = ".".join(map(str, bw2data_version))

bw2data_version = Version(bw2data_version)

if bw2data_version >= Version("4.0.0"):
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

_DEFAULT_SQLITE_VARIABLE_LIMIT = 999
_SQL_VARIABLE_SAFETY_MARGIN = 8
_MAX_ACTIVITY_QUERY_BATCH_SIZE = 20_000


DEFAULT_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "max": max,
    "min": min,
    "round": round,
    "pow": pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log10": math.log10,
}

_SAFE_EVAL_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_SAFE_EVAL_UNARYOPS = (ast.UAdd, ast.USub)


def _prepare_safe_globals(SAFE_GLOBALS: dict | None = None) -> dict:
    safe_globals = dict(DEFAULT_SAFE_GLOBALS)
    if SAFE_GLOBALS:
        for key, value in SAFE_GLOBALS.items():
            if key == "__builtins__":
                continue
            safe_globals[key] = value
    safe_globals["__builtins__"] = {}
    return safe_globals


def _safe_global_cache_token(value):
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", getattr(value, "__name__", repr(value))),
            id(value),
        )
    return ("value", type(value).__module__, type(value).__qualname__, repr(value))


def _safe_globals_cache_key(safe_globals: dict) -> tuple:
    return tuple(
        sorted(
            (name, _safe_global_cache_token(value))
            for name, value in safe_globals.items()
            if name != "__builtins__"
        )
    )


class _SafeEvalValidator(ast.NodeVisitor):
    def __init__(self, allowed_call_names: set[str]):
        self.allowed_call_names = allowed_call_names

    def generic_visit(self, node):
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, str, bool)):
            return
        raise ValueError(f"Disallowed constant: {node.value!r}")

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Only variable reads are allowed")

    def visit_BinOp(self, node):
        if not isinstance(node.op, _SAFE_EVAL_BINOPS):
            raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, _SAFE_EVAL_UNARYOPS):
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_List(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Only list literals are allowed")
        for element in node.elts:
            self.visit(element)

    def visit_Tuple(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Only tuple literals are allowed")
        for element in node.elts:
            self.visit(element)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only bare allowlisted function calls are allowed")
        if node.func.id not in self.allowed_call_names:
            raise ValueError(f"Function '{node.func.id}' is not allowlisted")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Variadic keyword arguments are not allowed")
            self.visit(keyword.value)


def _validate_safe_eval_ast(tree: ast.AST, safe_globals: dict) -> None:
    allowed_call_names = {
        name
        for name, value in safe_globals.items()
        if name != "__builtins__" and callable(value)
    }
    _SafeEvalValidator(allowed_call_names=allowed_call_names).visit(tree)


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

    if not data:
        raise ValueError("No exchanges provided in LCIA method.")

    for cf in data:
        if not all(x in cf for x in ["supplier", "consumer"]):
            raise ValueError(f"Missing supplier or consumer in exchange: {cf}")
        if not any(x in cf for x in ["value", "formula"]):
            raise ValueError(f"Missing 'value' or 'formula' in exchange: {cf}")
        if "matrix" not in cf["supplier"]:
            raise ValueError(f"Missing supplier 'matrix' in exchange: {cf}")
        if "matrix" not in cf["consumer"]:
            raise ValueError(f"Missing consumer 'matrix' in exchange: {cf}")

        for side in ("supplier", "consumer"):
            op = cf[side].get("operator", "equals")
            if op not in {"equals", "contains", "startswith"}:
                raise ValueError(
                    f"Invalid operator '{op}' in {side} for exchange: {cf}"
                )


def format_data(data: dict, weight: str) -> tuple[list, dict[Any, Any]]:
    """
    Format the data for the LCIA method.
    :param data: The data for the LCIA method.
    :param weight: The type of weight to include.
    :return: The formatted data for the LCIA method.
    """

    if not isinstance(data, dict):
        raise TypeError("Method data must be a mapping/dictionary.")
    if "exchanges" not in data:
        raise ValueError("Method data must contain an 'exchanges' field.")
    if not isinstance(data["exchanges"], list):
        raise TypeError("'exchanges' must be a list.")

    # Extract and attach scenario-specific parameters if present
    scenario_parameters = data.get("parameters", {})
    total_exchanges = len(data["exchanges"])
    preweighted_rows = sum(1 for cf in data["exchanges"] if "weight" in cf)

    for cf in data["exchanges"]:
        for category in ["supplier", "consumer"]:
            for field, value in cf.get(category, {}).items():
                if field == "categories":
                    cf[category][field] = tuple(value)

    check_presence_of_required_fields(data["exchanges"])

    formatted_exchanges = add_population_and_gdp_data(
        data=data["exchanges"], weight=weight
    )
    postweighted_rows = sum(1 for cf in formatted_exchanges if "weight" in cf)

    if total_exchanges and preweighted_rows == total_exchanges:
        effective_source = "method"
        label = "embedded method weights"
    elif (
        preweighted_rows == 0 and postweighted_rows and weight in {"population", "gdp"}
    ):
        effective_source = weight
        label = f"{weight} metadata weights"
    elif preweighted_rows and postweighted_rows:
        effective_source = "mixed"
        if weight in {"population", "gdp"}:
            label = f"mixed method + {weight} metadata weights"
        else:
            label = "mixed weights"
    else:
        effective_source = None
        label = "no weights"

    metadata = {
        "name": data.get("name", "Custom LCIA method"),
        "version": data.get("version", "0.0"),
        "unit": data.get("unit", "unspecified"),
        "weighting_metadata": {
            "requested_scheme": weight,
            "effective_source": effective_source,
            "label": label,
            "preweighted_rows": preweighted_rows,
            "weighted_rows": postweighted_rows,
            "total_rows": total_exchanges,
        },
        **{
            k: v
            for k, v in data.items()
            if k not in {"name", "version", "unit", "exchanges"}
        },
    }
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
    weighting_data = {}

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
    elif weight not in {"population", "gdp", None}:
        logger.warning(
            "Unknown weight scheme '%s'. No metadata weights will be injected.", weight
        )

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


def _get_sqlite_variable_limit() -> int:
    """Return the active SQLite bound-variable limit for Brightway's activity DB."""
    try:
        database = AD._meta.database
        if hasattr(database, "connect"):
            database.connect(reuse_if_open=True)
        connection = database.connection()
        if hasattr(connection, "getlimit"):
            limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            if isinstance(limit, int) and limit > 0:
                return limit
    except Exception:
        logger.debug("Could not detect SQLite variable limit", exc_info=True)
    return _DEFAULT_SQLITE_VARIABLE_LIMIT


def _activity_query_batch_size(extra_variables: int = 0) -> int:
    limit = _get_sqlite_variable_limit()
    available = limit - int(extra_variables) - _SQL_VARIABLE_SAFETY_MARGIN
    return max(1, min(_MAX_ACTIVITY_QUERY_BATCH_SIZE, available))


def _iter_batches(values: list, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _activity_filter_fields() -> dict[str, Any]:
    return {
        "id": AD.id,
        "code": AD.code,
        "database": AD.database,
        "location": AD.location,
        "name": AD.name,
        "product": AD.product,
        "type": AD.type,
    }


def _apply_activity_filters(qs, filters: dict):
    field_mapping = _activity_filter_fields()
    for key, value in filters.items():
        if key in field_mapping:
            qs = qs.where(field_mapping[key] == value)
    return qs


def _wrap_activity_dataset(obj):
    if NODE_PROCESS_CLASS_MAPPING is not None:
        backend = databases[obj.database].get("backend", "sqlite")
        cls = NODE_PROCESS_CLASS_MAPPING.get(backend, lambda x: x)
        return cls(obj)
    return obj


def get_activities(keys, **kwargs):
    """
    Retrieve multiple activity objects using batched SQL queries.

    Args:
        keys: An iterable of keys, each being either a tuple (database, code)
              or an integer (the activity id).
        **kwargs: Additional filtering criteria.

    Returns:
        A list of activity objects. For bw2data >= 4.0.0 they are wrapped via
        NODE_PROCESS_CLASS_MAPPING, and for earlier versions the raw objects are returned.
    """

    keys = list(keys)
    if not keys:
        return []

    field_mapping = _activity_filter_fields()
    extra_filter_variables = sum(1 for key in kwargs if key in field_mapping)
    nodes = []

    # If keys are tuples, group by database and use an IN clause on code.
    if all(isinstance(k, tuple) for k in keys):
        groups = {}
        for db, code in keys:
            groups.setdefault(db, []).append(code)
        batch_size = _activity_query_batch_size(
            extra_variables=1 + extra_filter_variables
        )
        for db, codes in groups.items():
            for batch in _iter_batches(codes, batch_size):
                qs = AD.select().where((AD.database == db) & (AD.code.in_(batch)))
                qs = _apply_activity_filters(qs, kwargs)
                nodes.extend(_wrap_activity_dataset(obj) for obj in qs)
    # If keys are integers, assume they are activity ids.
    elif all(isinstance(k, numbers.Integral) for k in keys):
        batch_size = _activity_query_batch_size(
            extra_variables=extra_filter_variables
        )
        for batch in _iter_batches(keys, batch_size):
            qs = AD.select().where(AD.id.in_(batch))
            qs = _apply_activity_filters(qs, kwargs)
            nodes.extend(_wrap_activity_dataset(obj) for obj in qs)
    else:
        raise TypeError(
            "All keys must be either tuples (database, code) or integers (ids)."
        )

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


@cache
def load_legacy_geographies() -> dict[str, Any]:
    """
    Load legacy geography aliases and placeholders from the YAML file.
    """
    with open(
        DATA_DIR / "metadata" / "legacy_geographies.yaml", "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f) or {}


@cache
def load_builtin_topologies() -> dict[str, dict[str, list[str]]]:
    """
    Load bundled IAM and ecoinvent topology definitions.

    The returned mapping is keyed by a short namespace derived from each file
    name, e.g. ``remind`` for ``remind-topology.json``.
    """
    topologies_dir = DATA_DIR / "metadata" / "topologies"
    if not topologies_dir.exists():
        return {}

    topologies = {}
    for path in sorted(topologies_dir.glob("*.json")):
        namespace = path.stem
        if namespace.endswith("-topology"):
            namespace = namespace[: -len("-topology")]
        with open(path, "r", encoding="utf-8") as f:
            topologies[namespace] = json.load(f)
    return topologies


def get_str(loc):
    if isinstance(loc, tuple):
        return loc[1]
    return str(loc)


def safe_eval(expr, parameters, SAFE_GLOBALS=None, scenario_idx: int | str = 0):
    """
    Evaluate a narrow mathematical expression safely.
    :param expr: The expression to evaluate.
    :param parameters: A dictionary of parameters to use in the evaluation.
    :param SAFE_GLOBALS: A dictionary of explicitly allowed names. Callable entries
        can be called by bare name, e.g. ``GWP(...)``.
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

    safe_globals = _prepare_safe_globals(SAFE_GLOBALS)

    try:
        tree = ast.parse(expr, mode="eval")
        _validate_safe_eval_ast(tree, safe_globals)
        code = compile(tree, "<edges-safe-eval>", "eval")
        return eval(code, safe_globals, eval_params)
    except NameError as e:
        missing_param = getattr(e, "name", None) or str(e).split("'")[1]
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
    safe_globals = _prepare_safe_globals(SAFE_GLOBALS)

    # Convert parameters into a hashable string key
    key = (
        expr,
        scenario_idx,
        json.dumps(parameters, sort_keys=True),  # string representation
        _safe_globals_cache_key(safe_globals),
    )
    cache_key = hashlib.md5(str(key).encode()).hexdigest()

    if cache_key in _eval_cache:
        return _eval_cache[cache_key]

    result = safe_eval(
        expr, parameters, SAFE_GLOBALS=safe_globals, scenario_idx=scenario_idx
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
