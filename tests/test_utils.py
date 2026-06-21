import pytest
import edges.utils as utils
from edges.utils import (
    format_method_name,
    format_data,
    make_hashable,
    get_str,
    interpolate_indexed_value,
    safe_eval,
    safe_eval_cached,
    get_activities,
)

INTERPOLATION_POLICY = {
    "axis": "scenario_idx",
    "axis_type": "year",
    "method": "linear",
    "extrapolation": "nearest",
    "source_years": ["2019", "2024", "2029"],
}


def test_format_method_name():
    assert format_method_name("foo_bar") == ("foo", "bar")


def test_make_hashable():
    d = {"b": 2, "a": 1}
    h = make_hashable(d)
    assert isinstance(h, tuple)
    assert h == (("a", 1), ("b", 2))


def test_get_str():
    assert get_str(42) == "42"
    assert get_str(3.14) == "3.14"
    assert get_str("foo") == "foo"
    assert get_str(None) == "None"
    assert get_str(("baz", "qux")) == "qux"


def test_interpolate_indexed_value_exact_interpolated_and_clamped():
    values = {"2019": 10.0, "2024": 20.0, "2029": 50.0}

    assert interpolate_indexed_value(values, "2024") == pytest.approx(20.0)
    assert interpolate_indexed_value(
        values, "2026", interpolation_policy=INTERPOLATION_POLICY
    ) == pytest.approx(32.0)
    assert interpolate_indexed_value(
        values, "2010", interpolation_policy=INTERPOLATION_POLICY
    ) == pytest.approx(10.0)
    assert interpolate_indexed_value(
        values, "2050", interpolation_policy=INTERPOLATION_POLICY
    ) == pytest.approx(50.0)


def test_interpolate_indexed_value_handles_numeric_sequences():
    values = {"2019": [0.0, 10.0], "2029": [10.0, 30.0]}

    assert interpolate_indexed_value(
        values, "2024", interpolation_policy=INTERPOLATION_POLICY
    ) == pytest.approx([5.0, 20.0])


def test_interpolate_indexed_value_keeps_legacy_fallback_without_policy():
    values = {"2019": 10.0, "2024": 20.0, "2029": 50.0}

    assert interpolate_indexed_value(values, "2026") == pytest.approx(50.0)
    assert interpolate_indexed_value(
        values,
        "2026",
        interpolation_policy={"method": "linear", "extrapolation": "nearest"},
    ) == pytest.approx(50.0)


def test_interpolate_indexed_value_keeps_legacy_fallback_for_non_numeric_index():
    values = {"baseline": 1.0, "future": 2.0}

    assert interpolate_indexed_value(values, "unknown") == pytest.approx(2.0)


def test_safe_eval():
    params = {"x": 3}
    assert (
        safe_eval("2 + x", SAFE_GLOBALS={"__builtins__": None}, parameters=params) == 5
    )
    assert safe_eval("sqrt(4)", SAFE_GLOBALS=None, parameters={}) == 2
    assert safe_eval("sum([1, 2, 3])", SAFE_GLOBALS={"sum": sum}, parameters={}) == 6


def test_safe_eval_allows_user_defined_functions():
    def scale(value, factor):
        return value * factor

    assert (
        safe_eval(
            "scale(x, 4)",
            SAFE_GLOBALS={"scale": scale},
            parameters={"x": 3},
        )
        == 12
    )


@pytest.mark.parametrize(
    "expr",
    [
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('os')",
        "(1).to_bytes(1, 'big')",
        "[x * 2 for x in range(3)]",
        "(1, 2)[0]",
        "lambda x: x",
        "f'{1}'",
    ],
)
def test_safe_eval_rejects_unsafe_syntax(expr):
    with pytest.raises(ValueError, match="Invalid expression"):
        safe_eval(expr, SAFE_GLOBALS={"range": range}, parameters={})


def test_safe_eval_missing_parameter_still_raises_keyerror():
    with pytest.raises(KeyError):
        safe_eval("missing + 1", SAFE_GLOBALS={}, parameters={})


def test_safe_eval_cached():
    SAFE_GLOBALS = {"sum": sum}
    params = {}
    assert (
        safe_eval_cached(
            "10 + 5", SAFE_GLOBALS=SAFE_GLOBALS, parameters=params, scenario_idx="foo"
        )
        == 15
    )
    assert (
        safe_eval_cached(
            "sum([1, 2, 3])",
            SAFE_GLOBALS=SAFE_GLOBALS,
            scenario_idx="foo",
            parameters=params,
        )
        == 6
    )


def test_safe_eval_cached_includes_allowed_function_namespace():
    def offset_one(value):
        return value + 1

    def offset_two(value):
        return value + 2

    expr = "offset(x)"
    params = {"x": 1}

    assert (
        safe_eval_cached(
            expr,
            SAFE_GLOBALS={"offset": offset_one},
            scenario_idx="same",
            parameters=params,
        )
        == 2
    )
    assert (
        safe_eval_cached(
            expr,
            SAFE_GLOBALS={"offset": offset_two},
            scenario_idx="same",
            parameters=params,
        )
        == 3
    )


def test_format_data_minimal_schema_defaults():
    data = {
        "exchanges": [
            {
                "supplier": {"matrix": "biosphere", "name": "CO2"},
                "consumer": {"matrix": "technosphere"},
                "value": 1.0,
            }
        ]
    }
    formatted, metadata = format_data(data, weight=None)
    assert len(formatted) == 1
    assert metadata["name"] == "Custom LCIA method"
    assert metadata["version"] == "0.0"
    assert metadata["unit"] == "unspecified"
    assert metadata["weighting_metadata"]["effective_source"] is None


def test_format_data_unknown_weight_scheme_does_not_crash():
    data = {
        "name": "test",
        "version": "1",
        "unit": "kg",
        "exchanges": [
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "CO2",
                    "location": "CH",
                },
                "consumer": {
                    "matrix": "technosphere",
                    "location": "FR",
                },
                "value": 1.0,
            }
        ],
    }
    formatted, _ = format_data(data, weight="unknown-scheme")
    assert formatted[0].get("weight") == 0


def test_format_data_tracks_embedded_method_weights():
    data = {
        "name": "test",
        "exchanges": [
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "CO2",
                    "location": "CH",
                },
                "consumer": {
                    "matrix": "technosphere",
                    "location": "FR",
                },
                "value": 1.0,
                "weight": 123.0,
            }
        ],
    }

    _, metadata = format_data(data, weight="population")

    assert metadata["weighting_metadata"]["effective_source"] == "method"
    assert metadata["weighting_metadata"]["label"] == "embedded method weights"
    assert metadata["weighting_metadata"]["preweighted_rows"] == 1


class _FakeExpression:
    def __init__(self, operation, field=None, value=None, left=None, right=None):
        self.operation = operation
        self.field = field
        self.value = value
        self.left = left
        self.right = right

    def __and__(self, other):
        return _FakeExpression("and", left=self, right=other)


class _FakeField:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return _FakeExpression("eq", field=self.name, value=value)

    def in_(self, values):
        return _FakeExpression("in", field=self.name, value=tuple(values))


class _FakeActivity:
    def __init__(self, database="fake-db", code=None, id_=None):
        self.database = database
        self.code = code
        self.id = id_


def _find_expression(expr, operation, field):
    if expr is None:
        return None
    if expr.operation == operation and expr.field == field:
        return expr
    return _find_expression(expr.left, operation, field) or _find_expression(
        expr.right, operation, field
    )


def test_get_activities_chunks_tuple_keys(monkeypatch):
    executed_batches = []

    class FakeActivityDataset:
        database = _FakeField("database")
        code = _FakeField("code")
        id = _FakeField("id")
        location = _FakeField("location")
        name = _FakeField("name")
        product = _FakeField("product")
        type = _FakeField("type")

        @classmethod
        def select(cls):
            return FakeQuery()

    class FakeQuery:
        def __init__(self):
            self.conditions = []

        def where(self, condition):
            self.conditions.append(condition)
            return self

        def __iter__(self):
            main_condition = self.conditions[0]
            database = _find_expression(main_condition, "eq", "database").value
            codes = _find_expression(main_condition, "in", "code").value
            executed_batches.append(codes)
            return iter(_FakeActivity(database=database, code=code) for code in codes)

    monkeypatch.setattr(utils, "AD", FakeActivityDataset)
    monkeypatch.setattr(utils, "NODE_PROCESS_CLASS_MAPPING", None)
    monkeypatch.setattr(utils, "_get_sqlite_variable_limit", lambda: 23)

    keys = [("fake-db", f"act-{index}") for index in range(35)]

    nodes = get_activities(keys)

    assert len(nodes) == 35
    assert [len(batch) for batch in executed_batches] == [14, 14, 7]


def test_get_activities_chunks_integer_keys(monkeypatch):
    executed_batches = []

    class FakeActivityDataset:
        database = _FakeField("database")
        code = _FakeField("code")
        id = _FakeField("id")
        location = _FakeField("location")
        name = _FakeField("name")
        product = _FakeField("product")
        type = _FakeField("type")

        @classmethod
        def select(cls):
            return FakeQuery()

    class FakeQuery:
        def __init__(self):
            self.conditions = []

        def where(self, condition):
            self.conditions.append(condition)
            return self

        def __iter__(self):
            ids = _find_expression(self.conditions[0], "in", "id").value
            executed_batches.append(ids)
            return iter(_FakeActivity(id_=id_) for id_ in ids)

    monkeypatch.setattr(utils, "AD", FakeActivityDataset)
    monkeypatch.setattr(utils, "NODE_PROCESS_CLASS_MAPPING", None)
    monkeypatch.setattr(utils, "_get_sqlite_variable_limit", lambda: 23)

    nodes = get_activities(list(range(31)))

    assert len(nodes) == 31
    assert [len(batch) for batch in executed_batches] == [15, 15, 1]
