import pytest
from edges.utils import (
    format_method_name,
    format_data,
    make_hashable,
    get_str,
    safe_eval,
    safe_eval_cached,
)


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


def test_safe_eval():
    params = {"x": 3}
    assert (
        safe_eval("2 + x", SAFE_GLOBALS={"__builtins__": None}, parameters=params)
        == 5
    )
    assert safe_eval("sqrt(4)", SAFE_GLOBALS=None, parameters={}) == 2
    assert (
        safe_eval("sum([1, 2, 3])", SAFE_GLOBALS={"sum": sum}, parameters={}) == 6
    )


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
