import pytest
from edges.utils import (
    format_method_name,
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
    SAFE_GLOBALS = {"range": range}
    params = {}
    assert safe_eval("2 + 2", SAFE_GLOBALS=SAFE_GLOBALS, parameters=params) == 4
    assert safe_eval(
        "[x * 2 for x in range(3)]", SAFE_GLOBALS=SAFE_GLOBALS, parameters=params
    ) == [0, 2, 4]


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
