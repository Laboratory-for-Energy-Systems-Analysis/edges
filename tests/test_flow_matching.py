import pytest
from collections import defaultdict
from edges.flow_matching import (
    matches_classifications,
    match_operator,
    normalize_classification_entries,
    build_cf_index,
    preprocess_flows,
    build_index,
    normalize_signature_data,
    make_hashable,
    compute_average_cf,
)


def test_matches_classifications_exact():
    cf_class = [("cpc", "123")]
    flow_class = [("cpc", "123")]
    assert matches_classifications(cf_class, flow_class)


def test_matches_classifications_partial():
    cf_class = [("cpc", "123")]
    flow_class = [("cpc", "123456")]
    assert matches_classifications(cf_class, flow_class)


def test_matches_classifications_no_match():
    cf_class = [("cpc", "789")]
    flow_class = [("cpc", "123456")]
    assert not matches_classifications(cf_class, flow_class)


def test_match_operator_equals():
    assert match_operator("foo", "foo", "equals")


def test_match_operator_startswith():
    assert match_operator("foobar", "foo", "startswith")
    assert not match_operator("barfoo", "foo", "startswith")


def test_match_operator_contains():
    assert match_operator("foobar", "oba", "contains")
    assert not match_operator("foobar", "baz", "contains")


def test_normalize_classification_entries():
    raw = [
        # Case 1: dict input
        {
            "supplier": {
                "location": "GLO",
                "matrix": "technosphere",
                "classifications": {"CPC": ["01", "02"], "ISIC": ["A"]},
            },
            "consumer": {"matrix": "technosphere"},
            "value": 5,
        },
        # Case 2: tuple input (legacy)
        {
            "supplier": {
                "location": "GLO",
                "matrix": "technosphere",
                "classifications": (("CPC", ("03",)), ("ISIC", ("B",))),
            },
            "consumer": {"matrix": "technosphere"},
            "value": 8,
        },
        # Case 3: list of tuples (already normalized format)
        {
            "supplier": {
                "location": "GLO",
                "matrix": "technosphere",
                "classifications": [("CPC", "04"), ("ISIC", "C")],
            },
            "consumer": {"matrix": "technosphere"},
            "value": 10,
        },
    ]

    normalized = normalize_classification_entries(raw)

    expected_classifications = [
        # Case 1 normalized
        (("CPC", "01"), ("CPC", "02"), ("ISIC", "A")),
        # Case 2 normalized
        (("CPC", "03"), ("ISIC", "B")),
        # Case 3 normalized
        (("CPC", "04"), ("ISIC", "C")),
    ]

    for i, cf in enumerate(normalized):
        assert isinstance(cf["supplier"]["classifications"], tuple)
        assert cf["supplier"]["classifications"] == expected_classifications[i]


def test_build_cf_index():
    raw_cfs = [
        {"supplier": {"location": "GLO"}, "consumer": {}, "value": 5},
        {"supplier": {"location": "US"}, "consumer": {}, "value": 10},
    ]
    index = build_cf_index(raw_cfs)

    assert ("GLO", "__ANY__") in index
    assert ("US", "__ANY__") in index


def test_preprocess_flows():
    flows = [
        {"name": "foo", "location": "CH", "position": 0},
        {"name": "bar", "location": "DE", "position": 1},
    ]
    result = preprocess_flows(flows, mandatory_fields={"name", "location"})
    key1 = make_hashable({"name": "foo", "location": "CH"})
    key2 = make_hashable({"name": "bar", "location": "DE"})
    assert 0 in result[key1]
    assert 1 in result[key2]


def test_build_index():
    flows = defaultdict(list)
    flows[make_hashable({"name": "foo"})].append(1)
    flows[make_hashable({"name": "bar"})].append(2)

    index = build_index(flows, required_fields={"name"})

    assert "name" in index
    assert "foo" in index["name"]
    assert "bar" in index["name"]

    # Optionally check the values too
    assert index["name"]["foo"][0][1] == [1]
    assert index["name"]["bar"][0][1] == [2]


def test_normalize_signature_data():
    data = {
        "location": "CH",
        "classifications": {"cpc": ["123", "456"], "isic": ["789"]},
    }
    normalized = normalize_signature_data(
        data.copy(), required_fields={"location", "classifications"}
    )
    assert isinstance(normalized["classifications"], tuple)
    assert ("cpc", "123") in normalized["classifications"]
    assert ("cpc", "456") in normalized["classifications"]
    assert ("isic", "789") in normalized["classifications"]


def test_compute_average_cf_with_any_fallback():
    raw_cfs = [
        {
            "supplier": {"name": "Oil", "location": "GLO"},
            "consumer": {},  # No required fields
            "value": 10,
        }
    ]
    required_supplier_fields = {"name", "location"}
    required_consumer_fields = set()  # <- No required fields

    cf_index = build_cf_index(raw_cfs)

    supplier_info = {"name": "Oil", "location": "GLO"}
    consumer_info = {"location": "CH"}  # Any value — won't be used in matching
    candidate_consumers = ["__ANY__"]

    result, matched_cf, _ = compute_average_cf(
        candidate_suppliers=["GLO"],
        candidate_consumers=candidate_consumers,  # not empty!
        supplier_info=supplier_info,
        consumer_info=consumer_info,
        cf_index=cf_index,
        required_supplier_fields=required_supplier_fields,
        required_consumer_fields=required_consumer_fields,
    )

    assert matched_cf is not None, f"Expected a matched CF, got {matched_cf}"
    assert result == "(1.000 * (10))"
