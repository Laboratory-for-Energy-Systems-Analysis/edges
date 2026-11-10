"""
Tests for the opt-in ``"domestic_pairs": "zero"`` method key.

GeoPolRisk-style paired methods have no row for supplier == consumer, so domestic
technosphere exchanges used to fall through to ``map_remaining_locations_to_global``
and receive a weighted global-average CF. With ``"domestic_pairs": "zero"`` an
exchange whose supplier is the consumer country (or lies inside it) gets an explicit
CF of 0 instead. Region-to-region pairs and cross-border pairs are untouched.
"""

import copy

import bw2data
import pytest
from bw2data import __version__
from packaging.version import Version

from edges import EdgeLCIA
from edges.utils import format_data

if isinstance(__version__, tuple):
    __version__ = ".".join(map(str, __version__))
PROJECT = (
    "EdgeLCIA-Test"
    if Version(str(__version__)) < Version("4.0.0")
    else "EdgeLCIA-Test-bw25"
)
DB_NAME = "domestic-pairs-test-db"

STRATEGIES = [
    "map_exchanges",
    "map_aggregate_locations",
    "map_dynamic_locations",
    "map_contained_locations",
    "map_remaining_locations_to_global",
]

# (code, location, reference product, technosphere inputs)
ACTIVITIES = [
    ("NG_US", "US", "natural gas", ()),
    ("NG_USTX", "US-TX", "natural gas", ()),
    ("NG_JP", "JP", "natural gas", ()),
    ("NG_RER", "RER", "natural gas", ()),
    ("EL_US", "US", "electricity", ("NG_US", "NG_USTX")),
    ("EL_JP", "JP", "electricity", ("NG_JP",)),
    ("EL_BD", "BD", "electricity", ("NG_JP",)),
    ("EL_RER", "RER", "electricity", ("NG_RER",)),
    # Country and IAM-region namesakes (ME/NAM/CHN are also ISO2 or ISO3 codes).
    ("NG_CN", "CN", "natural gas", ()),
    ("NG_CHN", "CHN", "natural gas", ()),
    ("NG_ME", "ME", "natural gas", ()),
    ("NG_NAM", "NAM", "natural gas", ()),
    ("NG_ANZ", "Australia_NZ", "natural gas", ()),
    ("EL_CN", "CN", "electricity", ("NG_CN",)),
    ("EL_CN_FROM_CHN", "CN", "electricity", ("NG_CHN",)),
    ("EL_CHN", "CHN", "electricity", ("NG_CHN",)),
    ("EL_ME", "ME", "electricity", ("NG_ME",)),
    ("EL_NAM", "NAM", "electricity", ("NG_NAM",)),
    ("EL_ANZ", "Australia_NZ", "electricity", ("NG_ANZ",)),
    ("EL_GLO", "GLO", "electricity", ("NG_US",)),
    ("NG_BR", "BR", "natural gas", ()),
    ("EL_US_MIXED", "US", "electricity", ("NG_USTX", "NG_BR")),
    ("EL_US_MIXED_REV", "US", "electricity", ("NG_BR", "NG_USTX")),
]

# What the global fallback assigns today to a pair with no row: the weighted mean
# of the two rows it selects for this fixture (DE->FR 0.2 and QA->JP 0.003, equal
# weights), not of all five rows of the method below.
GLOBAL_AVERAGE = 0.5 * 0.2 + 0.5 * 0.003


@pytest.fixture(scope="module", autouse=True)
def domestic_db():
    bw2data.projects.set_current(PROJECT)
    if DB_NAME in bw2data.databases:
        del bw2data.databases[DB_NAME]
    data = {}
    for code, loc, product, inputs in ACTIVITIES:
        exchanges = [
            {"amount": 1, "type": "production", "input": (DB_NAME, code), "unit": "kg"}
        ] + [
            {"amount": 1, "type": "technosphere", "input": (DB_NAME, i), "unit": "kg"}
            for i in inputs
        ]
        data[(DB_NAME, code)] = {
            "name": code,
            "unit": "kg",
            "location": loc,
            "reference product": product,
            "exchanges": exchanges,
        }
    bw2data.Database(DB_NAME).write(data)
    yield
    del bw2data.databases[DB_NAME]


def _row(supplier_loc, consumer_loc, value):
    return {
        "supplier": {
            "reference product": "natural gas",
            "location": supplier_loc,
            "matrix": "technosphere",
        },
        "consumer": {"location": consumer_loc, "matrix": "technosphere"},
        "value": value,
        "weight": 1.0,
    }


def _method(**extra):
    method = {
        "name": "paired test",
        "version": "1.0",
        "unit": "u",
        "strategies": list(STRATEGIES),
        "exchanges": [
            _row("CA", "US", 0.1),
            _row("MX", "US", 0.01),
            _row("AU", "JP", 0.03),
            _row("QA", "JP", 0.003),
            _row("DE", "FR", 0.2),
        ],
    }
    method.update(extra)
    return method


def _cfs(consumer_code, method):
    """Return {(supplier loc, consumer loc): numeric CF} for one demand."""
    act = bw2data.get_activity((DB_NAME, consumer_code))
    lca = EdgeLCIA(demand={act: 1}, method=copy.deepcopy(method))
    lca.lci()
    lca.apply_strategies()
    lca.evaluate_cfs()
    out = {}
    for entry in lca.cfs_mapping:
        key = (entry["supplier"]["location"], entry["consumer"]["location"])
        for pos in entry["positions"]:
            out[key] = float(lca.characterization_matrix[pos])
    return out


def test_domestic_us_to_us_gets_zero_with_flag():
    cfs = _cfs("EL_US", _method(domestic_pairs="zero"))
    assert cfs[("US", "US")] == 0.0


def test_subnational_us_tx_to_us_gets_zero_with_flag():
    cfs = _cfs("EL_US", _method(domestic_pairs="zero"))
    assert cfs[("US-TX", "US")] == 0.0


def test_domestic_jp_to_jp_gets_zero_with_flag():
    cfs = _cfs("EL_JP", _method(domestic_pairs="zero"))
    assert cfs[("JP", "JP")] == 0.0


def test_region_to_region_pair_still_averaged_with_flag():
    cfs = _cfs("EL_RER", _method(domestic_pairs="zero"))
    assert cfs[("RER", "RER")] == pytest.approx(GLOBAL_AVERAGE)


def test_cross_border_missing_pair_still_averaged_with_flag():
    cfs = _cfs("EL_BD", _method(domestic_pairs="zero"))
    assert cfs[("JP", "BD")] == pytest.approx(GLOBAL_AVERAGE)


def test_without_flag_domestic_pair_keeps_global_average():
    cfs = _cfs("EL_US", _method())
    assert cfs[("US", "US")] == pytest.approx(GLOBAL_AVERAGE)
    assert cfs[("US-TX", "US")] == pytest.approx(GLOBAL_AVERAGE)


def test_explicit_domestic_row_beats_the_flag():
    method = _method(domestic_pairs="zero")
    method["exchanges"].append(_row("US", "US", 0.5))
    cfs = _cfs("EL_US", method)
    assert cfs[("US", "US")] == pytest.approx(0.5)
    assert cfs[("US-TX", "US")] == 0.0


def test_invalid_flag_value_is_rejected():
    with pytest.raises(ValueError, match="domestic_pairs"):
        format_data(_method(domestic_pairs="ignore"), weight=None)


def test_valid_flag_is_kept_in_method_metadata():
    _, metadata = format_data(_method(domestic_pairs="zero"), weight=None)
    assert metadata["domestic_pairs"] == "zero"


def test_domestic_cn_to_cn_gets_zero_with_flag():
    cfs = _cfs("EL_CN", _method(domestic_pairs="zero"))
    assert cfs[("CN", "CN")] == 0.0


def test_iam_region_namesake_of_iso2_code_keeps_average():
    # ME is Montenegro and also the IMAGE "Middle East" region: ambiguous.
    cfs = _cfs("EL_ME", _method(domestic_pairs="zero"))
    assert cfs[("ME", "ME")] == pytest.approx(GLOBAL_AVERAGE)


def test_iam_region_remapped_to_iso2_code_keeps_average():
    # NAM is a MESSAGE region, not Namibia (NA).
    cfs = _cfs("EL_NAM", _method(domestic_pairs="zero"))
    assert cfs[("NAM", "NAM")] == pytest.approx(GLOBAL_AVERAGE)


def test_iam_region_to_same_iam_region_keeps_average():
    cfs = _cfs("EL_CHN", _method(domestic_pairs="zero"))
    assert cfs[("CHN", "CHN")] == pytest.approx(GLOBAL_AVERAGE)


def test_iam_region_supplier_into_country_keeps_average():
    # CHN (IMAGE/MESSAGE region) is not the country CN.
    cfs = _cfs("EL_CN_FROM_CHN", _method(domestic_pairs="zero"))
    assert cfs[("CHN", "CN")] == pytest.approx(GLOBAL_AVERAGE)


def test_multi_country_region_with_country_looking_name_keeps_average():
    cfs = _cfs("EL_ANZ", _method(domestic_pairs="zero"))
    assert cfs[("Australia_NZ", "Australia_NZ")] == pytest.approx(GLOBAL_AVERAGE)


def test_glo_consumer_is_not_a_country_with_flag():
    flagged = _cfs("EL_GLO", _method(domestic_pairs="zero"))
    plain = _cfs("EL_GLO", _method())
    assert flagged[("US", "GLO")] == pytest.approx(plain[("US", "GLO")])
    assert flagged[("US", "GLO")] != 0.0


@pytest.mark.parametrize("consumer", ["EL_US_MIXED", "EL_US_MIXED_REV"])
def test_domestic_and_import_suppliers_into_one_consumer_are_decided_separately(
    consumer,
):
    cfs = _cfs(consumer, _method(domestic_pairs="zero"))
    assert cfs[("US-TX", "US")] == 0.0
    assert cfs[("BR", "US")] == pytest.approx(GLOBAL_AVERAGE)
