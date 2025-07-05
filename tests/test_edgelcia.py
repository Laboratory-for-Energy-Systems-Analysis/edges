import pytest

import pytest
from edges.edgelcia import EdgeLCIA
import bw2data


@pytest.fixture
def lcia():
    """Fixture to create an EdgeLCIA instance for testing."""
    # Set up once
    if bw2data.__version__ < (4, 0, 0):
        print("Using Brightway2.")
        bw2data.projects.set_current("EdgeLCIA-Test")
    else:
        print("Using Brightway2.5.")
        bw2data.projects.set_current("EdgeLCIA-Test-bw25")

    # Create a test database and activities
    db = bw2data.Database("lcia-test-db")

    act = db.random()

    lcia = EdgeLCIA(demand={act: 1}, method=("AWARE 2.0", "Country", "all", "yearly"))
    lcia.logger = type("Logger", (), {"info": lambda *a, **kw: None})()
    return lcia


def test_excluded_subregions_with_decomposition(lcia):
    lcia.position_to_technosphere_flows_lookup = {
        0: {"name": "electricity", "reference product": "high voltage"},
    }
    lcia.technosphere_flows_lookup = {
        ("electricity", "high voltage"): ["FR", "IT", "RoW"]
    }

    decomposed = frozenset(
        {
            ("FR", ("FR-IDF", "FR-BRE")),
            ("IT", ("IT-NO", "IT-SI")),
        }
    )

    result = lcia._extract_excluded_subregions(0, decomposed)
    assert result == frozenset(["FR-IDF", "FR-BRE", "IT-NO", "IT-SI"])


def test_excluded_subregions_with_partial_decomposition(lcia):
    lcia.position_to_technosphere_flows_lookup = {
        1: {"name": "heat", "reference product": "central"},
    }
    lcia.technosphere_flows_lookup = {("heat", "central"): ["PL", "HU"]}

    decomposed = frozenset(
        {
            ("PL", ("PL-MZ",)),
            # HU not decomposed
        }
    )

    result = lcia._extract_excluded_subregions(1, decomposed)
    assert result == frozenset(["PL-MZ", "HU"])


def test_excluded_subregions_skips_row(lcia):
    lcia.position_to_technosphere_flows_lookup = {
        2: {"name": "diesel", "reference product": "transport"},
    }
    lcia.technosphere_flows_lookup = {("diesel", "transport"): ["RoW", "ES"]}

    decomposed = frozenset(
        {("ES", ("ES-CT", "ES-MD")), ("RoW", ("XX",))}  # Should be ignored
    )

    result = lcia._extract_excluded_subregions(2, decomposed)
    assert result == frozenset(["ES-CT", "ES-MD"])


def test_missing_position_does_nothing(lcia):
    lcia.position_to_technosphere_flows_lookup = {}
    lcia.technosphere_flows_lookup = {}

    decomposed = frozenset()
    result = lcia._extract_excluded_subregions(999, decomposed)
    assert result == frozenset()


def test_missing_dataset_key_does_nothing(lcia):
    lcia.position_to_technosphere_flows_lookup = {
        4: {"name": "wind", "reference product": "onshore"},
    }
    lcia.technosphere_flows_lookup = {
        ("wind", "offshore"): ["SE"]  # Tuple key doesn't match
    }

    decomposed = frozenset({("SE", ("SE-N", "SE-S"))})

    result = lcia._extract_excluded_subregions(4, decomposed)
    assert result == frozenset()
