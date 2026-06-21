from pathlib import Path
from types import SimpleNamespace
import logging

import bw2data
import numpy as np
import pytest
import sparse
from scipy.sparse import csr_matrix

import edges.edgelcia as edgelcia_module
from edges.edgelcia import EdgeLCIA


class _HashableActivity(dict):
    def __hash__(self):
        return id(self)


INTERPOLATION_POLICY = {
    "axis": "scenario_idx",
    "axis_type": "year",
    "method": "linear",
    "extrapolation": "nearest",
    "source_years": ["2024", "2029"],
}


def test_mixed_supplier_matrices_allowed(monkeypatch):
    method = {
        "name": "mixed",
        "version": "1.0",
        "unit": "kg",
        "exchanges": [
            {
                "supplier": {"matrix": "biosphere", "name": "CO2"},
                "consumer": {"matrix": "technosphere"},
                "value": 1.0,
            },
            {
                "supplier": {"matrix": "technosphere", "name": "electricity"},
                "consumer": {"matrix": "technosphere"},
                "value": 2.0,
            },
        ],
    }

    calls = {}

    class FakeLCA:
        def __init__(self, demand):
            calls["demand"] = demand
            self.demand = demand

    monkeypatch.setattr(
        edgelcia_module, "_bw2calc_lca_accepts_use_distributions", lambda: False
    )
    monkeypatch.setattr(edgelcia_module.bw2calc, "LCA", FakeLCA)
    monkeypatch.setattr(EdgeLCIA, "log_platform", lambda self: None)
    monkeypatch.setattr(EdgeLCIA, "_get_candidate_supplier_keys", lambda self: set())

    lcia = EdgeLCIA(demand={}, method=method)

    assert isinstance(lcia.lca, FakeLCA)
    assert calls["demand"] == {}
    assert lcia.supplier_matrix_types == {"biosphere", "technosphere"}


def test_uncertainty_references_are_resolved_on_load(monkeypatch):
    method = {
        "name": "uncertainty-ref",
        "version": "1.0",
        "unit": "kg",
        "uncertainties": {
            "co2-country": {
                "distribution": "discrete_empirical",
                "parameters": {"values": [1.0, 2.0], "weights": [0.25, 0.75]},
            }
        },
        "exchanges": [
            {
                "supplier": {"matrix": "biosphere", "name": "CO2"},
                "consumer": {"matrix": "technosphere"},
                "value": 1.0,
                "uncertainty_ref": "co2-country",
                "uncertainty_negative": 1,
            }
        ],
    }

    class FakeLCA:
        def __init__(self, demand):
            self.demand = demand

    monkeypatch.setattr(
        edgelcia_module, "_bw2calc_lca_accepts_use_distributions", lambda: False
    )
    monkeypatch.setattr(edgelcia_module.bw2calc, "LCA", FakeLCA)
    monkeypatch.setattr(EdgeLCIA, "log_platform", lambda self: None)
    monkeypatch.setattr(EdgeLCIA, "_get_candidate_supplier_keys", lambda self: set())

    lcia = EdgeLCIA(demand={}, method=method)

    uncertainty = lcia.raw_cfs_data[0]["uncertainty"]
    assert uncertainty["ref"] == "co2-country"
    assert uncertainty["negative"] == 1
    assert uncertainty["parameters"]["values"] == [1.0, 2.0]


def test_lci_builds_both_edge_sets_for_internal_mixed_methods(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.raw_cfs_data = [
        {
            "supplier": {"matrix": "biosphere", "name": "CO2"},
            "consumer": {"matrix": "technosphere"},
            "value": 1.0,
        },
        {
            "supplier": {"matrix": "technosphere", "name": "electricity"},
            "consumer": {"matrix": "technosphere"},
            "value": 2.0,
        },
    ]

    class FakeLCA:
        def __init__(self):
            self.inventory = csr_matrix(([1.0], ([0], [1])), shape=(2, 2))
            self.technosphere_matrix = csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
            self.supply_array = np.array([1.0, 1.0])
            self.biosphere_dict = {("biosphere", "co2"): 0}
            self.activity_dict = {("db", "A"): 0, ("db", "B"): 1}

        def lci(self, factorize=True):
            self.factorize = factorize

    fake_tech_edges = csr_matrix(([3.0], ([1], [0])), shape=(2, 2))

    monkeypatch.setattr(edgelcia_module, "bw2", True)
    monkeypatch.setattr(
        edgelcia_module,
        "build_technosphere_edges_matrix",
        lambda technosphere_matrix, supply_array: fake_tech_edges,
    )
    monkeypatch.setattr(
        edgelcia_module,
        "get_flow_matrix_positions",
        lambda mapping: [
            {"position": position, "name": str(key)}
            for key, position in mapping.items()
        ],
    )

    lcia.lca = FakeLCA()
    lcia.biosphere_edges = set()
    lcia.technosphere_edges = set()

    lcia.lci()

    assert lcia.biosphere_edges == {(0, 1)}
    assert lcia.technosphere_edges == {(1, 0)}
    assert lcia.technosphere_flow_matrix is fake_tech_edges
    assert lcia.lca.factorize is True


def test_evaluate_cfs_and_lcia_support_mixed_supplier_methods():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 10
    lcia.scenario = None
    lcia.parameters = {}
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.random_seed = 42
    lcia.random_state = np.random.default_rng(42)
    lcia.logger = logging.getLogger("test.edgelcia.robustness.mixed")
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}},
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
        },
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "direction": "biosphere-technosphere",
            "value": 10.0,
        },
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(1, 0)],
            "direction": "technosphere-technosphere",
            "value": 5.0,
        },
    ]
    lcia.biosphere_edges = {(0, 1)}
    lcia.technosphere_edges = {(1, 0)}
    lcia.processed_biosphere_edges = {(0, 1)}
    lcia.processed_technosphere_edges = {(1, 0)}
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([2.0], ([0], [1])), shape=(2, 2)),
        technosphere_matrix=csr_matrix((2, 2)),
    )
    lcia.technosphere_flow_matrix = csr_matrix(([3.0], ([1], [0])), shape=(2, 2))

    lcia.evaluate_cfs()

    assert lcia.characterization_matrix is None
    assert lcia.characterization_matrices["biosphere"] is not None
    assert lcia.characterization_matrices["technosphere"] is not None

    lcia.lcia()

    assert lcia.score == pytest.approx(35.0)
    assert lcia.score_by_matrix["biosphere"] == pytest.approx(20.0)
    assert lcia.score_by_matrix["technosphere"] == pytest.approx(15.0)
    assert lcia.characterized_inventory is None
    assert lcia.characterized_inventories["biosphere"] is not None
    assert lcia.characterized_inventories["technosphere"] is not None


def test_evaluate_cfs_prefers_value_expression_over_baseline_value():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 1
    lcia.scenario = "SSP126"
    lcia.parameters = {"SSP126": {"cf_ad": {"2049": 93.9}}}
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.value_expression")
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}}
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 0)],
            "direction": "biosphere-technosphere",
            "value": 80.5,
            "value_expression": "cf_ad",
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([1.0], ([0], [0])), shape=(1, 1)),
        technosphere_matrix=csr_matrix((1, 1)),
    )

    lcia.evaluate_cfs(scenario_idx="2049")

    assert lcia.scenario_cfs[0]["value"] == pytest.approx(93.9)
    assert lcia.characterization_matrices["biosphere"][0, 0] == pytest.approx(93.9)


def test_evaluate_cfs_interpolates_value_expression_years():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 1
    lcia.scenario = "SSP126"
    lcia.parameters = {
        "SSP126": {
            "cf_ad": {"2024": 80.0, "2029": 100.0},
        }
    }
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.interpolate_value")
    lcia.method_metadata = {"interpolation": INTERPOLATION_POLICY}
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}}
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 0)],
            "direction": "biosphere-technosphere",
            "value": 80.0,
            "value_expression": "cf_ad",
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([1.0], ([0], [0])), shape=(1, 1)),
        technosphere_matrix=csr_matrix((1, 1)),
    )

    lcia.evaluate_cfs(scenario_idx="2026.5")

    assert lcia.scenario_cfs[0]["value"] == pytest.approx(90.0)
    assert lcia.characterization_matrices["biosphere"][0, 0] == pytest.approx(90.0)


def test_evaluate_cfs_keeps_legacy_parameter_fallback_without_policy():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 1
    lcia.scenario = "SSP126"
    lcia.parameters = {
        "SSP126": {
            "cf_ad": {"2024": 80.0, "2029": 100.0},
        }
    }
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.legacy_value")
    lcia.method_metadata = {}
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}}
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 0)],
            "direction": "biosphere-technosphere",
            "value": 80.0,
            "value_expression": "cf_ad",
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([1.0], ([0], [0])), shape=(1, 1)),
        technosphere_matrix=csr_matrix((1, 1)),
    )

    lcia.evaluate_cfs(scenario_idx="2026.5")

    assert lcia.scenario_cfs[0]["value"] == pytest.approx(100.0)
    assert lcia.characterization_matrices["biosphere"][0, 0] == pytest.approx(100.0)


def test_evaluate_cfs_uses_reporting_split_for_dynamic_aggregate_value():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 1
    lcia.scenario = "SSP126"
    lcia.parameters = {
        "SSP126": {
            "cf_ch": {"2049": 10.0},
            "cf_de": {"2049": 30.0},
            "wt_ch": {"2049": 1.0},
            "wt_de": {"2049": 3.0},
        }
    }
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.dynamic_split")
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}}
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 0)],
            "direction": "biosphere-technosphere",
            "value": "this_expression_should_not_be_evaluated",
            "reporting_split": (
                {
                    "consumer_location": "CH",
                    "share": 0.5,
                    "value": 1.0,
                    "value_expression": "cf_ch",
                    "weight": 1.0,
                    "weight_expression": "wt_ch",
                },
                {
                    "consumer_location": "DE",
                    "share": 0.5,
                    "value": 1.0,
                    "value_expression": "cf_de",
                    "weight": 1.0,
                    "weight_expression": "wt_de",
                },
            ),
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([1.0], ([0], [0])), shape=(1, 1)),
        technosphere_matrix=csr_matrix((1, 1)),
    )

    lcia.evaluate_cfs(scenario_idx="2049")

    assert lcia.scenario_cfs[0]["value"] == pytest.approx(25.0)
    assert lcia.scenario_cfs[0]["reporting_split"][0]["share"] == pytest.approx(0.25)
    assert lcia.scenario_cfs[0]["reporting_split"][1]["share"] == pytest.approx(0.75)
    assert lcia.characterization_matrices["biosphere"][0, 0] == pytest.approx(25.0)


def test_evaluate_cfs_interpolates_dynamic_aggregate_weights():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 1
    lcia.scenario = "SSP126"
    lcia.parameters = {
        "SSP126": {
            "cf_ch": {"2024": 10.0, "2029": 10.0},
            "cf_de": {"2024": 30.0, "2029": 30.0},
            "wt_ch": {"2024": 1.0, "2029": 3.0},
            "wt_de": {"2024": 3.0, "2029": 1.0},
        }
    }
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.interpolate_split")
    lcia.method_metadata = {"interpolation": INTERPOLATION_POLICY}
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}}
    ]
    lcia.cfs_mapping = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 0)],
            "direction": "biosphere-technosphere",
            "value": 0.0,
            "reporting_split": (
                {
                    "consumer_location": "CH",
                    "share": 0.25,
                    "value": 10.0,
                    "value_expression": "cf_ch",
                    "weight": 1.0,
                    "weight_expression": "wt_ch",
                },
                {
                    "consumer_location": "DE",
                    "share": 0.75,
                    "value": 30.0,
                    "value_expression": "cf_de",
                    "weight": 3.0,
                    "weight_expression": "wt_de",
                },
            ),
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([1.0], ([0], [0])), shape=(1, 1)),
        technosphere_matrix=csr_matrix((1, 1)),
    )

    lcia.evaluate_cfs(scenario_idx="2026.5")

    assert lcia.scenario_cfs[0]["value"] == pytest.approx(20.0)
    assert lcia.scenario_cfs[0]["reporting_split"][0]["share"] == pytest.approx(0.5)
    assert lcia.scenario_cfs[0]["reporting_split"][1]["share"] == pytest.approx(0.5)


def test_lcia_uncertainty_supports_mixed_supplier_methods():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = True
    lcia.iterations = 3
    lcia.processed_biosphere_edges = {(0, 1)}
    lcia.processed_technosphere_edges = {(1, 0)}
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}},
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
        },
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([2.0], ([0], [1])), shape=(2, 2)),
        technosphere_matrix=csr_matrix((2, 2)),
    )
    lcia.technosphere_flow_matrix = csr_matrix(([3.0], ([1], [0])), shape=(2, 2))
    lcia.characterization_matrix = None
    lcia.characterization_matrices = {
        "biosphere": sparse.COO(
            coords=np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]]),
            data=np.array([10.0, 20.0, 30.0]),
            shape=(2, 2, 3),
        ),
        "technosphere": sparse.COO(
            coords=np.array([[1, 1, 1], [0, 0, 0], [0, 1, 2]]),
            data=np.array([1.0, 2.0, 3.0]),
            shape=(2, 2, 3),
        ),
    }
    lcia.logger = logging.getLogger("test.edgelcia.robustness.mixed.uncertainty")

    lcia.lcia()

    assert isinstance(lcia.score, np.ndarray)
    assert np.allclose(lcia.score, np.array([23.0, 46.0, 69.0]))
    assert np.allclose(lcia.score_by_matrix["biosphere"], np.array([20.0, 40.0, 60.0]))
    assert np.allclose(lcia.score_by_matrix["technosphere"], np.array([3.0, 6.0, 9.0]))


def test_generate_cf_table_supports_mixed_supplier_methods(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.logger = logging.getLogger("test.edgelcia.robustness.mixed.table")
    lcia.use_distributions = False
    lcia.inventory_use_distributions = False
    lcia.iterations = 1
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}},
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
        },
    ]
    lcia.scenario_cfs = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "direction": "biosphere-technosphere",
            "value": 10.0,
        },
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(1, 0)],
            "direction": "technosphere-technosphere",
            "value": 5.0,
        },
    ]
    lcia.characterization_matrix = None
    lcia.characterization_matrices = {
        "biosphere": csr_matrix(([10.0], ([0], [1])), shape=(2, 2)),
        "technosphere": csr_matrix(([5.0], ([1], [0])), shape=(2, 2)),
    }
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([2.0], ([0], [1])), shape=(2, 2)),
        technosphere_matrix=csr_matrix((2, 2)),
    )
    lcia.technosphere_flow_matrix = csr_matrix(([3.0], ([1], [0])), shape=(2, 2))
    lcia.reversed_biosphere = {0: "bio-flow"}
    lcia.reversed_activity = {0: "consumer-activity", 1: "tech-supplier"}
    lcia.unprocessed_biosphere_edges = []
    lcia.unprocessed_technosphere_edges = []

    def _get_activity(key):
        if key == "bio-flow":
            return {
                "name": "Carbon dioxide",
                "categories": ("air",),
                "classifications": None,
            }
        if key == "tech-supplier":
            return {
                "name": "Electricity mix",
                "reference product": "electricity",
                "location": "CH",
                "classifications": None,
            }
        if key == "consumer-activity":
            return {
                "name": "Consumer activity",
                "reference product": "service",
                "location": "DE",
                "classifications": None,
            }
        raise KeyError(key)

    monkeypatch.setattr(bw2data, "get_activity", _get_activity)

    df = lcia.generate_cf_table()

    assert set(df["supplier matrix"]) == {"biosphere", "technosphere"}
    assert set(df["direction"]) == {
        "biosphere-technosphere",
        "technosphere-technosphere",
    }
    by_matrix = df.set_index("supplier matrix")
    assert by_matrix.loc["biosphere", "impact"] == pytest.approx(20.0)
    assert by_matrix.loc["biosphere", "supplier categories"] == ("air",)
    assert by_matrix.loc["technosphere", "impact"] == pytest.approx(15.0)
    assert by_matrix.loc["technosphere", "supplier location"] == "CH"
    assert by_matrix.loc["technosphere", "supplier reference product"] == "electricity"


def test_redo_lcia_supports_mixed_supplier_methods(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.inventory_use_distributions = False
    lcia.store_inventory_samples = False
    lcia.iterations = 1
    lcia.random_seed = 42
    lcia.random_state = np.random.default_rng(42)
    lcia.parameters = {}
    lcia.scenario = None
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.logger = logging.getLogger("test.edgelcia.robustness.mixed.redo")
    lcia.raw_cfs_data = [
        {"supplier": {"matrix": "biosphere"}, "consumer": {"matrix": "technosphere"}},
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
        },
    ]
    lcia.characterization_matrix = None
    lcia.characterization_matrices = {
        "biosphere": csr_matrix(([10.0], ([0], [1])), shape=(2, 2)),
        "technosphere": csr_matrix(([5.0], ([1], [0])), shape=(2, 2)),
    }
    lcia.scenario_cfs = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "direction": "biosphere-technosphere",
            "value": 10.0,
        },
        {
            "supplier": {"matrix": "technosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(1, 0)],
            "direction": "technosphere-technosphere",
            "value": 5.0,
        },
    ]
    lcia.scenario_cfs_by_matrix = {
        "biosphere": [lcia.scenario_cfs[0]],
        "technosphere": [lcia.scenario_cfs[1]],
    }
    lcia.cfs_mapping = list(lcia.scenario_cfs)
    lcia.processed_biosphere_edges = {(0, 1)}
    lcia.processed_technosphere_edges = {(1, 0)}
    lcia.unprocessed_biosphere_edges = []
    lcia.unprocessed_technosphere_edges = []
    lcia.biosphere_edges = {(0, 1)}
    lcia.technosphere_edges = {(1, 0)}
    lcia._last_edges_snapshot_bio = set()
    lcia._last_edges_snapshot_tech = set()
    lcia._last_nonempty_edges_snapshot_bio = set()
    lcia._last_nonempty_edges_snapshot_tech = set()
    lcia._ever_seen_edges_bio = set()
    lcia._ever_seen_edges_tech = set()
    lcia._failed_edges_bio = set()
    lcia._failed_edges_tech = set()
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.reversed_biosphere = {0: "bio-old", 1: "bio-new"}
    lcia.reversed_activity = {0: "tech-new", 1: "consumer"}

    current_inventory = csr_matrix(([2.0, 4.0], ([0, 1], [1, 1])), shape=(2, 2))
    current_tech_edges = csr_matrix(([5.0, 3.0], ([0, 1], [0, 0])), shape=(2, 2))

    class FakeLCA:
        def __init__(self):
            self.inventory = current_inventory
            self.technosphere_matrix = csr_matrix((2, 2))
            self.supply_array = np.array([1.0, 1.0])
            self.demand = {}

        def redo_lci(self, demand=None):
            self.last_redo_demand = demand

    lcia.lca = FakeLCA()
    lcia.technosphere_flow_matrix = csr_matrix(([3.0], ([1], [0])), shape=(2, 2))

    monkeypatch.setattr(
        edgelcia_module,
        "build_technosphere_edges_matrix",
        lambda technosphere_matrix, supply_array: current_tech_edges,
    )

    def _map_exchanges():
        assert lcia.biosphere_edges == {(1, 1)}
        assert lcia.technosphere_edges == {(0, 0)}
        lcia.cfs_mapping.extend(
            [
                {
                    "supplier": {"matrix": "biosphere"},
                    "consumer": {"matrix": "technosphere"},
                    "positions": [(1, 1)],
                    "direction": "biosphere-technosphere",
                    "value": 7.0,
                },
                {
                    "supplier": {"matrix": "technosphere"},
                    "consumer": {"matrix": "technosphere"},
                    "positions": [(0, 0)],
                    "direction": "technosphere-technosphere",
                    "value": 11.0,
                },
            ]
        )

    lcia.map_exchanges = _map_exchanges
    lcia.apply_strategies = lambda strategies=None: lcia

    lcia.redo_lcia(recompute_score=True)

    assert lcia.characterization_matrices["biosphere"][1, 1] == pytest.approx(7.0)
    assert lcia.characterization_matrices["technosphere"][0, 0] == pytest.approx(11.0)
    assert lcia.score == pytest.approx(118.0)
    assert lcia.score_by_matrix["biosphere"] == pytest.approx(48.0)
    assert lcia.score_by_matrix["technosphere"] == pytest.approx(70.0)
    assert set(lcia.biosphere_edges) == {(0, 1), (1, 1)}
    assert set(lcia.technosphere_edges) == {(0, 0), (1, 0)}


def test_statistics_accepts_string_method():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    act = _HashableActivity(name="dummy activity")
    lcia.lca = SimpleNamespace(demand={act: 1})
    lcia.method = "custom_method.json"
    lcia.method_metadata = {}
    lcia.filepath = Path("custom_method.json")
    lcia.cfs_number = 0
    lcia.cfs_mapping = []
    lcia.ignored_method_exchanges = []
    lcia.ignored_locations = set()
    lcia.processed_biosphere_edges = set()
    lcia.processed_technosphere_edges = set()
    lcia.unprocessed_biosphere_edges = []
    lcia.unprocessed_technosphere_edges = []

    lcia.statistics()


def test_constructor_skips_use_distributions_kw_when_bw2calc_lca_does_not_support_it(
    monkeypatch,
):
    calls = {}

    class FakeLCA:
        def __init__(self, demand):
            calls["demand"] = demand
            self.demand = demand

    monkeypatch.setattr(
        edgelcia_module, "_bw2calc_lca_accepts_use_distributions", lambda: False
    )
    monkeypatch.setattr(edgelcia_module.bw2calc, "LCA", FakeLCA)
    monkeypatch.setattr(
        EdgeLCIA,
        "_load_raw_lcia_data",
        lambda self: setattr(
            self,
            "raw_cfs_data",
            [
                {
                    "supplier": {"matrix": "biosphere", "name": "CO2"},
                    "consumer": {"matrix": "technosphere"},
                    "value": 1.0,
                }
            ],
        ),
    )
    monkeypatch.setattr(EdgeLCIA, "log_platform", lambda self: None)
    monkeypatch.setattr(EdgeLCIA, "_get_candidate_supplier_keys", lambda self: set())

    lcia = EdgeLCIA(
        demand={},
        method={
            "name": "dummy",
            "unit": "kg",
            "exchanges": [
                {
                    "supplier": {"matrix": "biosphere", "name": "CO2"},
                    "consumer": {"matrix": "technosphere"},
                    "value": 1.0,
                }
            ],
        },
    )

    assert isinstance(lcia.lca, FakeLCA)
    assert calls["demand"] == {}


def test_build_inventory_mc_lca_falls_back_to_legacy_monte_carlo(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.lca = SimpleNamespace(demand={"foo": 1})
    lcia.demand = {"foo": 1}
    lcia.random_seed = 42

    class FakeMonteCarloLCA:
        def __init__(self, demand, seed=None):
            self.demand = demand
            self.seed = seed
            self._step = 0
            self.inventory = csr_matrix(([0.0], ([0], [0])), shape=(1, 1))
            self.technosphere_matrix = csr_matrix((1, 1))
            self.supply_array = np.array([1.0])

        def __next__(self):
            self._step += 1
            self.inventory = csr_matrix(([float(self._step)], ([0], [0])), shape=(1, 1))
            return self.supply_array

    monkeypatch.setattr(
        edgelcia_module, "_bw2calc_lca_accepts_use_distributions", lambda: False
    )
    monkeypatch.setattr(
        edgelcia_module.bw2calc, "MonteCarloLCA", FakeMonteCarloLCA, raising=False
    )

    mc_lca = lcia._build_inventory_mc_lca()
    assert isinstance(mc_lca, edgelcia_module._LegacyInventoryMonteCarloAdapter)

    mc_lca.keep_first_iteration()
    next(mc_lca)
    first = mc_lca.inventory[0, 0]

    next(mc_lca)
    second = mc_lca.inventory[0, 0]

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(2.0)


def _build_minimal_lcia(cfs_mapping):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.iterations = 10
    lcia.cfs_mapping = cfs_mapping
    lcia.scenario = None
    lcia.parameters = {"baseline": {"x": {"2020": 2.0}}}
    lcia.SAFE_GLOBALS = {"__builtins__": None}
    lcia.biosphere_edges = {(0, 0)}
    lcia.technosphere_edges = set()
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix((2, 2)),
        technosphere_matrix=csr_matrix((2, 2)),
    )
    lcia._last_eval_scenario_name = None
    lcia._last_eval_scenario_idx = None
    lcia.logger = logging.getLogger("test.edgelcia.robustness")
    return lcia


def test_evaluate_cfs_deterministic_across_cf_order():
    cfs_a = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "value": "x + 1",
        },
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(1, 0)],
            "value": 5.0,
        },
    ]
    cfs_b = list(reversed(cfs_a))

    lcia_a = _build_minimal_lcia(cfs_a)
    lcia_b = _build_minimal_lcia(cfs_b)

    lcia_a.evaluate_cfs(scenario_idx="2020")
    lcia_b.evaluate_cfs(scenario_idx="2020")

    assert (lcia_a.characterization_matrix != lcia_b.characterization_matrix).nnz == 0


def test_lcia_uncertainty_returns_dense_score_vector():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = True
    lcia.iterations = 3
    lcia.processed_biosphere_edges = {(0, 0)}
    lcia.processed_technosphere_edges = set()
    lcia.raw_cfs_data = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
        }
    ]
    lcia.lca = SimpleNamespace(inventory=csr_matrix(([2.0], ([0], [1])), shape=(2, 2)))
    lcia.technosphere_flow_matrix = None
    lcia.characterization_matrix = sparse.COO(
        coords=np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]]),
        data=np.array([10.0, 20.0, 30.0]),
        shape=(2, 2, 3),
    )
    lcia.logger = logging.getLogger("test.edgelcia.robustness")

    lcia.lcia()

    assert isinstance(lcia.score, np.ndarray)
    assert lcia.score.shape == (3,)
    assert np.allclose(lcia.score, np.array([20.0, 40.0, 60.0]))


def test_map_exchanges_rejects_unknown_backend():
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.matcher_backend = "unknown"
    with pytest.raises(ValueError):
        lcia.map_exchanges()


def test_map_exchanges_dispatches_clips_backend(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.matcher_backend = "clips"

    import edges.rete.adapter as adapter

    called = {"clips": False}

    def _clips(obj):
        called["clips"] = True
        assert obj is lcia
        return "clips-ok"

    monkeypatch.setattr(adapter, "map_exchanges_clips", _clips)
    assert lcia.map_exchanges() == "clips-ok"
    assert called["clips"] is True


def test_warn_duplicate_matching_signatures_logs_warning(caplog):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.logger = logging.getLogger("test.edgelcia.robustness.duplicates")
    lcia.raw_cfs_data = [
        {
            "supplier": {
                "matrix": "biosphere",
                "name": "Water",
                "categories": ("water",),
                "unit": "m3",
            },
            "consumer": {"matrix": "technosphere", "location": "GLO"},
            "value": -42.95,
        },
        {
            "supplier": {
                "matrix": "biosphere",
                "name": "Water",
                "categories": ("water",),
                "unit": "m3",
            },
            "consumer": {"matrix": "technosphere", "location": "GLO"},
            "value": -0.04295,
        },
    ]

    with caplog.at_level(logging.WARNING, logger=lcia.logger.name):
        lcia._warn_duplicate_matching_signatures()

    assert len(lcia.duplicate_method_signature_groups) == 1
    assert lcia.duplicate_method_signature_groups[0]["indices"] == (0, 1)
    assert "duplicate CF matching signature group" in caplog.text
    assert "Water" in caplog.text


class _FakeInventoryMCLCA:
    def __init__(self, inventories):
        self._inventories = inventories
        self._index = 0
        self.inventory = inventories[0]

    def keep_first_iteration(self):
        self.keep_first_iteration_flag = True

    def __next__(self):
        if getattr(self, "keep_first_iteration_flag", False):
            delattr(self, "keep_first_iteration_flag")
        else:
            self._index += 1
            self.inventory = self._inventories[self._index]
        return self


def test_lcia_joint_uncertainty_reuses_inventory_iterations(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = True
    lcia.inventory_use_distributions = True
    lcia.store_inventory_samples = True
    lcia.iterations = 3
    lcia.processed_biosphere_edges = {(0, 1)}
    lcia.processed_technosphere_edges = set()
    lcia.raw_cfs_data = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
        }
    ]
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([2.0], ([0], [1])), shape=(2, 2)),
        demand={},
        use_distributions=True,
    )
    lcia.technosphere_flow_matrix = None
    lcia.characterization_matrix = sparse.COO(
        coords=np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]]),
        data=np.array([10.0, 20.0, 30.0]),
        shape=(2, 2, 3),
    )
    lcia.logger = logging.getLogger("test.edgelcia.robustness.joint")

    fake_iter = _FakeInventoryMCLCA(
        [
            csr_matrix(([2.0], ([0], [1])), shape=(2, 2)),
            csr_matrix(([3.0], ([0], [1])), shape=(2, 2)),
            csr_matrix(([4.0], ([0], [1])), shape=(2, 2)),
        ]
    )
    monkeypatch.setattr(lcia, "_build_inventory_mc_lca", lambda: fake_iter)

    lcia.lcia()

    assert isinstance(lcia.score, np.ndarray)
    assert np.allclose(lcia.score, np.array([20.0, 60.0, 120.0]))
    assert lcia.inventory_samples.shape == (2, 2, 3)
    assert np.allclose(
        np.array(lcia.inventory_samples[0, 1, :].todense()).reshape(-1),
        np.array([2.0, 3.0, 4.0]),
    )
    assert hasattr(lcia.lca, "inventory_samples")


def test_generate_cf_table_uses_inventory_samples_in_joint_mode(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.use_distributions = False
    lcia.inventory_use_distributions = True
    lcia.store_inventory_samples = True
    lcia.iterations = 3
    lcia.logger = logging.getLogger("test.edgelcia.robustness.joint.table")
    lcia.scenario_cfs = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "value": 10.0,
        }
    ]
    lcia.characterization_matrix = csr_matrix(([10.0], ([0], [1])), shape=(2, 2))
    lcia.technosphere_flow_matrix = None
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([999.0], ([0], [1])), shape=(2, 2)),
        use_distributions=True,
    )
    lcia.inventory_samples = sparse.COO(
        coords=np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]]),
        data=np.array([2.0, 3.0, 4.0]),
        shape=(2, 2, 3),
    )
    lcia._inventory_samples_matrix_kind = "biosphere"
    lcia.reversed_biosphere = {0: "bio-flow"}
    lcia.reversed_activity = {1: "consumer-activity"}

    def _get_activity(key):
        if key == "bio-flow":
            return {
                "name": "Water",
                "categories": ("water",),
                "classifications": None,
            }
        if key == "consumer-activity":
            return {
                "name": "Dummy activity",
                "reference product": "dummy product",
                "location": "CH",
                "classifications": None,
            }
        raise KeyError(key)

    monkeypatch.setattr(bw2data, "get_activity", _get_activity)

    df = lcia.generate_cf_table()

    assert df.shape[0] == 1
    assert df.loc[0, "amount"] == pytest.approx(3.0)
    assert df.loc[0, "amount (mean)"] == pytest.approx(3.0)
    assert df.loc[0, "CF (mean)"] == pytest.approx(10.0)
    assert df.loc[0, "impact (50th)"] == pytest.approx(30.0)


def test_generate_cf_table_splits_weighted_reporting_rows(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.logger = logging.getLogger("test.edgelcia.robustness.split.table")
    lcia.scenario_cfs = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "value": 16.0,
            "reporting_split": (
                {"consumer_location": "CH", "share": 0.25, "value": 10.0},
                {"consumer_location": "DE", "share": 0.25, "value": 14.0},
                {"consumer_location": "DE", "share": 0.50, "value": 20.0},
            ),
        }
    ]
    lcia.characterization_matrix = csr_matrix(([16.0], ([0], [1])), shape=(2, 2))
    lcia.technosphere_flow_matrix = None
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([4.0], ([0], [1])), shape=(2, 2)),
    )
    lcia.reversed_biosphere = {0: "bio-flow"}
    lcia.reversed_activity = {1: "consumer-activity"}

    def _get_activity(key):
        if key == "bio-flow":
            return {
                "name": "Water",
                "categories": ("water",),
                "classifications": None,
            }
        if key == "consumer-activity":
            return {
                "name": "Dummy activity",
                "reference product": "dummy product",
                "location": "RER",
                "classifications": None,
            }
        raise KeyError(key)

    monkeypatch.setattr(bw2data, "get_activity", _get_activity)

    df_unsplit = lcia.generate_cf_table(split_aggregate_consumers=False)
    df = lcia.generate_cf_table(split_aggregate_consumers=True)

    assert list(df_unsplit["consumer location"]) == ["RER"]
    assert df_unsplit["amount"].sum() == pytest.approx(4.0)
    assert df_unsplit["impact"].sum() == pytest.approx(64.0)
    assert df_unsplit["impact"].sum() == pytest.approx(df["impact"].sum())
    assert list(df["consumer location"]) == ["CH", "DE"]
    assert df["amount"].sum() == pytest.approx(4.0)
    assert df["impact"].sum() == pytest.approx(64.0)

    by_location = df.set_index("consumer location")
    assert by_location.loc["CH", "amount"] == pytest.approx(1.0)
    assert by_location.loc["CH", "CF"] == pytest.approx(10.0)
    assert by_location.loc["CH", "impact"] == pytest.approx(10.0)
    assert by_location.loc["DE", "amount"] == pytest.approx(3.0)
    assert by_location.loc["DE", "CF"] == pytest.approx(18.0)
    assert by_location.loc["DE", "impact"] == pytest.approx(54.0)


def test_generate_cf_table_splits_single_component_reporting_row(monkeypatch):
    lcia = EdgeLCIA.__new__(EdgeLCIA)
    lcia.logger = logging.getLogger("test.edgelcia.robustness.split.single")
    lcia.scenario_cfs = [
        {
            "supplier": {"matrix": "biosphere"},
            "consumer": {"matrix": "technosphere"},
            "positions": [(0, 1)],
            "value": 8.5e-07,
            "reporting_split": (
                {
                    "consumer_location": "BM",
                    "share": 1.0,
                    "value": 8.5e-07,
                    "weight": 5.0,
                },
            ),
        }
    ]
    lcia.characterization_matrix = csr_matrix(([8.5e-07], ([0], [1])), shape=(2, 2))
    lcia.technosphere_flow_matrix = None
    lcia.lca = SimpleNamespace(
        inventory=csr_matrix(([4.0], ([0], [1])), shape=(2, 2)),
    )
    lcia.reversed_biosphere = {0: "bio-flow"}
    lcia.reversed_activity = {1: "consumer-activity"}

    def _get_activity(key):
        if key == "bio-flow":
            return {
                "name": "Occupation, dump site",
                "categories": ("natural resource", "land"),
                "classifications": None,
            }
        if key == "consumer-activity":
            return {
                "name": "Hard coal, at mine",
                "reference product": "hard coal",
                "location": "RNA",
                "classifications": None,
            }
        raise KeyError(key)

    monkeypatch.setattr(bw2data, "get_activity", _get_activity)

    df_unsplit = lcia.generate_cf_table(split_aggregate_consumers=False)
    df_split = lcia.generate_cf_table(split_aggregate_consumers=True)

    assert list(df_unsplit["consumer location"]) == ["RNA"]
    assert list(df_split["consumer location"]) == ["BM"]
    assert df_unsplit["impact"].sum() == pytest.approx(df_split["impact"].sum())
