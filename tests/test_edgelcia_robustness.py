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


def test_mixed_supplier_matrices_rejected():
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

    with pytest.raises(NotImplementedError):
        EdgeLCIA(demand={}, method=method, lca=object())


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
            self.inventory = csr_matrix(
                ([float(self._step)], ([0], [0])), shape=(1, 1)
            )
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
