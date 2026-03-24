from pathlib import Path
from types import SimpleNamespace
import logging

import numpy as np
import pytest
import sparse
from scipy.sparse import csr_matrix

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
