import logging

from edges import EdgeLCIA
from edges.edgelcia import _equality_supplier_signature_cached
from edges.utils import make_hashable


class _FakeGeo:
    def batch(self, locations, containing=True, exceptions_map=None):
        assert containing is False
        mapping = {
            "CA-NF": ["CA"],
            "DE": ["RER"],
        }
        return {loc: mapping.get(loc, []) for loc in locations}


def test_map_contained_locations_fast_path_keeps_consumer_candidates_separate():
    lca = EdgeLCIA.__new__(EdgeLCIA)

    lca.logger = logging.getLogger(__name__)
    lca.raw_cfs_data = [
        {
            "supplier": {"name": "Carbon dioxide, fossil", "matrix": "biosphere"},
            "consumer": {"location": "CA", "matrix": "technosphere"},
            "value": 15,
        },
        {
            "supplier": {
                "name": "Carbon dioxide",
                "operator": "startswith",
                "matrix": "biosphere",
            },
            "consumer": {"location": "RER", "matrix": "technosphere"},
            "value": 3,
        },
    ]
    lca.weights = {
        ("__ANY__", "CA"): 0.0,
        ("__ANY__", "RER"): 0.0,
    }
    lca._geo = _FakeGeo()

    # Put the RER edge first to reproduce the old leakage bug.
    lca.unprocessed_biosphere_edges = [(1, 200), (1, 100)]
    lca.unprocessed_technosphere_edges = []
    lca.processed_biosphere_edges = []
    lca.processed_technosphere_edges = []
    lca.eligible_edges_for_next_bio = {(1, 200), (1, 100)}
    lca.eligible_edges_for_next_tech = set()

    lca.consumer_loc = {
        100: "CA-NF",
        200: "DE",
    }
    lca.required_supplier_fields = {"name"}
    lca.required_consumer_fields = set()
    lca._include_cls_in_supplier_sig = False
    lca._cached_supplier_keys = {
        _equality_supplier_signature_cached(
            make_hashable({"name": "Carbon dioxide, fossil"})
        )
    }
    lca.cf_index = {}
    lca.cfs_mapping = []
    lca._seen_positions = set()
    lca.applied_strategies = []

    supplier = {
        "name": "Carbon dioxide, fossil",
        "categories": ("air",),
        "matrix": "biosphere",
    }
    consumers = {
        100: {"location": "CA-NF", "matrix": "technosphere"},
        200: {"location": "DE", "matrix": "technosphere"},
    }

    lca._ensure_filtered_lookups_for_current_edges = lambda: None
    lca._prepare_restricted_lookups_from_unprocessed = lambda: None
    lca._initialize_weights = lambda: None
    lca._update_unprocessed_edges = lambda: None
    lca._get_supplier_info = lambda idx, direction: supplier
    lca._get_consumer_info = lambda idx: consumers[idx]

    calls = []

    def fake_compute_average_cf_cached(
        *,
        candidate_suppliers,
        candidate_consumers,
        supplier_info,
        consumer_info,
        cf_index,
        required_supplier_fields,
        required_consumer_fields,
    ):
        calls.append((tuple(candidate_suppliers), tuple(candidate_consumers)))
        if tuple(candidate_consumers) == ("CA",):
            return ("CF_CA", None, None)
        if tuple(candidate_consumers) == ("RER",):
            return ("CF_RER", None, None)
        return (0, None, None)

    lca._compute_average_cf_cached = fake_compute_average_cf_cached

    EdgeLCIA.map_contained_locations(lca)

    values_by_consumer = {}
    for cf in lca.cfs_mapping:
        for _, consumer_idx in cf["positions"]:
            values_by_consumer[consumer_idx] = cf["value"]

    assert values_by_consumer[100] == "CF_CA"
    assert values_by_consumer[200] == "CF_RER"
    assert set(calls) == {
        (("__ANY__",), ("CA",)),
        (("__ANY__",), ("RER",)),
    }
