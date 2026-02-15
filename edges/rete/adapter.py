from __future__ import annotations

from typing import TYPE_CHECKING

from .engine import ClipsEngine, ReteExecutionInput
from .facts import attach_suppliers, build_biosphere_nodes, build_technosphere_nodes
from .rules import compile_rules
from edges.edgelcia import add_cf_entry

if TYPE_CHECKING:
    from edges.edgelcia import EdgeLCIA


def map_exchanges_clips(lcia: "EdgeLCIA"):
    """
    CLIPSpy-backed exchange mapping adapter.

    Executes exchange matching through CLIPS/RETE and writes results into
    ``lcia.cfs_mapping`` via ``add_cf_entry``.
    """
    # Keep initialization behavior aligned with python backend.
    lcia._ensure_filtered_lookups_for_current_edges()
    lcia._initialize_weights()

    supported_side_fields = {
        "matrix",
        "operator",
        "name",
        "reference product",
        "location",
        "categories",
        "classifications",
        "excludes",
    }
    unsupported = set()
    for cf in lcia.raw_cfs_data:
        unsupported |= set((cf.get("supplier") or {}).keys()) - supported_side_fields
        unsupported |= set((cf.get("consumer") or {}).keys()) - supported_side_fields
    if unsupported:
        raise NotImplementedError(
            "CLIPS backend does not support these matching fields yet: "
            f"{sorted(unsupported)}"
        )

    rules = compile_rules(lcia.raw_cfs_data)
    rules_by_id = {int(r["id"]): r for r in rules}

    tech_nodes = build_technosphere_nodes(lcia.technosphere_flows or [])
    bio_nodes = build_biosphere_nodes(lcia.biosphere_flows or [])

    # Consumers are always technosphere in the current matching model.
    tech_edges = set(lcia.technosphere_edges or set())
    bio_edges = set(lcia.biosphere_edges or set())
    edge_union = tech_edges | bio_edges
    attach_suppliers(tech_nodes, tech_edges, slot_name="tech_suppliers")
    attach_suppliers(tech_nodes, bio_edges, slot_name="bio_suppliers")

    data = ReteExecutionInput(
        rules=rules,
        bio_nodes=bio_nodes,
        tech_nodes=tech_nodes,
        edges=sorted(edge_union),
    )

    def _on_match(rule_id: int, supplier_id: int, consumer_id: int):
        rule = rules_by_id[rule_id]
        rule_supplier = rule.get("supplier", {})
        rule_consumer = rule.get("consumer", {})
        direction = f"{rule_supplier.get('matrix', 'technosphere')}-{rule_consumer.get('matrix', 'technosphere')}"

        add_cf_entry(
            cfs_mapping=lcia.cfs_mapping,
            supplier_info=rule_supplier,
            consumer_info=rule_consumer,
            direction=direction,
            indices=[(int(supplier_id), int(consumer_id))],
            value=rule["value"],
            uncertainty=rule.get("uncertainty"),
            seen_positions=lcia._seen_positions,
        )

    engine = ClipsEngine()
    engine.run(data, on_match=_on_match)

    lcia._update_unprocessed_edges()
    # For now we don't propagate location-only allowlists from CLIPS.
    lcia.eligible_edges_for_next_bio = set()
    lcia.eligible_edges_for_next_tech = set()
    lcia.applied_strategies.append("map_exchanges")
