from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import TYPE_CHECKING

from .engine import ClipsEngine, ReteExecutionInput
from .facts import attach_suppliers, build_biosphere_nodes, build_technosphere_nodes
from .rules import compile_rules
from edges.edgelcia import add_cf_entry

if TYPE_CHECKING:
    from edges.edgelcia import EdgeLCIA

_CLIPS_ENGINE_CACHE_MAX = 4
_CLIPS_ENGINE_CACHE: "OrderedDict[tuple[str, str], ClipsEngine]" = OrderedDict()


def _rules_signature(rules: list[dict]) -> str:
    payload = json.dumps(rules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _engine_cache_key(lcia: "EdgeLCIA", rules_sig: str) -> tuple[str, str]:
    method_repr = repr(getattr(lcia, "method", None))
    return method_repr, rules_sig


def _get_or_create_engine(cache_key: tuple[str, str]) -> ClipsEngine:
    engine = _CLIPS_ENGINE_CACHE.get(cache_key)
    if engine is not None:
        _CLIPS_ENGINE_CACHE.move_to_end(cache_key)
        return engine

    engine = ClipsEngine()
    _CLIPS_ENGINE_CACHE[cache_key] = engine
    if len(_CLIPS_ENGINE_CACHE) > _CLIPS_ENGINE_CACHE_MAX:
        _CLIPS_ENGINE_CACHE.popitem(last=False)
    return engine


def _get_or_build_nodes(lcia: "EdgeLCIA") -> tuple[list[dict], list[dict]]:
    flow_key = (
        getattr(lcia, "_flows_version", None),
        len(lcia.technosphere_flows or ()),
        len(lcia.biosphere_flows or ()),
    )
    cache = getattr(lcia, "_clips_nodes_cache", None)
    if cache and cache.get("key") == flow_key:
        tech_base = cache["tech_base"]
        bio_base = cache["bio_base"]
    else:
        tech_base = build_technosphere_nodes(lcia.technosphere_flows or [])
        bio_base = build_biosphere_nodes(lcia.biosphere_flows or [])
        lcia._clips_nodes_cache = {
            "key": flow_key,
            "tech_base": tech_base,
            "bio_base": bio_base,
        }

    tech_nodes = [
        {
            **node,
            "bio_suppliers": [],
            "tech_suppliers": [],
        }
        for node in tech_base
    ]
    bio_nodes = [
        {
            **node,
            "bio_suppliers": [],
            "tech_suppliers": [],
        }
        for node in bio_base
    ]
    return tech_nodes, bio_nodes


def map_exchanges_clips(lcia: "EdgeLCIA"):
    """
    CLIPSpy-backed exchange mapping adapter.

    Executes exchange matching through CLIPS/RETE and writes results into
    ``lcia.cfs_mapping`` via ``add_cf_entry``.
    """
    # Keep initialization behavior aligned with legacy matcher setup.
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
    rules_sig = _rules_signature(rules)
    cache_key = _engine_cache_key(lcia, rules_sig)
    rules_by_id = {int(r["id"]): r for r in rules}
    direction_by_id: dict[int, str] = {}
    is_bio_rule_by_id: dict[int, bool] = {}
    for rid, rule in rules_by_id.items():
        supplier = rule.get("supplier", {}) or {}
        consumer = rule.get("consumer", {}) or {}
        direction = (
            f"{supplier.get('matrix', 'technosphere')}-"
            f"{consumer.get('matrix', 'technosphere')}"
        )
        direction_by_id[rid] = direction
        is_bio_rule_by_id[rid] = direction.startswith("biosphere-")

    tech_nodes, bio_nodes = _get_or_build_nodes(lcia)

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

    full_bio_matches: set[tuple[int, int]] = set()
    full_tech_matches: set[tuple[int, int]] = set()
    no_loc_bio_matches: set[tuple[int, int]] = set()
    no_loc_tech_matches: set[tuple[int, int]] = set()

    def _on_match(rule_id: int, supplier_id: int, consumer_id: int):
        rule = rules_by_id[rule_id]
        rule_supplier = rule.get("supplier", {})
        rule_consumer = rule.get("consumer", {})
        direction = direction_by_id[rule_id]

        pair = (int(supplier_id), int(consumer_id))
        if is_bio_rule_by_id[rule_id]:
            full_bio_matches.add(pair)
        else:
            full_tech_matches.add(pair)

        add_cf_entry(
            cfs_mapping=lcia.cfs_mapping,
            supplier_info=rule_supplier,
            consumer_info=rule_consumer,
            direction=direction,
            indices=[pair],
            value=rule["value"],
            uncertainty=rule.get("uncertainty"),
            seen_positions=lcia._seen_positions,
        )

    engine = _get_or_create_engine(cache_key)
    loc_rejects = engine.run(data, rules_signature=rules_sig, on_match=_on_match)
    for kind, supplier_id, consumer_id in loc_rejects:
        pair = (int(supplier_id), int(consumer_id))
        if kind == "bio":
            no_loc_bio_matches.add(pair)
        else:
            no_loc_tech_matches.add(pair)

    lcia._update_unprocessed_edges()

    lcia.eligible_edges_for_next_bio = no_loc_bio_matches - full_bio_matches
    lcia.eligible_edges_for_next_tech = no_loc_tech_matches - full_tech_matches
    lcia.applied_strategies.append("map_exchanges")
