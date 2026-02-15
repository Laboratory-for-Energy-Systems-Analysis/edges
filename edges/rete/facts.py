from __future__ import annotations

from collections import defaultdict
from typing import Any


def _normalize_categories(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "|||".join(str(v) for v in value)
    return str(value)


def _iter_classification_pairs(value: Any) -> list[tuple[str, str]]:
    if not value:
        return []
    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for scheme, codes in value.items():
            seq = codes if isinstance(codes, (list, tuple, set)) else [codes]
            for c in seq:
                code = str(c).split(":", 1)[0].strip()
                if code:
                    pairs.append((str(scheme).lower().strip(), code))
        return pairs
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                scheme, codes = item
                seq = codes if isinstance(codes, (list, tuple, set)) else [codes]
                for c in seq:
                    code = str(c).split(":", 1)[0].strip()
                    if code:
                        pairs.append((str(scheme).lower().strip(), code))
        return pairs
    return pairs


def _classification_prefix_tokens(value: Any) -> list[str]:
    tokens = set()
    for scheme, code in _iter_classification_pairs(value):
        for k in range(1, len(code) + 1):
            tokens.add(f"{scheme}|{code[:k]}")
    return sorted(tokens)


def _build_nodes(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for flow in flows:
        node = dict(flow)
        node["id"] = int(flow["position"])
        node.setdefault("bio_suppliers", [])
        node.setdefault("tech_suppliers", [])
        node["_categories_path"] = _normalize_categories(node.get("categories"))
        node["_class_prefixes"] = _classification_prefix_tokens(
            node.get("classifications")
        )
        nodes.append(node)
    return nodes


def build_technosphere_nodes(
    technosphere_flows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build RETE node facts for technosphere flows.
    """
    return _build_nodes(technosphere_flows)


def build_biosphere_nodes(biosphere_flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build RETE node facts for biosphere flows.
    """
    return _build_nodes(biosphere_flows)


def attach_suppliers(
    nodes: list[dict[str, Any]],
    edges: set[tuple[int, int]],
    *,
    slot_name: str,
) -> None:
    """
    Mutate node facts to include supplier ids for each consumer node.
    """
    suppliers_by_consumer: dict[int, list[int]] = defaultdict(list)
    for s, c in edges:
        suppliers_by_consumer[int(c)].append(int(s))
    by_id = {int(n["id"]): n for n in nodes}
    for consumer_id, suppliers in suppliers_by_consumer.items():
        if consumer_id in by_id:
            by_id[consumer_id][slot_name] = suppliers
