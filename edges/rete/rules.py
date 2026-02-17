from __future__ import annotations

from typing import Any


def compile_rules(raw_cfs_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a normalized RETE rule list from ``EdgeLCIA.raw_cfs_data``.

    Phase 1 scaffold: returns the same structures with stable rule ids.
    """
    out = []
    for rule_id, cf in enumerate(raw_cfs_data):
        c = dict(cf)
        c["id"] = rule_id
        out.append(c)
    return out
