from __future__ import annotations

from typing import Any


def compile_rules(
    raw_cfs_data: list[dict[str, Any]],
    *,
    supplier_names_by_matrix: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a normalized RETE rule list from ``EdgeLCIA.raw_cfs_data``.

    Keep original rule ids and order. Exact supplier names absent from the
    current nodes cannot match, including in the location fallback cascade.
    Skip compiling these rules for large vocabularies such as GLAM; retain
    wildcard and non-equality rules. Raw CFs and geographic indexes stay intact.
    """
    out = []
    for rule_id, cf in enumerate(raw_cfs_data):
        supplier = cf.get("supplier") or {}
        name = supplier.get("name")
        if (
            supplier_names_by_matrix is not None
            and supplier.get("operator", "equals") == "equals"
            and name not in (None, "__ANY__")
            and str(name).strip()
        ):
            matrix = str(supplier.get("matrix", "technosphere")).strip().lower()
            # Match the whitespace normalization in ClipsEngine._safe_lexeme.
            if str(name).strip() not in supplier_names_by_matrix.get(matrix, set()):
                continue
        c = dict(cf)
        c["id"] = rule_id
        out.append(c)
    return out
