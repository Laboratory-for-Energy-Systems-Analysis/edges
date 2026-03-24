from __future__ import annotations

from collections import defaultdict
from typing import Any

from .utils import make_hashable

SUPPORTED_CLIPS_SIDE_FIELDS = frozenset(
    {
        "matrix",
        "operator",
        "name",
        "reference product",
        "location",
        "unit",
        "categories",
        "classifications",
        "excludes",
        "__compiled_match__",
    }
)


def _normalize_wildcard(value: Any) -> Any:
    if value == "__ANY__":
        return None
    return value


def _normalize_categories(value: Any) -> tuple[Any, ...] | Any | None:
    value = _normalize_wildcard(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _iter_classification_pairs(value: Any) -> list[tuple[str, str]]:
    if not value:
        return []

    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for scheme, codes in value.items():
            seq = codes if isinstance(codes, (list, tuple, set)) else [codes]
            for code in seq:
                norm_code = str(code).split(":", 1)[0].strip()
                if norm_code:
                    pairs.append((str(scheme).lower().strip(), norm_code))
        return pairs

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                scheme, codes = item
                seq = codes if isinstance(codes, (list, tuple, set)) else [codes]
                for code in seq:
                    norm_code = str(code).split(":", 1)[0].strip()
                    if norm_code:
                        pairs.append((str(scheme).lower().strip(), norm_code))
        return pairs

    return pairs


def _normalize_classifications(value: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(set(_iter_classification_pairs(_normalize_wildcard(value)))))


def _normalize_excludes(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(v) for v in value}))
    return (str(value),)


def normalize_rule_side_for_clips_signature(
    side: dict[str, Any] | None, *, default_matrix: str = "technosphere"
) -> dict[str, Any]:
    side = side or {}
    return {
        "matrix": str(side.get("matrix", default_matrix)).strip().lower(),
        "operator": str(side.get("operator", "equals")).strip().lower(),
        "name": _normalize_wildcard(side.get("name")),
        "reference product": _normalize_wildcard(
            side.get("reference product", side.get("reference_product"))
        ),
        "location": _normalize_wildcard(side.get("location")),
        "unit": _normalize_wildcard(side.get("unit")),
        "categories": _normalize_categories(side.get("categories")),
        "classifications": _normalize_classifications(side.get("classifications")),
        "excludes": _normalize_excludes(side.get("excludes")),
    }


def normalize_rule_for_clips_signature(cf: dict[str, Any]) -> dict[str, Any]:
    return {
        "supplier": normalize_rule_side_for_clips_signature(cf.get("supplier")),
        "consumer": normalize_rule_side_for_clips_signature(cf.get("consumer")),
    }


def find_duplicate_clips_signatures(
    raw_cfs_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[int, dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )

    for index, cf in enumerate(raw_cfs_data):
        signature = normalize_rule_for_clips_signature(cf)
        grouped[make_hashable(signature)].append((index, cf, signature))

    duplicates: list[dict[str, Any]] = []
    for matches in grouped.values():
        if len(matches) < 2:
            continue
        first_signature = matches[0][2]
        duplicates.append(
            {
                "count": len(matches),
                "indices": tuple(index for index, _, _ in matches),
                "values": tuple(cf.get("value") for _, cf, _ in matches),
                "supplier": first_signature["supplier"],
                "consumer": first_signature["consumer"],
            }
        )

    duplicates.sort(key=lambda group: group["indices"])
    return duplicates
