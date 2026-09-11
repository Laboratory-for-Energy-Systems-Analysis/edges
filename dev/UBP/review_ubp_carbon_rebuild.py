"""Review a source-derived UBP carbon-alias rebuild against the bundled baseline.

Run before replacing the bundled JSON. This fails on any change beyond the three
reviewed aliases and records the precise new audit hashes for explicit review.
"""

from pathlib import Path
import hashlib
import json

import build_ubp_methods as b

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/UBP/simapro-method-audit/carbon-rebuild-review.json"
ALIASES = {
    "Carbon dioxide, land transformation": "Carbon dioxide, from soil or biomass stock",
    "Methane, biogenic": "Methane, non-fossil",
    "Methane, land transformation": "Methane, from soil or biomass stock",
}
BASELINE_SOURCE_ONLY_GROUPS_SHA256 = (
    "a6c4a7074335c16b03bfe64b7bf2e4ff4708bac2aabb90694601dd05180083dc"
)


def digest(value, **kwargs):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), **kwargs).encode()
    ).hexdigest()


def main():
    parsed = b.parse_simapro(ROOT / "dev/UBP/Ecological-Scarcity-21.CSV")
    flows, activities, locations, noise, _, targets = b.load_brightway_targets()
    reference = b.load_ecoinvent_312_reference_signatures()
    categories, total, dispositions, _ = b.build_methods(
        parsed, flows, locations, noise
    )
    old_aliases = b.NAME_ALIASES.copy()
    try:
        for name in ALIASES:
            del b.NAME_ALIASES[name]
        baseline, baseline_total, baseline_dispositions, _ = b.build_methods(
            parsed, flows, locations, noise
        )
    finally:
        b.NAME_ALIASES.clear()
        b.NAME_ALIASES.update(old_aliases)
    additions = {}
    before_hashes = {}
    after_hashes = {}
    for category, payload in {**categories, "total": total}.items():
        filename = b.safe_category_filename(category)
        path = ROOT / "edges/data" / filename
        existing = json.loads(path.read_text())
        old = baseline_total if category == "total" else baseline[category]
        assert (
            existing == old or existing == payload
        ), f"Unexplained bundled drift: {category}"
        old_rows = {b.edge_signature(r): r for r in old["exchanges"]}
        new_rows = {b.edge_signature(r): r for r in payload["exchanges"]}
        assert old_rows.keys() <= new_rows.keys(), f"Removed rules: {category}"
        assert all(
            new_rows[k] == r for k, r in old_rows.items()
        ), f"Changed rules: {category}"
        added = [r for k, r in new_rows.items() if k not in old_rows]
        assert all(r["supplier"]["name"] in ALIASES.values() for r in added)
        if category not in {"Global warming", "total"}:
            assert not added, f"Other category changed: {category}"
        elif not added:
            raise AssertionError(f"Missing carbon additions: {category}")
        if added:
            additions[category] = added
        before_hashes[filename] = hashlib.sha256(
            b.render_json(old).encode()
        ).hexdigest()
        after_hashes[filename] = hashlib.sha256(
            b.render_json(payload).encode()
        ).hexdigest()
    source_only = b.summarize_source_only_rows(parsed, dispositions)
    baseline_groups = b.summarize_source_only_rows(parsed, baseline_dispositions)
    assert (
        digest(baseline_groups, sort_keys=True, ensure_ascii=False)
        == BASELINE_SOURCE_ONLY_GROUPS_SHA256
    )
    removed_groups = [g for g in baseline_groups if g not in source_only]
    assert {g["base_name"] for g in removed_groups} == set(ALIASES)
    assert all(
        g["category"] == "Global warming" and g["rows"] == 1 for g in removed_groups
    )
    assert len(source_only) == len(baseline_groups) - 3
    gaps = []
    for category, expected in sorted(reference.items()):
        actual = {
            (
                r["supplier"]["name"],
                tuple(r["supplier"].get("categories", ())),
                r["supplier"]["unit"],
            )
            for r in categories[category]["exchanges"]
            if r["supplier"]["matrix"] == "biosphere"
        }
        gaps.extend((category, *signature) for signature in sorted(expected - actual))
    result = {
        "source_sha256": parsed.source_sha256,
        "source_carbon_rows": [
            vars(r)
            for r in parsed.rows
            if r.category == "Global warming" and r.substance in ALIASES
        ],
        "targets": targets,
        "baseline_reproduced_without_new_aliases": True,
        "existing_rules_unchanged": True,
        "added_rules": additions,
        "removed_source_only_groups": removed_groups,
        "new_source_only_groups_sha256": digest(
            source_only, sort_keys=True, ensure_ascii=False
        ),
        "new_reference_gaps_sha256": digest(gaps),
        "new_reference_gaps_count": len(gaps),
        "before_method_sha256": before_hashes,
        "after_method_sha256": after_hashes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(b.render_json(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in {
                    "baseline_reproduced_without_new_aliases",
                    "existing_rules_unchanged",
                    "new_source_only_groups_sha256",
                    "new_reference_gaps_sha256",
                    "new_reference_gaps_count",
                }
            },
            indent=2,
        )
    )
    print("Added rules:", {k: len(v) for k, v in additions.items()})


if __name__ == "__main__":
    main()
