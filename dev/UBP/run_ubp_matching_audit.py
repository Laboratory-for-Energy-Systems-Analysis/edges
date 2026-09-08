#!/usr/bin/env python3
"""Run the bundled UBP method against ecoinvent 3.10, 3.11, and 3.12.

The audit uses the same exact functional unit in each cutoff project and checks
the mixed biosphere/technosphere matching invariants. A separate Noise run
proves that the technosphere component of the total is exactly the Noise
category score, and a passenger-rail probe exercises the sixth Noise family.
Complete CF tables containing ecoinvent exchange details are optional local
artifacts and are never required for the compact audit summary.

This script requires local licensed ecoinvent projects; it does not create or
modify Brightway databases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bw2calc
import bw2data as bd
import pandas as pd

from edges import EdgeLCIA, __version__ as edges_version


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "edges" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "UBP"

TARGETS = {
    "3.10": ("ecoinvent-3.10-cutoff", "ecoinvent-3.10-cutoff"),
    "3.11": ("ecoinvent-3.11-cutoff", "ecoinvent-3.11-cutoff"),
    "3.12": ("ecoinvent-3.12-cutoff", "ecoinvent-3.12-cutoff"),
}

ACTIVITY_NAME = (
    "market for transport, freight, lorry 28 metric ton, "
    "fatty acid methyl ester 100%"
)
REFERENCE_PRODUCT = ACTIVITY_NAME.removeprefix("market for ")
LOCATION = "CH"
AMOUNT = 1.0
UNIT = "ton kilometer"

TOTAL_METHOD = ("Ecological Scarcity 2021", "total", "UBP")
NOISE_METHOD = ("Ecological Scarcity 2021", "Noise", "UBP")
EXPECTED_DIRECT_NOISE_CF = 180.0 / 5.79
EXPECTED_RAIL_PASSENGER_CF = 8.0
EXPECTED_NOISE_FAMILIES = {
    "aircraft freight",
    "aircraft passenger",
    "lorry",
    "passenger car",
    "rail freight",
    "rail passenger",
}
RAIL_PROBES = {
    "3.10": (
        "market for transport, passenger train",
        "transport, passenger train",
        "GLO",
        "person kilometer",
    ),
    "3.11": (
        "market for transport, passenger, train, fleet average",
        "transport, passenger, train, fleet average",
        "GLO",
        "person kilometer",
    ),
    "3.12": (
        "market for transport, passenger, train, fleet average",
        "transport, passenger, train, fleet average",
        "GLO",
        "person kilometer",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(TARGETS),
        required=True,
        help=(
            "ecoinvent versions to audit; pass versions compatible with the "
            "active Brightway environment"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for CSV and JSON audit artifacts",
    )
    parser.add_argument(
        "--full-cf-table",
        action="store_true",
        help=(
            "write gzip-compressed total CF tables including unmatched "
            "ecoinvent exchanges (local/licensed artifacts)"
        ),
    )
    return parser.parse_args()


def load_method_payload(category: str) -> dict[str, Any]:
    path = DATA_DIR / f"Ecological Scarcity 2021_{category}_UBP.json"
    return json.loads(path.read_text(encoding="utf-8"))


def find_exact_activity(
    database: bd.Database,
    *,
    name: str,
    product: str,
    location: str,
    unit: str,
) -> Any:
    matches = [
        activity
        for activity in database
        if activity.get("name") == name
        and activity.get("reference product") == product
        and activity.get("location") == location
        and activity.get("unit") == unit
    ]
    if len(matches) != 1:
        details = [
            (
                activity.get("name"),
                activity.get("reference product"),
                activity.get("location"),
                activity.get("unit"),
                activity.key,
            )
            for activity in matches[:20]
        ]
        raise ValueError(
            "Expected exactly one audit activity for "
            f"{(name, product, location, unit)!r}; "
            f"found {len(matches)}: {details}"
        )
    return matches[0]


def find_activity(database: bd.Database) -> Any:
    return find_exact_activity(
        database,
        name=ACTIVITY_NAME,
        product=REFERENCE_PRODUCT,
        location=LOCATION,
        unit=UNIT,
    )


def node_signature(node: Any, matrix: str) -> tuple[Any, ...]:
    if matrix == "biosphere":
        return (
            node.get("name"),
            tuple(node.get("categories") or ()),
            node.get("unit"),
        )
    return (
        node.get("name"),
        node.get("reference product"),
        node.get("unit"),
    )


def noise_family(reference_product: str, unit: str) -> str | None:
    product = reference_product.lower()
    if unit == "ton kilometer" and product.startswith("transport, freight, aircraft"):
        return "aircraft freight"
    if unit == "person kilometer" and product.startswith(
        ("transport, passenger aircraft", "transport, passenger, aircraft")
    ):
        return "aircraft passenger"
    if unit == "ton kilometer" and product.startswith(
        ("transport, freight train", "transport, freight, train")
    ):
        return "rail freight"
    if unit == "person kilometer" and product.startswith(
        ("transport, passenger train", "transport, passenger, train")
    ):
        return "rail passenger"
    if unit == "kilometer" and product.startswith(
        ("transport, passenger car", "transport, passenger, car")
    ):
        return "passenger car"
    if unit == "ton kilometer" and product.startswith("transport, freight, lorry"):
        return "lorry"
    return None


def matched_noise_family_counts(lcia: EdgeLCIA) -> Counter[str]:
    counts: Counter[str] = Counter()
    for supplier_index, _ in lcia.processed_technosphere_edges:
        supplier = lcia.position_to_technosphere_flows_lookup[supplier_index]
        family = noise_family(
            supplier.get("reference product", ""), supplier.get("unit", "")
        )
        if family is None:
            raise AssertionError(
                "A characterized Noise supplier is outside the six reviewed "
                f"transport families: {supplier!r}"
            )
        counts[family] += 1
    return counts


def rule_signature(rule: dict[str, Any], matrix: str) -> tuple[Any, ...]:
    supplier = rule["supplier"]
    if matrix == "biosphere":
        return (
            supplier.get("name"),
            tuple(supplier.get("categories") or ()),
            supplier.get("unit"),
        )
    return (
        supplier.get("name"),
        supplier.get("reference product"),
        supplier.get("unit"),
    )


def expected_direct_positions(
    lcia: EdgeLCIA,
    payload: dict[str, Any],
    matrix: str,
) -> set[tuple[int, int]]:
    """Return edges guaranteed to match without geographic fallback.

    Supplier metadata must match a method rule exactly. A rule is directly
    applicable when its consumer has no location constraint or its location is
    exactly the inventory consumer location.
    """

    rules_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for rule in payload["exchanges"]:
        if rule["supplier"].get("matrix") != matrix:
            continue
        rules_by_signature.setdefault(rule_signature(rule, matrix), []).append(rule)

    edges = lcia.biosphere_edges if matrix == "biosphere" else lcia.technosphere_edges
    biosphere_lookup = {
        flow["position"]: {
            key: value for key, value in flow.items() if key != "position"
        }
        for flow in (lcia.biosphere_flows or ())
    }
    technosphere_lookup = lcia.position_to_technosphere_flows_lookup
    expected: set[tuple[int, int]] = set()
    for supplier_index, consumer_index in edges:
        supplier = (
            biosphere_lookup[supplier_index]
            if matrix == "biosphere"
            else technosphere_lookup[supplier_index]
        )
        consumer = technosphere_lookup[consumer_index]
        candidate_rules = rules_by_signature.get(node_signature(supplier, matrix), ())
        if any(
            rule["consumer"].get("location") in (None, consumer.get("location"))
            for rule in candidate_rules
        ):
            expected.add((supplier_index, consumer_index))
    return expected


def assert_unique_positions(positions: list[tuple[int, int]], label: str) -> None:
    duplicates = [
        position for position, count in Counter(positions).items() if count > 1
    ]
    if duplicates:
        raise AssertionError(
            f"Duplicate characterized {label} positions: {duplicates[:10]}"
        )


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"Expected a finite numeric result, got {value!r}")
    return result


def run_method(
    activity: Any, method: tuple[str, ...]
) -> tuple[EdgeLCIA, list[dict[str, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lcia = EdgeLCIA(demand={activity: AMOUNT}, method=method)
        lcia.lci()
        lcia.apply_strategies()
        lcia.evaluate_cfs()
        lcia.lcia()
    finite_float(lcia.score)
    warning_rows = sorted(
        {(item.category.__name__, str(item.message)) for item in caught}
    )
    return lcia, [
        {"category": category, "message": message} for category, message in warning_rows
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    head_path = REPO_ROOT / ".git" / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref_path = REPO_ROOT / ".git" / head.removeprefix("ref: ")
    return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None


def version_string(value: Any) -> str:
    return ".".join(map(str, value)) if isinstance(value, tuple) else str(value)


def matrix_summary(
    version: str,
    project: str,
    database: str,
    lcia: EdgeLCIA,
    table: pd.DataFrame,
    matrix: str,
    expected_direct: set[tuple[int, int]],
) -> dict[str, Any]:
    inventory_edges = (
        set(lcia.biosphere_edges)
        if matrix == "biosphere"
        else set(lcia.technosphere_edges)
    )
    processed = (
        list(lcia.processed_biosphere_edges)
        if matrix == "biosphere"
        else list(lcia.processed_technosphere_edges)
    )
    unmatched = (
        set(lcia.unprocessed_biosphere_edges)
        if matrix == "biosphere"
        else set(lcia.unprocessed_technosphere_edges)
    )
    processed_set = set(processed)

    assert_unique_positions(processed, matrix)
    if processed_set & unmatched:
        raise AssertionError(f"{matrix}: processed and unmatched positions overlap")
    if processed_set | unmatched != inventory_edges:
        raise AssertionError(f"{matrix}: processed/unmatched partition is incomplete")
    missed_expected = expected_direct - processed_set
    if missed_expected:
        raise AssertionError(
            f"{matrix}: {len(missed_expected)} directly eligible exchange(s) were not matched"
        )

    matrix_rows = table[table["supplier matrix"] == matrix]
    matched_rows = matrix_rows[matrix_rows["CF"].notna()]
    unmatched_rows = matrix_rows[matrix_rows["CF"].isna()]
    if len(matched_rows) != len(processed_set):
        raise AssertionError(
            f"{matrix}: CF table has {len(matched_rows)} matched rows, "
            f"but runtime reports {len(processed_set)}"
        )
    if len(unmatched_rows) != len(unmatched):
        raise AssertionError(
            f"{matrix}: CF table has {len(unmatched_rows)} unmatched rows, "
            f"but runtime reports {len(unmatched)}"
        )

    impact_sum = finite_float(matched_rows["impact"].sum())
    matrix_score = finite_float(lcia.score_by_matrix.get(matrix, 0.0))
    if not math.isclose(impact_sum, matrix_score, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(
            f"{matrix}: CF-table impacts ({impact_sum}) do not reconcile with "
            f"the matrix score ({matrix_score})"
        )

    return {
        "version": version,
        "project": project,
        "database": database,
        "matrix": matrix,
        "inventory_edges": len(inventory_edges),
        "matched_edges": len(processed_set),
        "unmatched_edges": len(unmatched),
        "directly_eligible_edges": len(expected_direct),
        "directly_eligible_missed": len(missed_expected),
        "matching_rate_all_inventory_percent": (
            100.0 * len(processed_set) / len(inventory_edges)
            if inventory_edges
            else 0.0
        ),
        "score_UBP": matrix_score,
        "cf_table_impact_sum_UBP": impact_sum,
    }


def run_version(
    version: str,
    output_dir: Path,
    total_payload: dict[str, Any],
    noise_payload: dict[str, Any],
    *,
    full_cf_table: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    project, database_name = TARGETS[version]
    bd.projects.set_current(project, writable=False, update=False)
    if database_name not in bd.databases:
        raise ValueError(
            f"Project {project!r} does not contain database {database_name!r}; "
            f"available databases: {sorted(bd.databases)}"
        )

    database = bd.Database(database_name)
    activity = find_activity(database)
    print(f"\n=== ecoinvent {version}: {activity.key} ===", flush=True)

    total, total_warnings = run_method(activity, TOTAL_METHOD)
    if total.duplicate_method_signature_groups:
        raise AssertionError(
            "Total method contains duplicate effective signatures: "
            f"{total.duplicate_method_signature_groups[:3]}"
        )
    if total.ignored_method_exchanges:
        raise AssertionError(
            f"Total method ignored {len(total.ignored_method_exchanges)} exchange rule(s)"
        )

    table = total.generate_cf_table(include_unmatched=True)
    table_path = None
    if full_cf_table:
        table_path = output_dir / f"ecoinvent-{version}-ubp-total-cf-table.csv.gz"
        table.to_csv(table_path, index=False, compression="gzip")

    expected_bio = expected_direct_positions(total, total_payload, "biosphere")
    expected_tech = expected_direct_positions(total, total_payload, "technosphere")
    rows = [
        matrix_summary(
            version,
            project,
            database_name,
            total,
            table,
            matrix,
            expected,
        )
        for matrix, expected in (
            ("biosphere", expected_bio),
            ("technosphere", expected_tech),
        )
    ]

    total_score = finite_float(total.score)
    component_sum = sum(finite_float(value) for value in total.score_by_matrix.values())
    if not math.isclose(total_score, component_sum, rel_tol=1e-12, abs_tol=1e-8):
        raise AssertionError(
            f"Total score {total_score} does not equal matrix components {component_sum}"
        )
    table_sum = finite_float(table.loc[table["CF"].notna(), "impact"].sum())
    if not math.isclose(total_score, table_sum, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(
            f"Total score {total_score} does not equal CF-table impact sum {table_sum}"
        )

    noise, noise_warnings = run_method(activity, NOISE_METHOD)
    noise_table = noise.generate_cf_table(include_unmatched=False)
    noise_path = output_dir / f"ecoinvent-{version}-ubp-noise-cf-table.csv"
    noise_table.to_csv(noise_path, index=False)

    expected_noise = expected_direct_positions(noise, noise_payload, "technosphere")
    processed_noise = list(noise.processed_technosphere_edges)
    assert_unique_positions(processed_noise, "Noise technosphere")
    missed_noise = expected_noise - set(processed_noise)
    if missed_noise:
        raise AssertionError(
            f"Noise: {len(missed_noise)} eligible transport exchange(s) were not matched"
        )
    if not processed_noise:
        raise AssertionError("Noise matched no technosphere transport exchanges")

    noise_score = finite_float(noise.score)
    total_tech_score = finite_float(total.score_by_matrix.get("technosphere", 0.0))
    if not math.isclose(noise_score, total_tech_score, rel_tol=1e-12, abs_tol=1e-8):
        raise AssertionError(
            f"Noise score {noise_score} != total technosphere component {total_tech_score}"
        )

    direct_rows = noise_table[
        (noise_table["supplier name"] == REFERENCE_PRODUCT)
        & (noise_table["consumer name"] == ACTIVITY_NAME)
    ]
    if len(direct_rows) != 1:
        raise AssertionError(
            f"Expected one direct lorry Noise row; found {len(direct_rows)}"
        )
    direct_row = direct_rows.iloc[0]
    if not math.isclose(
        finite_float(direct_row["CF"]),
        EXPECTED_DIRECT_NOISE_CF,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            f"Direct lorry Noise CF is {direct_row['CF']}, "
            f"expected {EXPECTED_DIRECT_NOISE_CF}"
        )

    rail_name, rail_product, rail_location, rail_unit = RAIL_PROBES[version]
    rail_activity = find_exact_activity(
        database,
        name=rail_name,
        product=rail_product,
        location=rail_location,
        unit=rail_unit,
    )
    rail_noise, rail_warnings = run_method(rail_activity, NOISE_METHOD)
    rail_table = rail_noise.generate_cf_table(include_unmatched=False)
    rail_direct_rows = rail_table[
        (rail_table["supplier reference product"] == rail_product)
        & (rail_table["consumer name"] == rail_name)
    ]
    if rail_direct_rows.empty:
        raise AssertionError(
            "Expected at least one direct passenger-rail Noise row; found none"
        )
    rail_cfs = {finite_float(value) for value in rail_direct_rows["CF"]}
    if any(
        not math.isclose(
            value,
            EXPECTED_RAIL_PASSENGER_CF,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in rail_cfs
    ):
        raise AssertionError(
            f"Direct passenger-rail Noise CFs are {sorted(rail_cfs)}, "
            f"expected {EXPECTED_RAIL_PASSENGER_CF}"
        )
    rail_direct_amount = finite_float(rail_direct_rows["amount"].sum())
    main_noise_families = matched_noise_family_counts(noise)
    rail_noise_families = matched_noise_family_counts(rail_noise)
    covered_noise_families = set(main_noise_families) | set(rail_noise_families)
    if covered_noise_families != EXPECTED_NOISE_FAMILIES:
        raise AssertionError(
            "Runtime probes do not cover all reviewed Noise families; "
            f"covered {sorted(covered_noise_families)}, "
            f"expected {sorted(EXPECTED_NOISE_FAMILIES)}"
        )

    strategy_match_counts = list(total.strategy_match_counts)
    if [row["strategy"] for row in strategy_match_counts] != list(
        total.applied_strategies
    ):
        raise AssertionError("Strategy match accounting is incomplete or out of order")
    if sum(row["new_biosphere_matches"] for row in strategy_match_counts) != len(
        total.processed_biosphere_edges
    ):
        raise AssertionError("Biosphere strategy-stage counts do not reconcile")
    if sum(row["new_technosphere_matches"] for row in strategy_match_counts) != len(
        total.processed_technosphere_edges
    ):
        raise AssertionError("Technosphere strategy-stage counts do not reconcile")

    total_method_path = DATA_DIR / "Ecological Scarcity 2021_total_UBP.json"
    noise_method_path = DATA_DIR / "Ecological Scarcity 2021_Noise_UBP.json"

    summary = {
        "version": version,
        "project": project,
        "database": database_name,
        "database_size": len(database),
        "functional_unit": {
            "amount": AMOUNT,
            "unit": UNIT,
            "activity": ACTIVITY_NAME,
            "reference_product": REFERENCE_PRODUCT,
            "location": LOCATION,
            "activity_key": list(activity.key),
        },
        "total_method": list(TOTAL_METHOD),
        "noise_method": list(NOISE_METHOD),
        "total_score_UBP": total_score,
        "biosphere_score_UBP": finite_float(
            total.score_by_matrix.get("biosphere", 0.0)
        ),
        "technosphere_noise_score_UBP": total_tech_score,
        "noise_only_score_UBP": noise_score,
        "total_cf_table_rows": len(table),
        "noise_cf_table_rows": len(noise_table),
        "noise_matched_edges": len(processed_noise),
        "noise_directly_eligible_edges": len(expected_noise),
        "noise_directly_eligible_missed": len(missed_noise),
        "noise_transport_family_matches": dict(sorted(main_noise_families.items())),
        "direct_lorry_cf_UBP_per_tkm": finite_float(direct_row["CF"]),
        "direct_lorry_inventory_amount_tkm": finite_float(direct_row["amount"]),
        "direct_lorry_impact_UBP": finite_float(direct_row["impact"]),
        "passenger_rail_probe": {
            "activity": rail_name,
            "reference_product": rail_product,
            "location": rail_location,
            "unit": rail_unit,
            "activity_key": list(rail_activity.key),
            "score_UBP": finite_float(rail_noise.score),
            "direct_edges": len(rail_direct_rows),
            "direct_cf_UBP_per_person_km": EXPECTED_RAIL_PASSENGER_CF,
            "direct_inventory_amount_person_km": rail_direct_amount,
            "noise_transport_family_matches": dict(sorted(rail_noise_families.items())),
        },
        "applied_strategies": list(total.applied_strategies),
        "strategy_match_counts": strategy_match_counts,
        "duplicate_method_signature_groups": len(
            total.duplicate_method_signature_groups
        ),
        "ignored_method_exchanges": len(total.ignored_method_exchanges),
        "runtime": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "edges": edges_version,
            "bw2data": version_string(bd.__version__),
            "bw2calc": version_string(bw2calc.__version__),
            "brightway_25_project": bool(getattr(bd.projects, "twofive", False)),
            "total_method_sha256": file_sha256(total_method_path),
            "noise_method_sha256": file_sha256(noise_method_path),
            "warnings": {
                "total": total_warnings,
                "noise": noise_warnings,
                "passenger_rail_noise": rail_warnings,
            },
        },
        "artifacts": {
            "total_cf_table": (
                str(table_path.relative_to(REPO_ROOT)) if table_path else None
            ),
            "noise_cf_table": str(noise_path.relative_to(REPO_ROOT)),
        },
        "checks": "passed",
    }
    print(
        f"ecoinvent {version}: total={total_score:.12g} UBP; "
        f"Noise={noise_score:.12g} UBP; "
        f"matched bio/tech={rows[0]['matched_edges']}/{rows[1]['matched_edges']}",
        flush=True,
    )
    return summary, rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_payload = load_method_payload("total")
    noise_payload = load_method_payload("Noise")
    json_path = output_dir / "ubp-run-audit.json"
    matrix_csv_path = output_dir / "ubp-exchange-matching-summary.csv"
    summaries = []
    matrix_rows = []
    if json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        summaries.extend(
            row
            for row in previous.get("results", [])
            if row.get("version") not in args.versions
        )
    if matrix_csv_path.exists():
        previous_matrix_rows = pd.read_csv(
            matrix_csv_path, dtype={"version": "string"}
        ).to_dict("records")
        matrix_rows.extend(
            row
            for row in previous_matrix_rows
            if str(row.get("version")) not in args.versions
        )
    for version in args.versions:
        summary, rows = run_version(
            version,
            output_dir,
            total_payload,
            noise_payload,
            full_cf_table=args.full_cf_table,
        )
        summaries.append(summary)
        matrix_rows.extend(rows)

    summaries.sort(key=lambda row: row["version"])
    matrix_rows.sort(key=lambda row: (str(row["version"]), row["matrix"]))
    pd.DataFrame(matrix_rows).to_csv(matrix_csv_path, index=False)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "edges_version": edges_version,
        "ecoinvent_versions": [row["version"] for row in summaries],
        "functional_unit": {
            "amount": AMOUNT,
            "unit": UNIT,
            "activity": ACTIVITY_NAME,
            "reference_product": REFERENCE_PRODUCT,
            "location": LOCATION,
        },
        "method_version": total_payload["version"],
        "method_unit": total_payload["unit"],
        "results": summaries,
        "matrix_summary_csv": str(matrix_csv_path.relative_to(REPO_ROOT)),
        "overall_checks": "passed",
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote audit summary to {json_path}", flush=True)


if __name__ == "__main__":
    main()
