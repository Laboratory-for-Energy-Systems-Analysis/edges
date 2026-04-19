#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

DEFAULT_PROJECT = "ecoinvent-3.12-cutoff"
DEFAULT_DATABASE = "ecoinvent-3.12-cutoff"
DEFAULT_SCOPE = "overall"
DEFAULT_OUTPUT_DIR = Path("dev/outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the IBIF roads method on a random activity from an ecoinvent "
            "database and export the resulting characterization-factor table."
        )
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Brightway project name. Default: {DEFAULT_PROJECT}",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"Brightway database name. Default: {DEFAULT_DATABASE}",
    )
    parser.add_argument(
        "--scope",
        choices=("overall", "vertebrates"),
        default=DEFAULT_SCOPE,
        help=f"IBIF road method scope. Default: {DEFAULT_SCOPE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random activity sampling. Default: 42",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="Demand amount for the sampled activity. Default: 1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV export. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of CF-table rows to print. Default: 20",
    )
    parser.set_defaults(include_unmatched=True)
    parser.add_argument(
        "--include-unmatched",
        dest="include_unmatched",
        action="store_true",
        help="Include unmatched inventory rows in the exported CF table.",
    )
    parser.add_argument(
        "--matched-only",
        dest="include_unmatched",
        action="store_false",
        help="Only export matched characterization rows.",
    )
    return parser


def install_optional_dependency_stubs() -> None:
    try:
        import highspy  # noqa: F401
    except ModuleNotFoundError:
        import types

        highspy = types.ModuleType("highspy")
        highspy.Highs = object
        sys.modules["highspy"] = highspy

    try:
        import plotly  # noqa: F401
    except ModuleNotFoundError:
        import types

        plotly = types.ModuleType("plotly")
        graph_objects = types.ModuleType("plotly.graph_objects")
        graph_objects.Figure = object
        io = types.ModuleType("plotly.io")
        io.to_html = lambda *args, **kwargs: ""
        plotly.graph_objects = graph_objects
        plotly.io = io
        sys.modules["plotly"] = plotly
        sys.modules["plotly.graph_objects"] = graph_objects
        sys.modules["plotly.io"] = io


def choose_random_activity(database, seed: int):
    activities = [
        activity
        for activity in database
        if "recycled" not in activity.get("name", "").lower()
    ]
    if not activities:
        raise RuntimeError(
            f"Database '{database.name}' contains no activities after filtering."
        )
    return random.Random(seed).choice(activities)


def describe_activity(activity) -> str:
    return (
        f"name={activity.get('name')!r}, "
        f"reference product={activity.get('reference product')!r}, "
        f"location={activity.get('location')!r}, "
        f"unit={activity.get('unit')!r}, "
        f"code={activity.get('code')!r}"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    import bw2data as bd
    import pandas as pd

    install_optional_dependency_stubs()

    from edges import EdgeLCIA

    available_projects = {project.name for project in bd.projects}
    if args.project not in available_projects:
        raise RuntimeError(
            f"Project '{args.project}' not found. "
            f"Available projects: {sorted(available_projects)}"
        )

    bd.projects.set_current(args.project)

    if args.database not in bd.databases:
        raise RuntimeError(
            f"Database '{args.database}' not found in project '{args.project}'. "
            f"Available databases: {sorted(bd.databases)}"
        )

    database = bd.Database(args.database)
    activity = choose_random_activity(database, seed=args.seed)
    method = ("IBIF", "biodiversity", "roads", args.scope)

    print(f"Project: {args.project}")
    print(f"Database: {args.database}")
    print(f"Method: {method}")
    print(f"Seed: {args.seed}")
    print("Sampled activity:")
    print(describe_activity(activity))

    lcia = EdgeLCIA(demand={activity: args.amount}, method=method)
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

    df = lcia.generate_cf_table(
        include_unmatched=args.include_unmatched,
        split_aggregate_consumers=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        args.output_dir / f"ibif_roads_{args.scope}_cf_table_seed_{args.seed}.csv"
    )
    df.to_csv(csv_path, index=False)

    print(f"Score: {lcia.score}")
    print(f"CF table rows: {len(df)}")
    print(f"CF table CSV: {csv_path.resolve()}")

    preview = df
    if "impact" in preview.columns and not preview["impact"].isna().all():
        preview = preview.assign(_abs_impact=preview["impact"].abs()).sort_values(
            "_abs_impact", ascending=False
        )
        preview = preview.drop(columns="_abs_impact")
    elif "amount" in preview.columns and not preview.empty:
        preview = preview.assign(_abs_amount=preview["amount"].abs()).sort_values(
            "_abs_amount", ascending=False
        )
        preview = preview.drop(columns="_abs_amount")

    preview = preview.head(args.preview_rows)
    if preview.empty:
        print("CF table is empty for the sampled activity.")
        return 0

    print()
    print(f"Preview of first {len(preview)} rows:")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        80,
    ):
        print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
