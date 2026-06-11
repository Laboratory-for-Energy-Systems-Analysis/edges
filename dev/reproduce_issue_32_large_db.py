#!/usr/bin/env python
"""Reproduce GitHub issue #32: SQLite variable limit in get_activities.

The original issue used a 50k-activity fake Brightway database and failed in
``edges.utils.get_activities`` on SQLite builds capped at 32766 variables. The
local ``edges`` conda environment may have a higher compiled cap, so this script
can lower the active Peewee SQLite connection limit to match the reported cap.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import time
import traceback

import bw2data as bd

import edges

PROJECT_NAME = "edges_issue_32_large_db"
DB_NAME = "fake_large_db"
FLOW_DB_NAME = "fake_large_biosphere"
FLOW_CODE = "issue_32_elementary_flow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT_NAME)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--flow-database", default=FLOW_DB_NAME)
    parser.add_argument("--activities", type=int, default=50_000)
    parser.add_argument("--exchanges-per-activity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Overwrite the fake database even if it already exists.",
    )
    parser.add_argument(
        "--sqlite-variable-limit",
        type=int,
        default=32_766,
        help=(
            "Lower the active SQLite variable limit before EdgeLCIA.lci(); "
            "use 0 to keep the compiled/default limit."
        ),
    )
    return parser.parse_args()


def compile_option_value(name: str) -> str | None:
    with sqlite3.connect(":memory:") as con:
        for (option,) in con.execute("pragma compile_options"):
            if option.startswith(f"{name}="):
                return option.split("=", 1)[1]
    return None


def get_activity_dataset_model():
    if hasattr(bd, "backends"):
        try:
            return bd.backends.ActivityDataset
        except AttributeError:
            pass

    try:
        from bw2data.backends import ActivityDataset
    except ImportError:
        from bw2data.backends.peewee import ActivityDataset

    return ActivityDataset


def set_sqlite_variable_limit(limit: int) -> None:
    if not limit:
        return

    activity_dataset = get_activity_dataset_model()
    peewee_db = activity_dataset._meta.database
    peewee_db.connect(reuse_if_open=True)
    conn = peewee_db.connection()

    if not hasattr(conn, "setlimit"):
        print("SQLite connection has no setlimit(); using compiled/default limit.")
        return

    previous = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    current = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    print(
        "SQLite variable limit for ActivityDataset connection: "
        f"{previous} -> {current}"
    )


def node_for_key(key: tuple[str, str]):
    database, code = key
    if hasattr(bd, "get_node"):
        return bd.get_node(database=database, code=code)
    if hasattr(bd, "get_activity"):
        return bd.get_activity(key)
    return bd.Database(database).get(code)


def ensure_flow_database(flow_database: str) -> tuple[str, str]:
    flow_key = (flow_database, FLOW_CODE)
    if flow_database in bd.databases:
        return flow_key

    print(f"Writing flow database {flow_database!r}.")
    bd.Database(flow_database).write(
        {
            flow_key: {
                "name": "Issue 32 elementary flow",
                "unit": "kg",
                "categories": ("air",),
                "type": "emission",
            }
        }
    )
    return flow_key


def random_other_indices(
    rng: random.Random, current: int, total: int, count: int
) -> list[int]:
    """Sample distinct activity indices without building a 50k candidate list."""
    if total <= 1 or count <= 0:
        return []

    count = min(count, total - 1)
    selected: set[int] = set()
    while len(selected) < count:
        index = rng.randrange(total - 1)
        if index >= current:
            index += 1
        selected.add(index)
    return sorted(selected)


def build_fake_database(
    database: str,
    flow_key: tuple[str, str],
    activities: int,
    exchanges_per_activity: int,
    seed: int,
    progress_every: int,
) -> None:
    rng = random.Random(seed)
    keys = [(database, f"act_{i}") for i in range(activities)]
    data = {}

    start = time.perf_counter()
    for i, key in enumerate(keys):
        exchanges = [
            {
                "input": key,
                "amount": 1.0,
                "type": "production",
                "unit": "unit",
            },
            {
                "input": flow_key,
                "amount": round(rng.uniform(0.1, 10.0), 4),
                "type": "biosphere",
                "unit": "kg",
            },
        ]

        for input_index in random_other_indices(
            rng, i, activities, exchanges_per_activity
        ):
            exchanges.append(
                {
                    "input": keys[input_index],
                    "amount": round(rng.uniform(0.01, 1.0), 4),
                    "type": "technosphere",
                    "unit": "unit",
                }
            )

        data[key] = {
            "name": f"Activity {i}",
            "reference product": f"Product {i}",
            "unit": "unit",
            "location": "GLO",
            "categories": ("test",),
            "exchanges": exchanges,
        }

        if progress_every and i and i % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i:,}/{activities:,} activities prepared ({elapsed:.1f}s).")

    print(f"Writing database {database!r} with {activities:,} activities.")
    bd.Database(database).write(data)


def ensure_fake_database(
    database: str,
    flow_key: tuple[str, str],
    activities: int,
    exchanges_per_activity: int,
    seed: int,
    progress_every: int,
    rebuild: bool,
) -> None:
    if database in bd.databases and not rebuild:
        existing = len(bd.Database(database))
        if existing == activities:
            print(f"Reusing existing database {database!r} ({existing:,} activities).")
            return
        raise RuntimeError(
            f"Database {database!r} already exists with {existing:,} activities; "
            f"expected {activities:,}. Re-run with --rebuild to overwrite it."
        )

    build_fake_database(
        database=database,
        flow_key=flow_key,
        activities=activities,
        exchanges_per_activity=exchanges_per_activity,
        seed=seed,
        progress_every=progress_every,
    )


def inline_method() -> dict:
    return {
        "name": "Issue 32 large database reproducer",
        "version": "0.1",
        "unit": "dimensionless",
        "exchanges": [
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "Issue 32 elementary flow",
                    "unit": "kg",
                    "categories": ("air",),
                },
                "consumer": {
                    "matrix": "technosphere",
                    "name": "Activity 0",
                    "reference product": "Product 0",
                    "unit": "unit",
                    "location": "GLO",
                },
                "value": 1.0,
            }
        ],
    }


def main() -> int:
    args = parse_args()

    print(f"Python sqlite version: {sqlite3.sqlite_version}")
    print(
        "Compiled SQLite MAX_VARIABLE_NUMBER: "
        f"{compile_option_value('MAX_VARIABLE_NUMBER') or 'unknown'}"
    )

    bd.projects.set_current(args.project)
    print(f"Brightway project: {bd.projects.current}")

    flow_key = ensure_flow_database(args.flow_database)
    ensure_fake_database(
        database=args.database,
        flow_key=flow_key,
        activities=args.activities,
        exchanges_per_activity=args.exchanges_per_activity,
        seed=args.seed,
        progress_every=args.progress_every,
        rebuild=args.rebuild,
    )

    activity = node_for_key((args.database, "act_0"))
    edge_lca = edges.EdgeLCIA(demand={activity: 1}, method=inline_method())

    set_sqlite_variable_limit(args.sqlite_variable_limit)

    print(
        "Running EdgeLCIA.lci() for "
        f"{args.activities:,} activities and "
        f"{args.exchanges_per_activity} technosphere inputs/activity."
    )
    try:
        edge_lca.lci()
    except Exception as exc:
        print(f"Reproduced failure: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    print("EdgeLCIA.lci() completed without reproducing the SQL variable error.")
    print(f"Activity dict size: {len(edge_lca.lca.activity_dict):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
