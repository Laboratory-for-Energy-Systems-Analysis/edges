from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import time

import bw2data
import bw2io

from edges import EdgeLCIA

AWARE_METHOD = ("AWARE 2.0", "Country", "all", "yearly")
DEFAULT_PROJECT = "ecoinvent-3.12-cutoff"
DEFAULT_OUTPUT = Path("dev/profiling/outputs/aware_clips.prof")


def ensure_h2_pem(project_name: str) -> None:
    bw2data.projects.set_current(project_name)
    print(f"Project: {project_name}")

    if "h2_pem" in bw2data.databases:
        return

    lci = bw2io.ExcelImporter("lci-hydrogen-electrolysis-ei310.xlsx")
    lci.apply_strategies()
    lci.match_database(fields=["name", "reference product", "location"])
    lci.match_database(project_name, fields=["name", "reference product", "location"])
    lci.match_database("biosphere3", fields=["name", "categories"])
    lci.statistics()
    lci.drop_unlinked(i_am_reckless=True)
    if len(list(lci.unlinked)) == 0:
        lci.write_database()


def pick_activity():
    return [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ][0]


def run_stage(activity, stage: str) -> tuple[EdgeLCIA, float]:
    t0 = time.time()
    lca = EdgeLCIA({activity: 1}, AWARE_METHOD, matcher_backend="clips")

    if stage == "map_exchanges":
        lca.lci()
        lca.map_exchanges()
    elif stage == "apply_strategies":
        lca.apply_strategies()
    else:
        lca.apply_strategies()
        lca.evaluate_cfs()
        lca.lcia()

    elapsed = time.time() - t0
    return lca, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile AWARE with CLIPSpy backend only."
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Brightway project name (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--stage",
        choices=["map_exchanges", "apply_strategies", "full"],
        default="full",
        help="What to profile (default: full)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to output .prof file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import clips  # noqa: F401
    except Exception:
        raise RuntimeError("clipspy is not available in this environment.")

    ensure_h2_pem(args.project)
    activity = pick_activity()

    prof = cProfile.Profile()
    prof.enable()
    lca, elapsed = run_stage(activity, args.stage)
    prof.disable()
    prof.dump_stats(str(out))

    print(f"Wrote profile: {out}")
    print(f"Stage: {args.stage}")
    print(f"Elapsed (s): {elapsed:.3f}")
    if args.stage == "full":
        print(f"Score: {float(lca.score)}")
        print(f"Characterization sum: {float(lca.characterization_matrix.sum())}")
        print(
            f"Characterized inventory sum: {float(lca.characterized_inventory.sum())}"
        )
    print(f"Mapped CF entries: {len(lca.cfs_mapping)}")


if __name__ == "__main__":
    main()
