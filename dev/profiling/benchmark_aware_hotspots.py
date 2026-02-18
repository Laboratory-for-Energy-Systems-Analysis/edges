from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import bw2data
import bw2io

from edges import EdgeLCIA

AWARE_METHOD = ("AWARE 2.0", "Country", "all", "yearly")
DEFAULT_PROJECT = "ecoinvent-3.12-cutoff"
HOTSPOTS = (
    "map_exchanges_clips",
    "map_dynamic_locations",
    "_compute_average_cf_cached",
    "compute_average_cf",
    "process_cf_list",
)


def ensure_h2_pem(project_name: str, excel_path: Path, reset: bool) -> None:
    bw2data.projects.set_current(project_name)
    if reset and "h2_pem" in bw2data.databases:
        del bw2data.databases["h2_pem"]

    if "h2_pem" in bw2data.databases:
        return

    lci = bw2io.ExcelImporter(str(excel_path))
    lci.apply_strategies()
    lci.match_database(fields=["name", "reference product", "location"])
    lci.match_database(project_name, fields=["name", "reference product", "location"])
    lci.match_database("biosphere3", fields=["name", "categories"])
    lci.drop_unlinked(i_am_reckless=True)
    if len(list(lci.unlinked)) == 0:
        lci.write_database()


def pick_activity():
    name = (
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, "
        "from offshore wind electricity"
    )
    return [a for a in bw2data.Database("h2_pem") if a["name"] == name][0]


def _extract_hotspots(stats: pstats.Stats) -> dict[str, float]:
    out = {k: 0.0 for k in HOTSPOTS}
    for (filename, _lineno, funcname), (
        _cc,
        _nc,
        _tt,
        ct,
        _callers,
    ) in stats.stats.items():
        if not filename.startswith("/Users/romain/GitHub/edges/edges/"):
            continue
        if funcname in out:
            out[funcname] += float(ct)
    return out


def _fmt(v: float) -> str:
    return f"{v:.3f}s"


def run_once(activity, run_id: int, profile_dir: Path) -> dict:
    prof = cProfile.Profile()
    lca = EdgeLCIA({activity: 1}, AWARE_METHOD, matcher_backend="clips")

    t0 = time.time()
    prof.enable()
    lca.apply_strategies()
    lca.evaluate_cfs()
    lca.lcia()
    prof.disable()
    elapsed = time.time() - t0

    prof_path = profile_dir / f"aware_hotspots_run{run_id:02d}.prof"
    prof.dump_stats(str(prof_path))

    ps = pstats.Stats(prof)
    hotspots = _extract_hotspots(ps)

    cf_calls = lca._cf_avg_cache_hits + lca._cf_avg_cache_misses
    cf_hit_rate = (lca._cf_avg_cache_hits / cf_calls * 100.0) if cf_calls else 0.0
    runtime_stats = dict(getattr(lca, "_cf_runtime_stats", {}) or {})

    return {
        "run_id": run_id,
        "elapsed_s": float(elapsed),
        "score": float(lca.score),
        "profile": str(prof_path),
        "hotspots": hotspots,
        "cf_cache_calls": int(cf_calls),
        "cf_cache_hits": int(lca._cf_avg_cache_hits),
        "cf_cache_misses": int(lca._cf_avg_cache_misses),
        "cf_cache_hit_rate": float(cf_hit_rate),
        "pair_match_hits": int(runtime_stats.get("pair_match_cache_hits", 0)),
        "pair_match_misses": int(runtime_stats.get("pair_match_cache_misses", 0)),
        "valid_pairs_hits": int(runtime_stats.get("valid_pairs_cache_hits", 0)),
        "valid_pairs_misses": int(runtime_stats.get("valid_pairs_cache_misses", 0)),
        "process_cf_list_calls": int(runtime_stats.get("process_cf_list_calls", 0)),
    }


def print_summary(results: list[dict]) -> None:
    print("\n== AWARE hotspot benchmark (CLIPSpy)")
    print(
        "run | elapsed | score | map_exchanges_clips | map_dynamic_locations | "
        "_compute_average_cf_cached | compute_average_cf | process_cf_list"
    )
    for r in results:
        h = r["hotspots"]
        print(
            f"{r['run_id']:>3} | {_fmt(r['elapsed_s'])} | {r['score']:.12f} | "
            f"{_fmt(h['map_exchanges_clips'])} | {_fmt(h['map_dynamic_locations'])} | "
            f"{_fmt(h['_compute_average_cf_cached'])} | {_fmt(h['compute_average_cf'])} | "
            f"{_fmt(h['process_cf_list'])}"
        )

    print(
        "\nrun | cf_hit_rate | cf_hits/misses | pair_match hits/misses | valid_pairs hits/misses | process_cf_list_calls"
    )
    for r in results:
        print(
            f"{r['run_id']:>3} | {r['cf_cache_hit_rate']:.2f}% | "
            f"{r['cf_cache_hits']}/{r['cf_cache_misses']} | "
            f"{r['pair_match_hits']}/{r['pair_match_misses']} | "
            f"{r['valid_pairs_hits']}/{r['valid_pairs_misses']} | "
            f"{r['process_cf_list_calls']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AWARE CLIPSpy hotspots and cache behavior."
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Brightway project name (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated full runs (default: 3)",
    )
    parser.add_argument(
        "--reset-h2-pem",
        action="store_true",
        help="Rebuild h2_pem before benchmark run",
    )
    parser.add_argument(
        "--profile-dir",
        default="dev/profiling/outputs",
        help="Directory where .prof files are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import clips  # noqa: F401
    except Exception:
        raise RuntimeError("clipspy is not available in this environment.")

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    excel_path = (
        Path(__file__).resolve().parent / "lci-hydrogen-electrolysis-ei310.xlsx"
    )

    ensure_h2_pem(args.project, excel_path=excel_path, reset=args.reset_h2_pem)
    activity = pick_activity()

    results = []
    for i in range(1, max(1, args.repeat) + 1):
        results.append(run_once(activity, run_id=i, profile_dir=profile_dir))

    print_summary(results)
    print("\nProfiles written:")
    for r in results:
        print(f"- {r['profile']}")


if __name__ == "__main__":
    main()
