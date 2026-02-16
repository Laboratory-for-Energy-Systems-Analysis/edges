from __future__ import annotations

import json
import time
from pathlib import Path

import bw2data
import bw2io

from edges import EdgeLCIA

OUTPUT_DIR = Path("dev/profiling/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_h2_pem(project_name: str = "ecoinvent-3.12-cutoff") -> None:
    bw2data.projects.set_current(project_name)
    print("Project:", project_name)
    print("Available databases:", list(bw2data.databases))

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


def mapping_keys(lca: EdgeLCIA) -> set[tuple[int, int]]:
    return {
        (int(i), int(j)) for cf in lca.cfs_mapping for (i, j) in cf.get("positions", [])
    }


def run_one(method: tuple, act, backend: str) -> dict:
    start = time.time()
    lca = EdgeLCIA(
        {act: 1},
        method,
        matcher_backend=backend,
    )
    lca.apply_strategies()
    # lca.map_exchanges()
    lca.evaluate_cfs()
    lca.lcia()
    elapsed = time.time() - start

    return {
        "backend": backend,
        "method": method,
        "score": float(lca.score),
        "elapsed_s": elapsed,
        "mapping_keys": sorted(list(mapping_keys(lca))),
        "mapping_count": len(mapping_keys(lca)),
        "characterization_sum": float(lca.characterization_matrix.sum()),
        "characterized_inventory_sum": float(lca.characterized_inventory.sum()),
    }


def compare(method: tuple, act):
    py = run_one(method, act, "python")
    clips = run_one(method, act, "clips")

    py_keys = set(map(tuple, py["mapping_keys"]))
    clips_keys = set(map(tuple, clips["mapping_keys"]))

    missing = sorted(list(py_keys - clips_keys))
    additional = sorted(list(clips_keys - py_keys))

    diff = {
        "method": method,
        "python": py,
        "clips": clips,
        "same_score": abs(py["score"] - clips["score"]) < 1e-12,
        "same_mapping_keys": not missing and not additional,
        "missing_keys_in_clips": missing,
        "additional_keys_in_clips": additional,
    }
    return diff


def main():
    try:
        import clips  # noqa: F401
    except Exception:
        print("clipspy is not available in this environment.")
        return

    ensure_h2_pem()
    act = pick_activity()

    methods = [
        ("AWARE 2.0", "Country", "all", "yearly"),
        ("GeoPolRisk", "paired", "2024"),
    ]

    for method in methods:
        print("\n== Method:", method)
        diff = compare(method, act)
        print("Same score:", diff["same_score"])
        print("Same mapping keys:", diff["same_mapping_keys"])
        print("Python mapping count:", diff["python"]["mapping_count"])
        print("CLIPS mapping count:", diff["clips"]["mapping_count"])
        print("Python elapsed (s):", round(diff["python"]["elapsed_s"], 3))
        print("CLIPS elapsed (s):", round(diff["clips"]["elapsed_s"], 3))
        print("Missing keys in CLIPS:", len(diff["missing_keys_in_clips"]))
        print("Additional keys in CLIPS:", len(diff["additional_keys_in_clips"]))

        out = OUTPUT_DIR / f"rete_parity_{'_'.join(method)}.json"
        out.write_text(json.dumps(diff, indent=2))
        print("Wrote:", out)


if __name__ == "__main__":
    main()
