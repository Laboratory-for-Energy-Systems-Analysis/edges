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
        (int(i), int(j))
        for cf in lca.cfs_mapping
        for (i, j) in cf.get("positions", [])
    }


def run_one(method: tuple, act) -> dict:
    start = time.time()
    lca = EdgeLCIA(
        {act: 1},
        method,
        matcher_backend="clips",
    )
    lca.apply_strategies()
    #lca.map_exchanges()
    lca.evaluate_cfs()
    lca.lcia()
    elapsed = time.time() - start

    return {
        "backend": "clips",
        "method": method,
        "score": float(lca.score),
        "elapsed_s": elapsed,
        "mapping_keys": sorted(list(mapping_keys(lca))),
        "mapping_count": len(mapping_keys(lca)),
        "characterization_sum": float(lca.characterization_matrix.sum()),
        "characterized_inventory_sum": float(lca.characterized_inventory.sum()),
    }


def profile_method(method: tuple, act):
    clips = run_one(method, act)
    return {"method": method, "clips": clips}


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
        result = profile_method(method, act)
        print("CLIPS mapping count:", result["clips"]["mapping_count"])
        print("CLIPS elapsed (s):", round(result["clips"]["elapsed_s"], 3))

        out = OUTPUT_DIR / f"clips_profile_{'_'.join(method)}.json"
        out.write_text(json.dumps(result, indent=2))
        print("Wrote:", out)


if __name__ == "__main__":
    main()
