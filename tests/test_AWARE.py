import numpy as np
import logging
import bw2data
from edges import EdgeLCIA, setup_package_logging

setup_package_logging(level=logging.DEBUG)


def test_brightway(test_debug_dir):

    act = [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ][0]

    method = ("AWARE 2.0", "Country", "all", "yearly")

    LCA = EdgeLCIA(
        {act: 1},
        method,
    )

    LCA.apply_strategies()
    LCA.evaluate_cfs()
    LCA.lcia()

    df = LCA.generate_cf_table(include_unmatched=False)

    # --- Dump everything useful for CI inspection ---
    # DataFrame: CSV (easy to peek) + Parquet (lossless types, faster)
    df.to_csv(test_debug_dir / "cf_table.csv", index=False)
    try:
        df.to_csv(test_debug_dir / "cf_table.csv")
    except Exception:
        pass

    # Scalars / quick diagnostics
    (test_debug_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"Sum of inventory matrix: {LCA.lca.inventory.sum()}",
                f"Sum of characterization matrix: {LCA.characterization_matrix.sum()}",
                f"Sum of characterized inventory matrix: {LCA.characterized_inventory.sum()}",
                f"Score: {LCA.score}",
            ]
        )
    )

    assert np.isclose(LCA.score, 0.7971, rtol=1e-3)
