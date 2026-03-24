import numpy as np
import bw2data
import pytest

from edges import EdgeLCIA


def test_brightway_uncertainty_10000(test_debug_dir):
    acts = [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ]
    if not acts:
        pytest.skip("Required activity not found in 'h2_pem'.")
    act = acts[0]

    method = ("AWARE 2.0", "Country", "all", "yearly")

    lca = EdgeLCIA(
        {act: 1},
        method,
        use_distributions=True,
        iterations=10000,
    )

    lca.apply_strategies()
    lca.evaluate_cfs()
    lca.lcia()

    scores = np.asarray(lca.score, dtype=float)
    median_score = float(np.median(scores))

    (test_debug_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"Iterations: {lca.iterations}",
                f"Median score: {median_score}",
                f"Mean score: {float(np.mean(scores))}",
                f"P05 score: {float(np.percentile(scores, 5))}",
                f"P95 score: {float(np.percentile(scores, 95))}",
            ]
        )
    )

    assert scores.shape[0] == 10000
    assert np.isfinite(median_score)
    assert median_score > 0
