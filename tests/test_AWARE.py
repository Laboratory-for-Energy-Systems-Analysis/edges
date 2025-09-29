import numpy as np
import bw2data
from edges import EdgeLCIA


def test_brightway():

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

    print(f"Sum of inventoriy matrix: {LCA.lca.inventory.sum()}")
    print(
        f"Sum of characterized inventoriy matrix: {LCA.lca.characterized_inventory.sum()}"
    )
    print(f"Score: {LCA.score}")

    assert np.isclose(LCA.score, 0.648, rtol=1e-3)
