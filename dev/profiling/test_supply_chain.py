from edges import SupplyChain
import bw2data
import time
import pandas as pd

bw2data.projects.set_current("ecoinvent-3.10-cutoff")

use_example_df = False
start_time = time.time()

if not use_example_df:

    act = [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ][0]

    method = ("AWARE 2.0", "Country", "all", "yearly")
    method = ("GeoPolRisk", "paired", "2024")

    sc = SupplyChain(
        activity=act,
        method=method,
        amount=1,
        level=7,
        cutoff=0.005,
        redo_flags=dict(
            run_aggregate=True,
            run_dynamic=True,
            run_contained=True,
            run_global=True,
        ),
        collapse_markets=True,
        debug=False,  # <— turn on logging
        dbg_max_prints=5000,
    )

    # Build initial CM & total score
    total = sc.bootstrap()

    # Walk supply chain
    df, total_score, ref_amount = sc.calculate()
    print("Total score:", total_score, "Reference amount:", ref_amount)

    df.to_csv("example_df.csv")

else:

    act = [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ][0]

    method = ("AWARE 2.0", "Country", "all", "yearly")

    sc = SupplyChain(
        activity=act,
        method=method,
        amount=1,
        level=5,
        cutoff=0.01,
        redo_flags=dict(
            run_aggregate=True,
            run_dynamic=True,
            run_contained=True,
            run_global=True,
        ),
        collapse_markets=True,
    )
    df = pd.read_csv("example_df.csv")

fig = sc.plot_sankey(df, width_max=1800, height_max=800)
fig.show()

# Stop timer
elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time} seconds.")
