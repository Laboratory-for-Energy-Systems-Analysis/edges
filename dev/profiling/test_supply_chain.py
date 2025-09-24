from edges import SupplyChain
import bw2data
import time
import pandas as pd

# bw2data.projects.set_current("ecoinvent-3.10-cutoff")
bw2data.projects.set_current("bw25_ei310")

use_example_df = False
start_time = time.time()

if not use_example_df:

    act = [
        a
        for a in bw2data.Database("h2_pem")
        if a["name"]
        == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
    ][0]

    # method = ("AWARE 2.0", "Country", "all", "yearly")
    method = ("GeoPolRisk", "paired", "2024")
    # method = ("ImpactWorld+ 2.1", "Particulate matter formation", "midpoint")
    # method = ("RELICS", "copper", "primary")

    sc = SupplyChain(
        activity=act,
        method=method,
        amount=1,
        level=8,
        cutoff=0.01,
        cutoff_basis="total",  # "total" or "parent"
        redo_flags=dict(
            run_aggregate=False,
            run_dynamic=False,
            run_contained=False,
            run_global=False,
        ),
        collapse_markets=False,
        debug=False,  # <— turn on logging
        dbg_max_prints=5000,
        market_top_k=100,
    )

    # Build initial CM & total score
    total = sc.bootstrap()

    # Walk supply chain
    df, total_score, ref_amount = sc.calculate()
    print("Total score:", total_score, "Reference amount:", ref_amount)

    df.to_csv("example_df.csv")
    print(f"Saved dataframe to example_df.csv with {len(df)} rows.")

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
        collapse_markets=False,
    )
    df = pd.read_csv("example_df.csv")

# fig = sc.plot_sankey(df, width_max=1800, height_max=800, enable_highlight=True)
sc.save_html(
    df,
    path="example_sankey.html",
    y_spacing="linear",
    height_max=1000,
    width_max=1800,
    node_instance_mode="by_parent",  # or "by_child_level" / "by_level"
    node_thickness=12,  # or 10 / 8
    node_pad=8,  # a bit tighter spacing
)
# fig.show()

# Stop timer
elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time} seconds.")
