from edges import SupplyChain
from edges.supply_chain import save_html_multi_methods_for_activity
import bw2data
import time
import pandas as pd

bw2data.projects.set_current("ecoinvent-3.10-cutoff")

use_example_df = False
start_time = time.time()

act = [
    a
    for a in bw2data.Database("h2_pem")
    if a["name"]
    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
][0]

methods = [("AWARE 2.0", "Country", "all", "yearly"), ("GeoPolRisk", "paired", "2024")]

save_html_multi_methods_for_activity(
    activity=act,
    methods=methods,
    path="outputs/multi_impact.html",
    level=4,
    cutoff=0.01,
    cutoff_basis="total",
    collapse_markets=True,
    plot_kwargs=dict(width_max=1800, height_max=800),
    offline=True,
    auto_open=True,
)
