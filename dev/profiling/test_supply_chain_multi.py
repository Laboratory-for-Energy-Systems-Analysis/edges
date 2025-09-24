from edges import SupplyChain
from edges.supply_chain import save_html_multi_methods_for_activity
import bw2data
import time
import pandas as pd

#bw2data.projects.set_current("ecoinvent-3.10-cutoff")
bw2data.projects.set_current("bw25_ei310")

use_example_df = False
start_time = time.time()

act = [
    a
    for a in bw2data.Database("h2_pem")
    if a["name"]
    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
][0]

methods = [("RELICS", "copper", "primary"), ("RELICS", "copper", "secondary")]

save_html_multi_methods_for_activity(
    activity=act,
    methods=methods,
    path="multi_impact.html",
    level=8,
    cutoff=0.01,
    cutoff_basis="total",
    collapse_markets=False,
    plot_kwargs=dict(width_max=1800, height_max=800,
    node_instance_mode="by_child_level",),
    offline=True,
    auto_open=True,
    redo_flags=dict(
        run_aggregate=False,
        run_dynamic=False,
        run_contained=False,
        run_global=False,
    ),
)
