from edges import EdgeLCIA, get_available_methods
import bw2data, bw2io
import time

# Start timer
start_time = time.time()
# bw2data.projects.set_current("bw25_ei310")
bw2data.projects.set_current("ecoinvent-3.10.1-cutoff")

act = [
    a
    for a in bw2data.Database("h2_pem")
    if a["name"]
    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
][0]

# method = ("GeoPolRisk", "paired", "2024", "short")
method = ("AWARE 2.0", "Country", "all", "yearly")
# method = ("GeoPolRisk", "paired", "2024")

LCA = EdgeLCIA({act: 1}, method)
LCA.lci()

LCA.map_exchanges()
LCA.map_aggregate_locations()
LCA.map_dynamic_locations()
LCA.map_contained_locations()
LCA.map_remaining_locations_to_global()

LCA.evaluate_cfs()
LCA.lcia()

# Stop timer
elapsed_time = time.time() - start_time
print(f"Score: {LCA.score}. Time elapsed: {elapsed_time} seconds.")
