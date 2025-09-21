from numpy.f2py.cfuncs import includes

from edges import EdgeLCIA, get_available_methods
import bw2data, bw2io
import time


total_start_time = time.time()
# Start timer
start_time = time.time()
# bw2data.projects.set_current("bw25_ei310")
# bw2data.projects.set_current("ecoinvent-3.10.1-cutoff")
bw2data.projects.set_current("ecoinvent-3.10-cutoff")

# act = [
#    a
#    for a in bw2data.Database("h2_pem")
#    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
# ][0]

db = bw2data.Database("ecoinvent-3.10-cutoff")
# activities = [a for a in bw2data.Database("h2_pem") if a["name"].startswith("hydrogen")]

# print(activities[0]["name"])
# method = ("GeoPolRisk", "paired", "2024", "short")
method = ("AWARE 2.0", "Country", "all", "yearly")
# method = ("GeoPolRisk", "paired", "2024")

LCA = EdgeLCIA(
    {db.random(): 1},
    method,
    # use_distributions=True,
    # iterations=10000
)
LCA.lci()

LCA.map_exchanges()
LCA.map_aggregate_locations()
LCA.map_dynamic_locations()
LCA.map_contained_locations()
LCA.map_remaining_locations_to_global()

LCA.evaluate_cfs()
LCA.lcia()

# df = LCA.generate_cf_table(include_unmatched=False)
# df.to_excel("df_AWARE.xlsx")

# Stop timer
elapsed_time = time.time() - start_time
print(f"Score: {LCA.score}. Time elapsed: {elapsed_time} seconds.")

for r in range(100):
    act = db.random()
    print(act["name"])
    start_time = time.time()

    LCA.redo_lcia(
        demand={act: 1},
    )

    elapsed_time = time.time() - start_time
    print(f"Score: {LCA.score}. Time elapsed: {elapsed_time} seconds.")

total_elapsed_time = time.time() - total_start_time
print(f"Total time elapsed: {total_elapsed_time} seconds.")
