from edges import EdgeLCIA
import bw2data

bw2data.projects.set_current("bw25_ei310")
# bw2data.projects.set_current("ecoinvent-3.10-cutoff")

# act = bw2data.Database("ecoinvent-3.10.1-cutoff").random()
# act = bw2data.Database("ecoinvent-3.10-cutoff").random()
act = [
    a
    for a in bw2data.Database("h2_pem")
    if a["name"]
    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
][0]
print(act)


method = ("AWARE 2.0", "Country", "all", "yearly")

LCA = EdgeLCIA(
    demand={act: 1},
    method=method,
    # use_distributions=True,
    # iterations=100
)
LCA.lci()

LCA.map_exchanges()
LCA.map_aggregate_locations()
LCA.map_dynamic_locations()
LCA.map_contained_locations()
LCA.map_remaining_locations_to_global()
df = LCA.generate_cf_table(include_unmatched=False)
print(LCA.score)
df.to_excel("df_AWARE.xlsx")
