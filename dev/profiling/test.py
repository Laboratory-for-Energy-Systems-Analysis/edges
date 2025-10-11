from edges import EdgeLCIA, get_available_methods, setup_package_logging
import bw2data, bw2io
import time
import logging

# setup_package_logging(level=logging.DEBUG)

# Start timer
start_time = time.time()
bw2data.projects.set_current("bw25_ei310")
# bw2data.projects.set_current("ecoinvent-3.10.1-cutoff")
# bw2data.projects.set_current("ecoinvent-3.10-cutoff")
# bw2data.projects.set_current("ecoinvent-3.11-cutoff")
# bw2data.projects.set_current("ecoinvent-3.11-cutoff-bw25")

if "h2_pem" not in bw2data.databases:
    lci = bw2io.ExcelImporter("lci-hydrogen-electrolysis-ei310.xlsx")
    lci.apply_strategies()
    lci.match_database(fields=["name", "reference product", "location"])
    lci.match_database(
        "ecoinvent-3.11-cutoff", fields=["name", "reference product", "location"]
    )
    lci.match_database("ecoinvent-3.11-biosphere", fields=["name", "categories"])
    lci.statistics()
    lci.drop_unlinked(i_am_reckless=True)
    if len(list(lci.unlinked)) == 0:
        lci.write_database()

    # we assign manually classifications to activities
    # since it is unclear how to do that in the Excel inventory file.

    classifications = {
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from grid electricity": (
            "CPC",
            "34210",
        ),
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from solar photovoltaic electricity": (
            "CPC",
            "34210",
        ),
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from onshore wind electricity": (
            "CPC",
            "34210",
        ),
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity": (
            "CPC",
            "34210",
        ),
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from geothermal electricity": (
            "CPC",
            "34210",
        ),
        "electrolyzer production, 1MWe, PEM, Stack": (
            "CPC",
            "4220:Construction of utility projects",
        ),
        "treatment of electrolyzer stack, 1MWe, PEM": ("CPC", "3830"),
        "electrolyzer production, 1MWe, PEM, Balance of Plant": (
            "CPC",
            "4220:Construction of utility projects",
        ),
        "treatment of electrolyzer balance of plant, 1MWe, PEM": ("CPC", "3830"),
        "platinum group metal, extraction and refinery operations": ("CPC", "2420"),
        "deionized water production, via reverse osmosis, from brackish water": (
            "CPC",
            "34210",
        ),
    }
    for ds in bw2data.Database("h2_pem"):
        if ds["name"] in classifications:
            ds["classifications"] = [classifications[ds["name"]]]
            ds.save()

act = [
    a
    for a in bw2data.Database("h2_pem")
    if a["name"]
    == "hydrogen production, gaseous, 30 bar, from PEM electrolysis, from offshore wind electricity"
][0]

# method = ("AWARE 2.0", "Country", "all", "yearly")
method = ("GLAM3", "biodiversity", "occupation", "average", "amphibians")
# method = ("RELICS", "copper", "secondary")

LCA = EdgeLCIA(
    {act: 1},
    method,
    # use_distributions=True,
    # iterations=10000
)
# LCA.lci()

LCA.apply_strategies()
# LCA.map_exchanges()
# LCA.map_aggregate_locations()
# LCA.map_dynamic_locations()
# LCA.map_contained_locations()
# LCA.map_remaining_locations_to_global()

LCA.evaluate_cfs()
LCA.lcia()

# df = LCA.generate_cf_table(include_unmatched=True)
# df.to_excel("df_GeoPolRisk.xlsx")

# Stop timer
elapsed_time = time.time() - start_time

print(f"Sum of inventory matrix: {LCA.lca.inventory.sum()}")
print(f"Sum of characterization matrix: {LCA.characterization_matrix.sum()}")
print(f"Sum of characterized inventory matrix: {LCA.characterized_inventory.sum()}")
print(f"Score: {LCA.score}")

# df = LCA.generate_cf_table(include_unmatched=False)
# df.to_csv("cf_table (local).csv", index=False)

print(f"Score: {LCA.score}. Time elapsed: {elapsed_time} seconds.")
