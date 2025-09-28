import os
import numpy as np

import bw2data
import bw2io
import pytest
from dotenv import load_dotenv
from edges import EdgeLCIA
from pathlib import Path

load_dotenv()

this_dir = Path(__file__).parent


ei_user = os.environ["EI_USERNAME"]
ei_pass = os.environ["EI_PASSWORD"]

ei_version = "3.11"
system_model = "cutoff"

bw2data.projects.set_current(f"ecoinvent-{ei_version}-{system_model}")

bw2io.import_ecoinvent_release(
    version=ei_version,
    system_model=system_model,
    username=ei_user,
    password=ei_pass,
)
print(f"Current project: {bw2data.projects.current}")
print(f"Databases: {bw2data.databases}")

if f"ecoinvent-{ei_version}-biosphere" not in bw2data.databases:
    biosphere_name = "biosphere3"
else:
    biosphere_name = f"ecoinvent-{ei_version}-biosphere"

lci = bw2io.ExcelImporter(this_dir / "data" / "lci-hydrogen-electrolysis-ei310.xlsx")
lci.apply_strategies()
lci.match_database(fields=["name", "reference product", "location"])
lci.match_database(
    "ecoinvent-3.11-cutoff", fields=["name", "reference product", "location"]
)
lci.match_database(biosphere_name, fields=["name", "categories"])
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

    assert np.isclose(LCA.score, 0.648, rtol=1e-3)
