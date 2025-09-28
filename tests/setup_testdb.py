from bw2data import Database, projects, databases, __version__
import bw2data
import bw2io
import os
from dotenv import load_dotenv
from pathlib import Path


if __version__ < (4, 0, 0):
    is_bw2 = True
else:
    is_bw2 = False
    try:
        projects.migrate_project_25()
    except:
        pass

print(f"Using bw2: {is_bw2}")


if is_bw2 is True:
    project = "EdgeLCIA-Test"
    projects.set_current(project)
else:
    project = "EdgeLCIA-Test-bw25"
    projects.set_current(project)

# Clean up if exists
if "lcia-test-db" in databases:
    del databases["lcia-test-db"]

if "biosphere" in databases:
    del databases["biosphere"]

# Define biosphere flows
biosphere = Database("biosphere")
biosphere.write(
    {
        ("biosphere", "co2"): {
            "name": "Carbon dioxide, in air",
            "unit": "kilogram",
            "categories": ("air",),
            "type": "emission",
        },
        ("biosphere", "co2_low_pop"): {
            "name": "Carbon dioxide, in air",
            "unit": "kilogram",
            "categories": ("air", "low population"),
            "type": "emission",
        },
    }
)

test_db = Database("lcia-test-db")
test_data = [
    # Technosphere activities
    {
        "name": "A",
        "unit": "kg",
        "location": "RER",
        "reference product": "foo",
        "classifications": [
            ("cpc", "01: crops"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "foo",
                "unit": "kg",
                "name": "A",
                "input": ("lcia-test-db", "A"),
            },
            {
                "amount": 0.5,
                "type": "technosphere",
                "input": ("lcia-test-db", "B"),
                "unit": "kg",
            },
            {
                "amount": 0.1,
                "type": "biosphere",
                "input": ("biosphere", "co2"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "B",
        "unit": "kg",
        "location": "CH",
        "reference product": "bar",
        "classifications": [
            ("cpc", "01.1: cereals"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "bar",
                "unit": "kg",
                "name": "B",
                "input": ("lcia-test-db", "B"),
            },
            {
                "amount": 0.2,
                "type": "biosphere",
                "input": ("biosphere", "co2"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "C",
        "unit": "kg",
        "location": "DE",
        "reference product": "baz",
        "classifications": [
            ("cpc", "01.2: vegetables"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "baz",
                "unit": "kg",
                "name": "C",
                "input": ("lcia-test-db", "C"),
            },
            {
                "amount": 0.3,
                "type": "technosphere",
                "input": ("lcia-test-db", "B"),
                "unit": "kg",
            },
            {
                "amount": 0.1,
                "type": "biosphere",
                "input": ("biosphere", "co2_low_pop"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "D",
        "unit": "kg",
        "location": "IT",
        "reference product": "boz",
        "classifications": [],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "boz",
                "unit": "kg",
                "name": "D",
                "input": ("lcia-test-db", "D"),
            },
            {
                "amount": 1,
                "type": "technosphere",
                "input": ("lcia-test-db", "A"),
                "unit": "kg",
            },  # Will inherit CF for RER
        ],
    },
    {
        "name": "E",
        "unit": "kg",
        "location": "CN",
        "reference product": "dummy",
        "classifications": [],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "dummy",
                "unit": "kg",
                "name": "E",
                "input": ("lcia-test-db", "E"),
            },
            {
                "amount": 1,
                "type": "technosphere",
                "input": ("lcia-test-db", "A"),
                "unit": "kg",
            },  # Will fall back to CF for GLO
        ],
    },
]

# Write the technosphere database
test_db.write({(test_db.name, d["name"]): d for d in test_data})


# Now set up ecoinvent database
load_dotenv()

this_dir = Path(__file__).parent


ei_user = os.environ["EI_USERNAME"]
ei_pass = os.environ["EI_PASSWORD"]

ei_version = "3.11"
system_model = "cutoff"

if f"ecoinvent-{ei_version}-{system_model}" not in bw2data.databases:
    bw2io.import_ecoinvent_release(
        version=ei_version,
        system_model=system_model,
        username=ei_user,
        password=ei_pass,
    )

if f"ecoinvent-{ei_version}-biosphere" not in bw2data.databases:
    biosphere_name = "biosphere3"
else:
    biosphere_name = f"ecoinvent-{ei_version}-biosphere"


if "h2_pem" not in bw2data.databases:
    lci = bw2io.ExcelImporter(
        this_dir / "data" / "lci-hydrogen-electrolysis-ei310.xlsx"
    )
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

print(f"Current project: {bw2data.projects.current}")
print(f"The following databases are available: {databases}")
