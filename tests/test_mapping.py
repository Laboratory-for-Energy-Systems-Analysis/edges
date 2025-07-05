import pytest
from pathlib import Path
from edges import EdgeLCIA
from bw2data import Database, projects, get_activity, __version__

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Set up once
if __version__ < (4, 0, 0):
    print("Using Brightway2.")
    projects.set_current("EdgeLCIA-Test")
else:
    print("Using Brightway2.5.")
    projects.set_current("EdgeLCIA-Test-bw25")

# Create a test database and activities
db = Database("lcia-test-db")
activity_A = get_activity(("lcia-test-db", "A"))
activity_B = get_activity(("lcia-test-db", "B"))
activity_C = get_activity(("lcia-test-db", "C"))
activity_D = get_activity(("lcia-test-db", "D"))
activity_E = get_activity(("lcia-test-db", "E"))

this_dir = Path(__file__).parent


@pytest.mark.parametrize(
    "filename, activity, expected",
    [
        ("technosphere_location.json", activity_A, 50),
        ("technosphere_location.json", activity_B, 0),
        ("technosphere_classifications.json", activity_B, 0),
        ("technosphere_classifications.json", activity_A, 50),
        ("biosphere_name.json", activity_A, 10),
        ("biosphere_categories.json", activity_C, 1.3),
        ("biosphere_categories.json", activity_A, 1.0),
        ("biosphere_categories.json", activity_D, 1.0),
        ("biosphere_name_categories.json", activity_A, 20),
        ("biosphere_name_categories.json", activity_C, 26),
        ("technosphere_name.json", activity_D, 150),
        ("technosphere_name.json", activity_E, 250),
    ],
)
def test_cf_mapping(filename, activity, expected):
    filepath = str(this_dir / "data" / filename)

    lca = EdgeLCIA(
        demand={activity: 1},
        filepath=filepath,
    )

    lca.lci()
    lca.map_exchanges()
    lca.map_aggregate_locations()
    lca.map_dynamic_locations()
    lca.map_contained_locations()
    lca.map_remaining_locations_to_global()
    lca.evaluate_cfs()
    lca.lcia()

    df = lca.generate_cf_table()

    if pytest.approx(lca.score) != expected:
        status = "failed"
    else:
        status = "passed"

    if pytest.approx(lca.score) != expected:
        print(f"\n🔍 DEBUG - Test failed for: {filename} / {activity['name']}")
        print(f"Expected score: {expected}, got: {lca.score}")
        if df is not None:
            print("\n🔎 Full CF table:")
            print(df.to_string(index=False))
            df.to_excel(
                f"test - {filename} {activity['name']} {status}.xlsx", index=False
            )

    assert pytest.approx(lca.score) == expected


def test_parameters():

    activity = activity_A
    filepath = str(this_dir / "data" / "biosphere_name_w_parameters.json")

    params = {
        "some scenario": {
            "parameter_1": {
                "1": 1,
                "2": 2,
            },
            "parameter_2": {
                "1": 1,
                "2": 2,
            },
        }
    }

    lca = EdgeLCIA(
        demand={activity: 1},
        filepath=filepath,
        parameters=params,
    )
    lca.lci()
    lca.map_exchanges()

    results = []
    for scenario in [
        "1",
        "2",
    ]:

        lca.evaluate_cfs(scenario="some scenario", scenario_idx=scenario)
        lca.lcia()
        results.append(lca.score)

    # assert that all values are different
    assert len(set(results)) == len(results), "Expected all values to be different"
