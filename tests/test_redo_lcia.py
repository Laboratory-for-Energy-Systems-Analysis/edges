from pathlib import Path
import random
import time

import pytest
from bw2data import __version__, get_activity, projects, Database, databases
from packaging.version import Version

from edges import EdgeLCIA

if isinstance(__version__, tuple):
    __version__ = ".".join(map(str, __version__))

__version__ = Version(__version__)

if __version__ < Version("4.0.0"):
    projects.set_current("EdgeLCIA-Test")
else:
    projects.set_current("EdgeLCIA-Test-bw25")

this_dir = Path(__file__).parent
activity_A = get_activity(("lcia-test-db", "A"))
activity_D = get_activity(("lcia-test-db", "D"))
activity_E = get_activity(("lcia-test-db", "E"))


def _run_full(demand, filepath):
    lca = EdgeLCIA(demand=demand, filepath=filepath)
    lca.lci()
    lca.map_exchanges()
    lca.map_aggregate_locations()
    lca.map_dynamic_locations()
    lca.map_contained_locations()
    lca.map_remaining_locations_to_global()
    lca.evaluate_cfs()
    lca.lcia()
    return lca


@pytest.mark.forked
def test_redo_lcia_matches_fresh_runs_after_demand_change():
    filepath = str(this_dir / "data" / "biosphere_name.json")

    lca = _run_full({activity_A: 1}, filepath)
    baseline = float(lca.score)
    assert baseline > 0

    lca.redo_lcia(demand={activity_A.id: 2})
    assert pytest.approx(lca.score) == baseline * 2

    lca.redo_lcia(demand={activity_A.id: 1})
    assert pytest.approx(lca.score) == baseline


@pytest.mark.forked
def test_redo_lcia_switch_activity_matches_fresh_full_run_with_strategies():
    method = {
        "name": "redo test",
        "version": "1.0",
        "unit": "kg",
        "strategies": [
            "map_exchanges",
            "map_aggregate_locations",
            "map_dynamic_locations",
            "map_contained_locations",
            "map_remaining_locations_to_global",
        ],
        "exchanges": [
            {
                "value": 100,
                "weight": 1.0,
                "supplier": {"matrix": "technosphere"},
                "consumer": {"matrix": "technosphere", "location": "RER"},
            },
            {
                "value": 200,
                "weight": 1.0,
                "supplier": {"matrix": "technosphere"},
                "consumer": {"matrix": "technosphere", "location": "GLO"},
            },
        ],
    }

    lca = EdgeLCIA(demand={activity_D: 1}, method=method)
    lca.lci()
    lca.apply_strategies()
    lca.evaluate_cfs()
    lca.lcia()

    fresh_d = EdgeLCIA(demand={activity_D: 1}, method=method)
    fresh_d.lci()
    fresh_d.apply_strategies()
    fresh_d.evaluate_cfs()
    fresh_d.lcia()
    assert pytest.approx(lca.score) == fresh_d.score

    lca.redo_lcia(demand={activity_E.id: 1})
    fresh_e = EdgeLCIA(demand={activity_E: 1}, method=method)
    fresh_e.lci()
    fresh_e.apply_strategies()
    fresh_e.evaluate_cfs()
    fresh_e.lcia()
    assert pytest.approx(lca.score) == fresh_e.score


def _run_full_method(demand, method):
    lca = EdgeLCIA(demand=demand, method=method)
    t0 = time.perf_counter()
    lca.lci()
    lca.apply_strategies()
    lca.evaluate_cfs()
    lca.lcia()
    elapsed = time.perf_counter() - t0
    return lca, elapsed


@pytest.mark.forked
def test_redo_lcia_hydrogen_then_two_random_activities_with_timing():
    if "h2_pem" not in databases:
        pytest.skip("Required database 'h2_pem' not found.")

    db = Database("h2_pem")
    hydrogen_name = (
        "hydrogen production, gaseous, 30 bar, from PEM electrolysis, "
        "from offshore wind electricity"
    )
    hydrogen = next((a for a in db if a["name"] == hydrogen_name), None)
    if hydrogen is None:
        pytest.skip("Hydrogen reference activity not found in 'h2_pem'.")

    candidates = [a for a in db if a.id != hydrogen.id]
    if len(candidates) < 2:
        pytest.skip("Need at least 2 additional activities in 'h2_pem'.")

    rng = random.Random(42)
    act2, act3 = rng.sample(candidates, 2)
    method = ("AWARE 2.0", "Country", "all", "yearly")

    lca, t_full = _run_full_method({hydrogen: 1}, method)

    t0 = time.perf_counter()
    lca.redo_lcia(demand={act2.id: 1})
    t_redo_2 = time.perf_counter() - t0
    redo_score_2 = lca.score

    t0 = time.perf_counter()
    lca.redo_lcia(demand={act3.id: 1})
    t_redo_3 = time.perf_counter() - t0
    redo_score_3 = lca.score

    fresh2, t_fresh_2 = _run_full_method({act2: 1}, method)
    fresh3, t_fresh_3 = _run_full_method({act3: 1}, method)

    assert pytest.approx(redo_score_2) == fresh2.score
    assert pytest.approx(redo_score_3) == fresh3.score

    print(
        "redo_lcia timings (s): "
        f"full={t_full:.3f}, "
        f"redo_2={t_redo_2:.3f}, "
        f"redo_3={t_redo_3:.3f}, "
        f"fresh_2={t_fresh_2:.3f}, "
        f"fresh_3={t_fresh_3:.3f}"
    )
    print(
        "activities: "
        f"start='{hydrogen['name']}', "
        f"second='{act2['name']}', "
        f"third='{act3['name']}'"
    )
