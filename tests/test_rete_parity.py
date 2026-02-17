from pathlib import Path

import pytest
from bw2data import Database, get_activity, projects, __version__
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
db = Database("lcia-test-db")
activity_A = get_activity(("lcia-test-db", "A"))
activity_D = get_activity(("lcia-test-db", "D"))


def _matched_positions(lcia: EdgeLCIA) -> set[tuple[int, int]]:
    return {
        (int(i), int(j))
        for cf in lcia.cfs_mapping
        for i, j in cf.get("positions", [])
    }


@pytest.mark.forked
@pytest.mark.parametrize(
    "filename, activity",
    [
        ("technosphere_location.json", activity_A),
        ("technosphere_name_refprod_location.json", activity_D),
        ("technosphere_all_fields.json", activity_A),
        ("biosphere_categories.json", activity_A),
    ],
)
def test_rete_supported_methods_clips(filename, activity):
    pytest.importorskip("clips")
    filepath = str(this_dir / "data" / filename)

    rt = EdgeLCIA(demand={activity: 1}, filepath=filepath, matcher_backend="clips")
    rt.lci()
    rt.map_exchanges()

    assert isinstance(_matched_positions(rt), set)
