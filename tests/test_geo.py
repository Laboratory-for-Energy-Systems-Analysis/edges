from edges.utils import get_str
from edges.georesolver import GeoResolver
from constructive_geometries import Geomatcher
import logging
import pytest


def test_geo():
    geo = Geomatcher()
    parents = [get_str(x) for x in geo.within("IT")]
    assert "RER" in parents
    assert "RER" in [get_str(x) for x in geo.within("IT")]


def test_within():
    from constructive_geometries import Geomatcher

    geo = Geomatcher()
    assert ("ecoinvent", "RER") in geo.within(
        "IT", include_self=True, exclusive=False, biggest_first=False
    )


def test_georesolver():
    geo = GeoResolver(
        weights={
            "RER": 1.0,
        }
    )
    assert "RER" in geo.resolve("IT", containing=False)


def test_georesolver_tuple_weights():
    geo = GeoResolver(
        weights={
            ("CH", "RER"): 1.0,
        }
    )
    assert "RER" in geo.resolve("IT", containing=False)


def test_georesolver_does_not_mutate_global_logging():
    before = logging.lastResort
    _ = GeoResolver(weights={"RER": 1.0})
    assert logging.lastResort is before


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("SGCC", "CN-SGCC"),
        ("HICC", "US-HICC"),
        ("TRE", "US-TRE"),
        ("MRO, US only", "US-MRO"),
        ("NPCC, US only", "US-NPCC"),
        ("WECC, US only", "US-WECC"),
        ("IAI Area 1", "IAI Area, Africa"),
        ("IAI Area 2, North America", "IAI Area, North America"),
        ("IAI Area 3", "IAI Area, South America"),
        ("IAI Area 4&5", "IAI Area, Asia, without China and GCC"),
        ("IAI Area 8", "IAI Area, Gulf Cooperation Council"),
        (
            "IAI Area, Europe outside EU & EFTA",
            "IAI Area, Russia & RER w/o EU27 & EFTA",
        ),
    ],
)
def test_georesolver_resolves_legacy_ecoinvent_aliases(alias, canonical):
    geo = GeoResolver(weights={canonical: 1.0})
    assert canonical in geo.resolve(alias, containing=False)


def test_georesolver_strips_trailing_commas_before_resolving():
    geo = GeoResolver(weights={"Europe": 1.0})
    assert "Europe" in geo.resolve("Europe, ", containing=False)


def test_georesolver_maps_world_alias_to_glo_without_output(capsys):
    geo = GeoResolver(weights={"GLO": 1.0})
    assert "GLO" in geo.resolve("World", containing=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_georesolver_silences_geomatcher_country_converter_output(capsys):
    geo = GeoResolver(weights={"GLO": 1.0, "IN": 1.0, "US": 1.0})

    assert "GLO" in geo.resolve("IND", containing=False)
    assert geo.resolve("XYZ_NOT_A_REGION", containing=False) == []

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_georesolver_uses_bundled_iam_topologies_without_output(capsys):
    geo = GeoResolver(weights={"PK": 1.0, "OAS": 1.0, "GLO": 1.0})

    assert "PK" in geo.resolve("OAS", containing=True)
    assert "OAS" in geo.resolve("PK", containing=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_georesolver_uses_bundled_ei312_topology_for_iai_regions(capsys):
    geo = GeoResolver(
        weights={
            "FR": 1.0,
            "IAI Area, Western and Central Europe": 1.0,
        }
    )

    assert "FR" in geo.resolve(
        "IAI Area, Western and Central Europe", containing=True
    )
    assert "IAI Area, Western and Central Europe" in geo.resolve(
        "FR", containing=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_georesolver_normalizes_topology_members_before_registration(capsys):
    geo = GeoResolver(weights={"PR": 1.0, "RCAM": 1.0})

    assert "PR" in geo.resolve("RCAM", containing=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_georesolver_uses_missing_geographies_for_in_dd():
    geo = GeoResolver(weights={"IN": 1.0, "SAS": 1.0, "UN-ASIA": 1.0})
    result = geo.resolve("IN-DD", containing=False)
    assert "IN" in result
    assert "SAS" in result
    assert "UN-ASIA" in result


def test_georesolver_skips_unresolvable_placeholders_without_output(capsys):
    geo = GeoResolver(weights={"GLO": 1.0})
    assert geo.resolve("not identified", containing=False) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
