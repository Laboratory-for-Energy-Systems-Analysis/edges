import copy
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

from edges import EdgeLCIA, get_available_methods
from edges.utils import format_data

BUILDER_PATH = Path(__file__).parents[1] / "dev" / "UBP" / "build_ubp_methods.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("edges_ubp_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ubp = load_builder()
DATA_DIR = Path(__file__).parents[1] / "edges" / "data"


def test_parse_simapro_export_and_cp1252(tmp_path):
    source = tmp_path / "method.csv"
    source.write_bytes(
        (
            "Name\n\nEcological Scarcity 2021\n\n"
            "Version\n\n1;06\n\n"
            "Comment\n\nReviewed caf\xe9 export\n\n"
            "Impact category\nWater pollutants;UBP\nSubstances\n"
            "Water;(unspecified);COD (Chemical Oxygen Demand);;6800;kg\n"
            "Normalization-Weighting set\nWeighting\nWater pollutants;1\nEnd\n"
        ).encode("cp1252")
    )

    parsed = ubp.parse_simapro(source, verify_fingerprint=False)

    assert parsed.name == "Ecological Scarcity 2021"
    assert parsed.version == "1.06"
    assert parsed.description == "Reviewed caf\xe9 export"
    assert parsed.category_weights == {"Water pollutants": 1.0}
    assert parsed.rows[0].value == 6800.0


def test_regional_location_split_does_not_consume_flow_name_suffix():
    row = ubp.SourceRow(
        row_id="Land use:1",
        category="Land use",
        compartment="Raw",
        subcompartment="(unspecified)",
        substance="Occupation, annual crop, flooded crop",
        cas="",
        value=520.0,
        unit="m2a",
        line=1,
    )

    assert ubp.split_source_location(row, {}, {"CH", "GLO"}) == (
        "Occupation, annual crop, flooded crop",
        None,
    )


@pytest.mark.parametrize(
    "source,target",
    [
        ("COD (Chemical Oxygen Demand)", "COD, Chemical Oxygen Demand"),
        ("Particulates, < 2.5 um", "Particulate Matter, < 2.5 um"),
        (
            "Particulates, < 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
        ),
        ("Water, well", "Water, well, in ground"),
    ],
)
def test_reviewed_name_aliases(source, target):
    assert target in ubp._name_candidates(source)


@pytest.mark.parametrize("prefix", [(), ("ecoinvent-3.12",)])
def test_reference_category_accepts_versioned_method_names(prefix):
    method = prefix + (
        "Ecological Scarcity 2021",
        "climate change",
        "global warming potential (GWP100)",
    )
    assert ubp.reference_category(method) == "Global warming"
    assert ubp.reference_category(list(method)) == "Global warming"


@pytest.mark.parametrize(
    "method",
    [
        (),
        ("Other method", "climate change", "global warming potential (GWP100)"),
        (
            "ecoinvent-3.11",
            "Ecological Scarcity 2021",
            "climate change",
            "global warming potential (GWP100)",
        ),
        (
            "ecoinvent-3.12",
            "Ecological Scarcity 2021 no LT",
            "climate change no LT",
            "global warming potential (GWP100) no LT",
        ),
        ("Ecological Scarcity 2021", "water use", "water resources, evaporated"),
    ],
)
def test_reference_category_does_not_mix_in_other_methods(method):
    assert ubp.reference_category(method) is None


@pytest.mark.parametrize(
    "source,target,value",
    [
        (
            "Carbon dioxide, land transformation",
            "Carbon dioxide, from soil or biomass stock",
            1000.0,
        ),
        ("Methane, biogenic", "Methane, non-fossil", 28000.0),
        (
            "Methane, land transformation",
            "Methane, from soil or biomass stock",
            30000.0,
        ),
    ],
)
def test_carbon_flow_aliases_do_not_select_other_carbon_origins(source, target, value):
    """Shared CAS numbers must not drop or misassign carbon-origin-specific CFs."""
    row = ubp.SourceRow(
        row_id="Global warming:1",
        category="Global warming",
        compartment="Air",
        subcompartment="(unspecified)",
        substance=source,
        cas="124-38-9" if source.startswith("Carbon dioxide") else "74-82-8",
        value=value,
        unit="kg",
        line=1,
    )
    prefix = "Carbon dioxide" if source.startswith("Carbon dioxide") else "Methane"
    flows = [
        ubp.FlowRecord(
            name=name,
            categories=("air",),
            unit="kilogram",
            cas=row.cas,
            versions=("3.12",),
        )
        for name in (
            f"{prefix}, fossil",
            f"{prefix}, non-fossil",
            f"{prefix}, from soil or biomass stock",
        )
    ]
    parsed = ubp.ParsedMethod(
        name=ubp.METHOD_NAME,
        version="1.06",
        description="Carbon-origin mapping regression",
        category_units={"Global warming": "UBP"},
        category_weights={"Global warming": 1.0},
        rows=[row],
        source_sha256="test-fixture",
    )

    categories, dispositions = ubp.build_biosphere_exchanges(parsed, flows, set())

    assert len(categories["Global warming"]) == 1
    exchange = categories["Global warming"][0]
    assert exchange["supplier"]["name"] == target
    assert exchange["value"] == value
    assert dispositions[row.row_id]["resolution"] == "exact-name"


@pytest.mark.parametrize(
    "marker,payload",
    [
        ("3.5-7.5 metric ton", 0.98),
        ("7.5-16 metric ton", 3.29),
        ("16-32 metric ton", 5.79),
        (">32 metric ton", 15.96),
        ("28 metric ton", 5.79),
    ],
)
def test_lorry_noise_uses_reviewed_payloads(marker, payload):
    assert ubp.lorry_payload(f"transport by lorry, {marker}", "") == payload
    assert 180 / ubp.lorry_payload(marker, "") == pytest.approx(180 / payload)


@pytest.mark.parametrize(
    "product,unit,family",
    [
        ("transport, freight, aircraft", "ton kilometer", "aircraft freight"),
        ("transport, passenger aircraft", "person kilometer", "aircraft passenger"),
        ("transport, freight, train", "ton kilometer", "rail freight"),
        ("transport, passenger, train", "person kilometer", "rail passenger"),
        ("transport, passenger, car", "kilometer", "passenger car"),
        ("transport, freight, lorry, 16-32 metric ton", "ton kilometer", "lorry"),
    ],
)
def test_noise_transport_classification(product, unit, family):
    assert ubp.classify_noise_activity(product, unit) == family


def test_consolidation_sums_categories_without_duplicate_signatures():
    exchange = {
        "supplier": {
            "name": "Carbon dioxide, fossil",
            "categories": ["air"],
            "unit": "kilogram",
            "matrix": "biosphere",
        },
        "consumer": {"matrix": "technosphere"},
        "value": 1000.0,
    }

    result = ubp.consolidate_exchanges([(exchange, 1.0), (exchange, 2.0)])

    assert len(result) == 1
    assert result[0]["value"] == 3000.0
    assert ubp.find_duplicate_signatures({"exchanges": result}) == []


def test_duplicate_effective_signatures_are_rejected():
    exchange = {
        "supplier": {
            "name": "Water",
            "categories": ["water"],
            "unit": "cubic meter",
            "matrix": "biosphere",
        },
        "consumer": {"matrix": "technosphere", "location": "CH"},
        "value": 1.0,
    }

    assert len(ubp.find_duplicate_signatures({"exchanges": [exchange, exchange]})) == 1


def load_bundled_methods():
    files = sorted(DATA_DIR.glob("Ecological Scarcity 2021_*_UBP.json"))
    return {
        path.stem.split("_", 1)[1].rsplit("_UBP", 1)[0]: json.loads(path.read_text())
        for path in files
    }


def test_all_category_methods_and_total_are_bundled():
    methods = load_bundled_methods()

    assert set(methods) == {
        *ubp.REFERENCE_LEAF_TO_CATEGORY.values(),
        "Water resources, net balance",
        "Noise",
        "total",
    }
    assert len(methods) == 21
    for category, payload in methods.items():
        assert payload["unit"] == "UBP"
        assert payload["version"] == "1.06"
        assert payload["source"]["sha256"] == ubp.EXPECTED_SOURCE_SHA256
        assert payload["source"]["category"] == category
        assert payload["strategies"] == ubp.STRATEGIES
        assert ubp.find_duplicate_signatures(payload) == []


def test_noise_is_technosphere_only_and_total_is_mixed():
    methods = load_bundled_methods()

    assert {
        exchange["supplier"]["matrix"] for exchange in methods["Noise"]["exchanges"]
    } == {"technosphere"}
    assert {
        exchange["supplier"]["matrix"] for exchange in methods["total"]["exchanges"]
    } == {"biosphere", "technosphere"}


def test_total_is_exact_sum_of_category_methods():
    methods = load_bundled_methods()
    expected = {}
    for category, payload in methods.items():
        if category == "total":
            continue
        for exchange in payload["exchanges"]:
            signature = ubp.edge_signature(exchange)
            expected[signature] = expected.get(signature, 0.0) + exchange["value"]
    actual = {
        ubp.edge_signature(exchange): exchange["value"]
        for exchange in methods["total"]["exchanges"]
    }

    assert actual == expected


def test_all_bundled_methods_pass_the_real_edges_loader():
    for payload in load_bundled_methods().values():
        format_data(copy.deepcopy(payload), weight="none")


def test_all_bundled_methods_are_discoverable():
    available = set(get_available_methods())
    expected = {
        ("Ecological Scarcity 2021", category, "UBP")
        for category in load_bundled_methods()
    }
    assert expected <= available


def test_reviewed_numeric_anchors_are_preserved():
    methods = load_bundled_methods()

    def values(category, name, categories=None):
        return {
            exchange["value"]
            for exchange in methods[category]["exchanges"]
            if exchange["supplier"].get("name") == name
            and (
                categories is None
                or tuple(exchange["supplier"].get("categories") or ()) == categories
            )
        }

    assert values("Global warming", "Carbon dioxide, fossil", ("air",)) == {1000.0}
    assert values(
        "Global warming",
        "Carbon dioxide, fossil",
        ("air", "lower stratosphere + upper troposphere"),
    ) == {3100.0}
    assert values(
        "Energy resources", "Gas, natural", ("natural resource", "in ground")
    ) == {330.0}


@pytest.mark.parametrize(
    "name,factor",
    [
        ("Carbon dioxide, from soil or biomass stock", 1000.0),
        ("Methane, non-fossil", 28000.0),
        ("Methane, from soil or biomass stock", 30000.0),
    ],
)
def test_bundled_carbon_origins_have_source_factors_in_all_supported_compartments(
    name, factor
):
    methods = load_bundled_methods()
    expected_categories = {
        ("air",),
        ("air", "low population density, long-term"),
        ("air", "non-urban air or from high stacks"),
        ("air", "urban air close to ground"),
    }
    for category in ("Global warming", "total"):
        rows = [
            r for r in methods[category]["exchanges"] if r["supplier"]["name"] == name
        ]
        assert {tuple(r["supplier"]["categories"]) for r in rows} == expected_categories
        assert len(rows) == len(expected_categories)
        assert all(
            r["value"] == factor and r["supplier"]["unit"] == "kilogram" for r in rows
        )


def test_noise_json_covers_every_reviewed_transport_family():
    noise = load_bundled_methods()["Noise"]
    families = Counter(
        ubp.classify_noise_activity(
            exchange["supplier"]["reference product"],
            exchange["supplier"]["unit"],
        )
        for exchange in noise["exchanges"]
    )
    assert families == Counter(
        {
            "aircraft freight": 8,
            "aircraft passenger": 8,
            "lorry": 121,
            "passenger car": 58,
            "rail freight": 10,
            "rail passenger": 10,
        }
    )
    assert None not in families
    for exchange in noise["exchanges"]:
        supplier = exchange["supplier"]
        assert (
            not supplier["name"]
            .lower()
            .startswith(("market for ", "market group for "))
        )


def test_validation_summary_pins_source_only_and_per_version_coverage():
    summary = json.loads(
        (BUILDER_PATH.parent / "ubp_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["source"]["source_only_groups_sha256"] == (
        ubp.EXPECTED_SOURCE_ONLY_GROUPS_SHA256
    )
    coverage = summary["generated"]["rule_reachability_by_version"]
    assert set(coverage) == {"3.10", "3.11", "3.12"}
    for row in coverage.values():
        assert row["biosphere_rules_reachable"] + row["biosphere_rules_absent"] == (
            row["biosphere_rules_total"]
        )
        assert (
            row["technosphere_rules_reachable"] + row["technosphere_rules_absent"]
            == row["technosphere_rules_total"]
        )


@pytest.mark.ecoinvent
@pytest.mark.serial
def test_total_method_against_ecoinvent_311_when_available(request):
    """Exercise the bundled mixed-matrix method on the CI ecoinvent fixture."""
    import bw2data as bd
    from packaging.version import Version

    original_project = bd.projects.current
    request.addfinalizer(
        lambda: bd.projects.set_current(original_project, writable=False, update=False)
    )

    version = bd.__version__
    if isinstance(version, tuple):
        version = ".".join(map(str, version))
    database_name = "ecoinvent-3.11-cutoff"
    test_project = (
        "EdgeLCIA-Test" if Version(version) < Version("4") else "EdgeLCIA-Test-bw25"
    )
    project_names = {project.name for project in bd.projects}
    selected_project = None
    for project in (test_project, database_name):
        if project not in project_names:
            continue
        bd.projects.set_current(project, writable=False, update=False)
        if database_name in bd.databases:
            selected_project = project
            break
    if selected_project is None:
        pytest.skip(f"{database_name} is not installed in a compatible project")

    activity_name = (
        "market for transport, freight, lorry 28 metric ton, "
        "fatty acid methyl ester 100%"
    )
    matches = [
        activity
        for activity in bd.Database(database_name)
        if activity.get("name") == activity_name
        and activity.get("reference product")
        == activity_name.removeprefix("market for ")
        and activity.get("location") == "CH"
        and activity.get("unit") == "ton kilometer"
    ]
    assert len(matches) == 1

    lcia = EdgeLCIA(
        demand={matches[0]: 1},
        method=("Ecological Scarcity 2021", "total", "UBP"),
    )
    lcia.lci()
    lcia.apply_strategies()
    lcia.evaluate_cfs()
    lcia.lcia()

    assert math.isfinite(lcia.score)
    # Allow small numerical differences between legacy Brightway and bw25.
    assert lcia.score == pytest.approx(951.572600991, rel=1e-7)
    assert lcia.score_by_matrix["technosphere"] == pytest.approx(
        32.44295695125423, rel=1e-8
    )
    assert sum(row["new_biosphere_matches"] for row in lcia.strategy_match_counts) > 0
    assert (
        sum(row["new_technosphere_matches"] for row in lcia.strategy_match_counts) > 0
    )
