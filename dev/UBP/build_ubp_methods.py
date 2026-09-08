#!/usr/bin/env python3
"""Build and validate Edges Ecological Scarcity 2021 methods.

The input is a SimaPro 10 CSV export. The export is intentionally not bundled:
pass it with ``--source``. Brightway projects are used as flow vocabularies so
that SimaPro compartment inheritance is expanded to exact Edges signatures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

METHOD_NAME = "Ecological Scarcity 2021"
METHOD_VERSION = "1.06"
METHOD_UNIT = "UBP"
EDGES_FORMAT_VERSION = "2.0"
EXPECTED_SOURCE_SHA256 = (
    "20e46d0acb9e2a676f35636e76ae0ccacacdae122e49459985608a290525147a"
)
EXPECTED_CATEGORY_COUNT = 20
EXPECTED_ROW_COUNT = 14959
EXPECTED_NONZERO_COUNT = 10067
# The 3.12 implementation contains 96 signatures absent from, or categorized
# differently by, the reviewed SimaPro 1.06 export. Pinning their full set turns
# this independent comparison into a regression check without importing their
# factors into the SimaPro-derived method.
EXPECTED_REFERENCE_GAPS_SHA256 = (
    "5c3882def52d9f8629a7fbb68c314bd2bbfbc7874ec0ab35b8ba0646f8596b83"
)
EXPECTED_SOURCE_ONLY_GROUPS_SHA256 = (
    "a6c4a7074335c16b03bfe64b7bf2e4ff4708bac2aabb90694601dd05180083dc"
)

DEFAULT_TARGETS = (
    (
        "3.10",
        "ecoinvent-3.10-cutoff",
        "ecoinvent-3.10-cutoff",
        "ecoinvent-3.10-biosphere",
    ),
    ("3.11", "ecoinvent-3.11-cutoff", "ecoinvent-3.11-cutoff", "biosphere-3.11"),
    ("3.12", "ecoinvent-3.12-cutoff", "ecoinvent-3.12-cutoff", "biosphere"),
)

STRATEGIES = [
    "map_exchanges",
    "map_aggregate_locations",
    "map_dynamic_locations",
    "map_contained_locations",
    "map_remaining_locations_to_global",
]

UNIT_MAP = {
    "kg": "kilogram",
    "m3": "cubic meter",
    "m2a": "square meter-year",
    "kBq": "kilo Becquerel",
    "MJ": "megajoule",
    "tkm": "ton kilometer",
    "personkm": "person kilometer",
    "km": "kilometer",
}

TOP_LEVEL_CATEGORY = {
    "Air": "air",
    "Water": "water",
    "Soil": "soil",
    "Raw": "natural resource",
    "Waste": "inventory indicator",
}

SUBCOMPARTMENT_MAP = {
    ("Air", "low. pop."): ("air", "non-urban air or from high stacks"),
    ("Air", "low. pop., long-term"): ("air", "low population density, long-term"),
    ("Air", "high. pop."): ("air", "urban air close to ground"),
    ("Air", "stratosphere + troposphere"): (
        "air",
        "lower stratosphere + upper troposphere",
    ),
    ("Water", "groundwater"): ("water", "ground-"),
    ("Water", "groundwater, long-term"): ("water", "ground-, long-term"),
    ("Water", "ocean"): ("water", "ocean"),
    ("Soil", "agricultural"): ("soil", "agricultural"),
    ("Soil", "industrial"): ("soil", "industrial"),
}

NAME_ALIASES = {
    "COD (Chemical Oxygen Demand)": "COD, Chemical Oxygen Demand",
    "4-Methyl-2-pentanone": "Methyl isobutyl ketone",
    "AOX, Adsorbable Organic Halogen": "AOX, Adsorbable Organic Halides",
    "Arsenic, ion": "Arsenic ion",
    "Chromium, ion": "Chromium III",
    "Copper, ion": ("Copper", "Copper ion"),
    "Gas, mine, off-gas, process, coal mining, 36 MJ per m3": (
        "Gas, mine, off-gas, process, coal mining",
    ),
    "Gas, natural, 36 MJ per m3": "Gas, natural",
    "Hydrocarbons, aromatic": "PAH, polycyclic aromatic hydrocarbons",
    "Iron, ion": "Iron ion",
    "Manganese (II)": ("Manganese", "Manganese II"),
    "Occupation, annual crop, fallow": "Occupation, cropland fallow (non-use)",
    "Occupation, urban, green areas": "Occupation, urban, green area",
    "Particulates, < 10 um": "Particulate Matter, > 2.5 um and < 10um",
    "Particulates, < 2.5 um": "Particulate Matter, < 2.5 um",
    "Phthalate, butyl-benzyl-": "Butyl benzyl phthalate",
    "Phthalate, dioctyl-": "Dioctyl phthalate",
    "Sand": "Sand, unspecified",
    "Water, cooling, unspecified natural origin/m3": "Water, cooling, unspecified natural origin",
    "Water, unspecified natural origin/m3": "Water, unspecified natural origin",
    "Water, well": "Water, well, in ground",
    "Water/m3": "Water",
    "Water, unspecified natural origin (kg)": "Water, unspecified natural origin",
    "Xylene": "Xylenes, unspecified",
}

STANDARD_CUBIC_METER_SOURCE_NAMES = {
    "Gas, mine, off-gas, process, coal mining, 36 MJ per m3",
    "Gas, natural, 36 MJ per m3",
}

EXTRA_SOURCE_LOCATIONS = {
    "BR-Mid-western grid",
    "BR-South-eastern grid",
    "Central Asia",
    "Europe, without Russia and Turkey",
    "IAI Area, North America, without Quebec",
    "IN-Islands",
    "OECD",
}

NOISE_FACTORS = {
    "aircraft freight": 40.0,
    "aircraft passenger": 4.0,
    "rail freight": 4.8,
    "rail passenger": 8.0,
    "passenger car": 23.0,
}

LORRY_PAYLOADS = {
    "3.5-7.5": 0.98,
    "7.5-16": 3.29,
    "16-32": 5.79,
    ">32": 15.96,
    "28 metric ton": 5.79,
}

NOISE_SOURCE_NAMES = {
    "Noise, aircraft, freight": "aircraft freight",
    "Noise, aircraft, passenger": "aircraft passenger",
    "Noise, rail, freight train": "rail freight",
    "Noise, rail, passenger train, average": "rail passenger",
    "Noise, road, lorry, average": "lorry",
    "Noise, road, passenger car, average": "passenger car",
}

NOISE_SOURCE_UNITS = {
    "Noise, aircraft, freight": "tkm",
    "Noise, aircraft, passenger": "personkm",
    "Noise, rail, freight train": "tkm",
    "Noise, rail, passenger train, average": "personkm",
    "Noise, road, lorry, average": "km",
    "Noise, road, passenger car, average": "km",
}

REFERENCE_LEAF_TO_CATEGORY = {
    "global warming potential (GWP100)": "Global warming",
    "air pollutants and PM": "Main air pollutants and particulates",
    "carcinogenic substances into air": "Carcinogenic substances into air",
    "heavy metals into air": "Heavy metals into air",
    "radioactive substances into air": "Radioactive substances into air",
    "heavy metals into soil": "Heavy metals into soil",
    "pesticides into soil": "Pesticides into soil",
    "heavy metals into water": "Heavy metals into water",
    "POP into water": "Persistent organic pollutants into water",
    "radioactive substances into water": "Radioactive substances into water",
    "water pollutants": "Water pollutants",
    "energy resources": "Energy resources",
    "land use": "Land use",
    "mineral resources": "Mineral resources",
    "biotic resources": "Biotic resources",
    "ozone depletion": "Ozone layer depletion",
    "radioactive waste to deposit": "Radioactive waste to deposit",
    "waste, non radioactive": "Waste, non radioactive",
}


class UBPError(RuntimeError):
    """Raised when the source or generated method is not auditable."""


@dataclass(frozen=True)
class SourceRow:
    row_id: str
    category: str
    compartment: str
    subcompartment: str
    substance: str
    cas: str
    value: float
    unit: str
    line: int


@dataclass
class ParsedMethod:
    name: str
    version: str
    description: str
    category_units: dict[str, str]
    category_weights: dict[str, float]
    rows: list[SourceRow]
    source_sha256: str


@dataclass(frozen=True)
class FlowRecord:
    name: str
    categories: tuple[str, ...]
    unit: str
    cas: str
    versions: tuple[str, ...]


@dataclass(frozen=True)
class ActivityRecord:
    name: str
    reference_product: str
    unit: str
    versions: tuple[str, ...]


def _next_nonempty(rows: list[list[str]], start: int) -> tuple[int, list[str]]:
    for index in range(start, len(rows)):
        if rows[index] and any(cell.strip() for cell in rows[index]):
            return index, rows[index]
    raise UBPError(f"Expected another non-empty row after input row {start + 1}")


def _parse_float(value: str, *, line: int) -> float:
    try:
        result = float(value.strip())
    except ValueError as exc:
        raise UBPError(f"Invalid numeric value {value!r} on line {line}") from exc
    if not math.isfinite(result):
        raise UBPError(f"Non-finite numeric value {value!r} on line {line}")
    return 0.0 if result == 0 else result


def parse_simapro(path: Path, *, verify_fingerprint: bool = True) -> ParsedMethod:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_fingerprint and digest != EXPECTED_SOURCE_SHA256:
        raise UBPError(
            f"Unexpected source SHA-256 {digest}; expected {EXPECTED_SOURCE_SHA256}. "
            "Use --allow-source-mismatch only after reviewing the new export."
        )

    text = raw.decode("cp1252")
    rows = list(csv.reader(text.splitlines(), delimiter=";"))

    def value_after(label: str) -> list[str]:
        try:
            index = next(i for i, row in enumerate(rows) if row == [label])
        except StopIteration as exc:
            raise UBPError(f"Missing SimaPro field {label!r}") from exc
        return _next_nonempty(rows, index + 1)[1]

    name = value_after("Name")[0].strip()
    version_parts = [part.strip() for part in value_after("Version") if part.strip()]
    version = ".".join(version_parts)
    description = value_after("Comment")[0].replace("\x7f", "\n").strip()

    category_units: dict[str, str] = {}
    source_rows: list[SourceRow] = []
    index = 0
    while index < len(rows):
        row = rows[index]
        if row and row[0] == "Normalization-Weighting set":
            break
        if row != ["Impact category"]:
            index += 1
            continue

        category_index, category_row = _next_nonempty(rows, index + 1)
        if len(category_row) != 2:
            raise UBPError(f"Invalid impact-category row on line {category_index + 1}")
        category, category_unit = (part.strip() for part in category_row)
        category_units[category] = category_unit

        substances_index = category_index + 1
        while substances_index < len(rows) and rows[substances_index] != ["Substances"]:
            substances_index += 1
        if substances_index == len(rows):
            raise UBPError(f"Missing Substances block for {category!r}")

        cursor = substances_index + 1
        ordinal = 0
        while cursor < len(rows):
            current = rows[cursor]
            if current and current[0] in {
                "Impact category",
                "Normalization-Weighting set",
                "End",
            }:
                break
            if current and any(cell.strip() for cell in current):
                if len(current) < 6:
                    raise UBPError(
                        f"Invalid substance row on line {cursor + 1}: {current!r}"
                    )
                compartment, subcompartment, substance, cas, factor, unit = (
                    cell.strip() for cell in current[:6]
                )
                ordinal += 1
                source_rows.append(
                    SourceRow(
                        row_id=f"{category}:{ordinal}",
                        category=category,
                        compartment=compartment,
                        subcompartment=subcompartment,
                        substance=substance,
                        cas=cas,
                        value=_parse_float(factor, line=cursor + 1),
                        unit=unit,
                        line=cursor + 1,
                    )
                )
            cursor += 1
        index = cursor

    try:
        weighting_index = next(i for i, row in enumerate(rows) if row == ["Weighting"])
    except StopIteration as exc:
        raise UBPError("Missing Weighting block") from exc
    category_weights: dict[str, float] = {}
    cursor = weighting_index + 1
    while cursor < len(rows):
        current = rows[cursor]
        if current == ["End"]:
            break
        if current and any(cell.strip() for cell in current):
            if len(current) != 2:
                raise UBPError(
                    f"Invalid weighting row on line {cursor + 1}: {current!r}"
                )
            category_weights[current[0].strip()] = _parse_float(
                current[1], line=cursor + 1
            )
        cursor += 1

    if name != METHOD_NAME or version != METHOD_VERSION:
        raise UBPError(
            f"Unexpected method identity {(name, version)!r}; "
            f"expected {(METHOD_NAME, METHOD_VERSION)!r}"
        )
    if set(category_units) != set(category_weights):
        raise UBPError("Impact-category and weighting-category sets differ")
    if any(unit != METHOD_UNIT for unit in category_units.values()):
        raise UBPError(f"All impact categories must use {METHOD_UNIT}")
    unknown_units = sorted({row.unit for row in source_rows} - set(UNIT_MAP))
    if unknown_units:
        raise UBPError(f"Unsupported SimaPro units: {unknown_units}")

    nonzero_count = sum(row.value != 0 for row in source_rows)
    if verify_fingerprint and (
        len(category_units) != EXPECTED_CATEGORY_COUNT
        or len(source_rows) != EXPECTED_ROW_COUNT
        or nonzero_count != EXPECTED_NONZERO_COUNT
    ):
        raise UBPError(
            "Source structure differs from the reviewed baseline: "
            f"categories={len(category_units)}, rows={len(source_rows)}, "
            f"nonzero={nonzero_count}"
        )

    identities = [
        (
            row.category,
            row.compartment,
            row.subcompartment,
            row.substance,
            row.cas,
            row.unit,
        )
        for row in source_rows
    ]
    duplicates = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicates:
        raise UBPError(f"Duplicate source identities found: {duplicates[:5]}")

    return ParsedMethod(
        name=name,
        version=version,
        description=description,
        category_units=category_units,
        category_weights=category_weights,
        rows=source_rows,
        source_sha256=digest,
    )


def normalize_cas(value: str | None) -> str:
    return re.sub(r"[^0-9]", "", value or "").lstrip("0")


def classify_noise_activity(reference_product: str, unit: str) -> str | None:
    product = reference_product.lower()
    if unit == "ton kilometer" and product.startswith("transport, freight, aircraft"):
        return "aircraft freight"
    if unit == "person kilometer" and (
        product.startswith("transport, passenger aircraft")
        or product.startswith("transport, passenger, aircraft")
    ):
        return "aircraft passenger"
    if unit == "ton kilometer" and (
        product.startswith("transport, freight train")
        or product.startswith("transport, freight, train")
    ):
        return "rail freight"
    if unit == "person kilometer" and (
        product.startswith("transport, passenger train")
        or product.startswith("transport, passenger, train")
    ):
        return "rail passenger"
    if unit == "kilometer" and (
        product.startswith("transport, passenger car")
        or product.startswith("transport, passenger, car")
    ):
        return "passenger car"
    if unit == "ton kilometer" and product.startswith("transport, freight, lorry"):
        return "lorry"
    return None


def lorry_payload(name: str, reference_product: str) -> float:
    text = f"{name} | {reference_product}".lower()
    for marker in ("3.5-7.5", "7.5-16", "16-32", ">32", "28 metric ton"):
        if marker in text:
            return LORRY_PAYLOADS[marker]
    raise UBPError(f"No reviewed payload for leaf lorry transport {name!r}")


def _load_brightway_targets_api(targets=DEFAULT_TARGETS):
    try:
        import bw2data as bd
    except ImportError as exc:
        raise UBPError(
            "Brightway is required to build and validate UBP methods"
        ) from exc

    flow_versions: dict[tuple, set[str]] = defaultdict(set)
    activity_versions: dict[tuple, set[str]] = defaultdict(set)
    locations: set[str] = set()
    target_metadata = []
    leaf_noise: dict[tuple, tuple[str, float]] = {}
    noise_aggregators: set[tuple] = set()

    for version, project, activity_database, biosphere_database in targets:
        bd.projects.set_current(project)
        missing = [
            name
            for name in (activity_database, biosphere_database)
            if name not in bd.databases
        ]
        if missing:
            raise UBPError(f"Missing databases in project {project!r}: {missing}")

        biosphere = bd.Database(biosphere_database)
        activities = bd.Database(activity_database)
        target_metadata.append(
            {
                "version": version,
                "project": project,
                "activity_database": activity_database,
                "biosphere_database": biosphere_database,
                "activities": len(activities),
                "biosphere_flows": len(biosphere),
            }
        )

        for flow in biosphere:
            signature = (
                str(flow.get("name") or ""),
                tuple(flow.get("categories") or ()),
                str(flow.get("unit") or ""),
                str(flow.get("CAS number") or ""),
            )
            flow_versions[signature].add(version)

        candidates = []
        for activity in activities:
            name = str(activity.get("name") or "")
            product = str(activity.get("reference product") or "")
            unit = str(activity.get("unit") or "")
            location = str(activity.get("location") or "")
            if location:
                locations.add(location)
            family = classify_noise_activity(product, unit)
            if family:
                candidates.append((activity, name, product, unit, family))

        for activity, name, product, unit, family in candidates:
            signature = (name, product, unit)
            activity_versions[signature].add(version)
            if name.lower().startswith(("market for ", "market group for ")):
                noise_aggregators.add(signature)
                continue
            consumes_same_family = any(
                classify_noise_activity(
                    str(exchange.input.get("reference product") or ""),
                    str(exchange.input.get("unit") or ""),
                )
                == family
                for exchange in activity.technosphere()
            )
            if consumes_same_family:
                noise_aggregators.add(signature)
                continue
            factor = (
                180.0 / lorry_payload(name, product)
                if family == "lorry"
                else NOISE_FACTORS[family]
            )
            existing = leaf_noise.get(signature)
            if existing and existing != (family, factor):
                raise UBPError(f"Conflicting Noise classification for {signature!r}")
            leaf_noise[signature] = (family, factor)

    flows = [
        FlowRecord(name, categories, unit, cas, tuple(sorted(versions)))
        for (name, categories, unit, cas), versions in flow_versions.items()
    ]
    activities = [
        ActivityRecord(name, product, unit, tuple(sorted(versions)))
        for (name, product, unit), versions in activity_versions.items()
    ]
    return flows, activities, locations, leaf_noise, noise_aggregators, target_metadata


def load_brightway_targets(targets=DEFAULT_TARGETS):
    """Load target catalogs through read-only SQLite connections.

    Avoiding ``projects.set_current`` makes this audit safe to run alongside
    another Brightway process and guarantees that validation cannot mutate a
    project merely by opening it.
    """
    try:
        import bw2data as bd
        from bw_processing import safe_filename
    except ImportError as exc:
        raise UBPError(
            "Brightway is required to locate the validation catalogs"
        ) from exc

    base_dir = Path(bd.projects._base_data_dir)
    registry_path = base_dir / "projects.db"
    registry = sqlite3.connect(f"{registry_path.as_uri()}?mode=ro", uri=True)

    flow_versions: dict[tuple, set[str]] = defaultdict(set)
    activity_versions: dict[tuple, set[str]] = defaultdict(set)
    locations: set[str] = set()
    target_metadata = []
    leaf_noise: dict[tuple, tuple[str, float]] = {}
    noise_aggregators: set[tuple] = set()

    try:
        for version, project, activity_database, biosphere_database in targets:
            project_row = registry.execute(
                "SELECT full_hash FROM projectdataset WHERE name = ?", (project,)
            ).fetchone()
            if project_row is None:
                raise UBPError(f"Missing Brightway project {project!r}")
            project_dir = base_dir / safe_filename(project, full=bool(project_row[0]))
            lci_path = project_dir / "lci" / "databases.db"
            if not lci_path.is_file():
                raise UBPError(f"Missing Brightway catalog {lci_path}")
            connection = sqlite3.connect(f"{lci_path.as_uri()}?mode=ro", uri=True)
            try:
                counts = dict(
                    connection.execute(
                        "SELECT database, COUNT(*) FROM activitydataset "
                        "WHERE database IN (?, ?) GROUP BY database",
                        (activity_database, biosphere_database),
                    )
                )
                missing = {
                    activity_database,
                    biosphere_database,
                } - set(counts)
                if missing:
                    raise UBPError(
                        f"Missing databases in project {project!r}: {sorted(missing)}"
                    )
                target_metadata.append(
                    {
                        "version": version,
                        "project": project,
                        "activity_database": activity_database,
                        "biosphere_database": biosphere_database,
                        "activities": counts[activity_database],
                        "biosphere_flows": counts[biosphere_database],
                        "catalog_access": "read-only SQLite",
                    }
                )

                for name, data_blob in connection.execute(
                    "SELECT name, data FROM activitydataset WHERE database = ?",
                    (biosphere_database,),
                ):
                    data = pickle.loads(data_blob)
                    signature = (
                        str(name or data.get("name") or ""),
                        tuple(data.get("categories") or ()),
                        str(data.get("unit") or ""),
                        str(data.get("CAS number") or ""),
                    )
                    flow_versions[signature].add(version)

                candidates = []
                family_by_code = {}
                for code, name, product, location, data_blob in connection.execute(
                    "SELECT code, name, product, location, data FROM activitydataset "
                    "WHERE database = ?",
                    (activity_database,),
                ):
                    data = pickle.loads(data_blob)
                    name = str(name or data.get("name") or "")
                    product = str(product or data.get("reference product") or "")
                    unit = str(data.get("unit") or "")
                    location = str(location or data.get("location") or "")
                    if location:
                        locations.add(location)
                    signature = (name, product, unit)
                    activity_versions[signature].add(version)
                    family = classify_noise_activity(product, unit)
                    if family:
                        family_by_code[code] = family
                        candidates.append((code, name, product, unit, family))

                for code, name, product, unit, family in candidates:
                    signature = (name, product, unit)
                    input_codes = connection.execute(
                        "SELECT input_code FROM exchangedataset WHERE "
                        "output_database = ? AND output_code = ? AND "
                        "input_database = ? AND type = 'technosphere'",
                        (activity_database, code, activity_database),
                    )
                    consumes_same_family = any(
                        family_by_code.get(input_code) == family
                        for (input_code,) in input_codes
                    )
                    if consumes_same_family:
                        noise_aggregators.add(signature)
                        continue
                    factor = (
                        180.0 / lorry_payload(name, product)
                        if family == "lorry"
                        else NOISE_FACTORS[family]
                    )
                    existing = leaf_noise.get(signature)
                    if existing and existing != (family, factor):
                        raise UBPError(
                            f"Conflicting Noise classification for {signature!r}"
                        )
                    leaf_noise[signature] = (family, factor)
            finally:
                connection.close()
    finally:
        registry.close()

    flows = [
        FlowRecord(name, categories, unit, cas, tuple(sorted(versions)))
        for (name, categories, unit, cas), versions in flow_versions.items()
    ]
    activities = [
        ActivityRecord(name, product, unit, tuple(sorted(versions)))
        for (name, product, unit), versions in activity_versions.items()
    ]
    return flows, activities, locations, leaf_noise, noise_aggregators, target_metadata


def load_ecoinvent_312_reference_signatures() -> dict[str, set[tuple]]:
    """Load supplier signatures from ecoinvent's official 3.12 UBP methods.

    These methods are an independent coverage oracle only. Their factors are
    not copied: the reviewed SimaPro export remains numerically authoritative.
    Water and Noise are intentionally excluded because their formulations
    differ between the two implementations.
    """
    try:
        import bw2data as bd
        from bw_processing import safe_filename
    except ImportError as exc:
        raise UBPError("Brightway is required to locate the reference methods") from exc

    base_dir = Path(bd.projects._base_data_dir)
    registry = sqlite3.connect(
        f"{(base_dir / 'projects.db').as_uri()}?mode=ro", uri=True
    )
    try:
        row = registry.execute(
            "SELECT full_hash FROM projectdataset WHERE name = ?",
            ("ecoinvent-3.12-cutoff",),
        ).fetchone()
    finally:
        registry.close()
    if row is None:
        raise UBPError("Missing ecoinvent-3.12-cutoff reference project")
    project_dir = base_dir / safe_filename("ecoinvent-3.12-cutoff", full=bool(row[0]))
    methods = json.loads((project_dir / "methods.json").read_text(encoding="utf-8"))
    lci_path = project_dir / "lci" / "databases.db"
    connection = sqlite3.connect(f"{lci_path.as_uri()}?mode=ro", uri=True)
    output: dict[str, set[tuple]] = defaultdict(set)
    try:
        for method_name, metadata in methods:
            if not method_name or method_name[0] != METHOD_NAME:
                continue
            category = REFERENCE_LEAF_TO_CATEGORY.get(method_name[-1])
            if category is None:
                continue
            data_path = (
                project_dir / "intermediate" / (metadata["abbreviation"] + ".pickle")
            )
            factors = pickle.loads(data_path.read_bytes())
            ids = [int(item[0]) for item in factors]
            placeholders = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT id, name, data FROM activitydataset WHERE id IN ({placeholders})",
                ids,
            )
            found = 0
            for _, name, data_blob in rows:
                data = pickle.loads(data_blob)
                output[category].add(
                    (
                        str(name or data.get("name") or ""),
                        tuple(data.get("categories") or ()),
                        str(data.get("unit") or ""),
                    )
                )
                found += 1
            if found != len(ids):
                raise UBPError(
                    f"Could not resolve all reference flow IDs for {method_name!r}"
                )
    finally:
        connection.close()
    missing_categories = set(REFERENCE_LEAF_TO_CATEGORY.values()) - set(output)
    if missing_categories:
        raise UBPError(
            f"Missing ecoinvent 3.12 reference methods for {sorted(missing_categories)}"
        )
    return dict(output)


def _source_location_vocabulary(
    rows: Iterable[SourceRow], locations: set[str]
) -> set[str]:
    vocabulary = set(locations)
    vocabulary.add("GLO")
    vocabulary.update(EXTRA_SOURCE_LOCATIONS)
    for row in rows:
        token = row.substance.rsplit(", ", 1)[-1]
        if re.fullmatch(r"[A-Z]{2}", token) or re.fullmatch(
            r"[A-Z]{2}-[A-Z0-9][A-Z0-9-]*", token
        ):
            vocabulary.add(token)
    return vocabulary


def split_source_location(
    row: SourceRow,
    regional_bases: dict[tuple[str, str, str, str, str], set[str]],
    locations: set[str],
) -> tuple[str, str | None]:
    for location in sorted(locations, key=len, reverse=True):
        suffix = f", {location}"
        if not row.substance.endswith(suffix):
            continue
        base = row.substance[: -len(suffix)]
        # Geography tokens come from ISO/subdivision syntax or actual target
        # activity locations. Regionalized SimaPro flows do not always include
        # an unlocated sibling in the same unit (notably Water in m3), so a
        # sibling cannot be required here.
        return base, location
    return row.substance, None


def _allowed_flow(row: SourceRow, flow: FlowRecord) -> bool:
    top = TOP_LEVEL_CATEGORY.get(row.compartment)
    if not top or not flow.categories or flow.categories[0] != top:
        return False
    mapped = SUBCOMPARTMENT_MAP.get((row.compartment, row.subcompartment))
    if mapped is not None:
        return flow.categories == mapped
    if row.subcompartment == "(unspecified)":
        if row.compartment == "Waste":
            return flow.categories[:2] == ("inventory indicator", "waste")
        return True
    raise UBPError(
        f"Unsupported compartment mapping {(row.compartment, row.subcompartment)!r} "
        f"on source line {row.line}"
    )


def _name_candidates(name: str) -> set[str]:
    aliases = NAME_ALIASES.get(name, ())
    if isinstance(aliases, str):
        aliases = (aliases,)
    candidates = {name, *aliases}
    if name.endswith("/m3"):
        candidates.add(name[:-3])
    return candidates


def edge_signature(exchange: dict) -> tuple:
    supplier = exchange["supplier"]
    consumer = exchange["consumer"]
    return (
        supplier.get("matrix"),
        supplier.get("name"),
        tuple(supplier.get("categories") or ()),
        supplier.get("reference product"),
        supplier.get("unit"),
        supplier.get("location"),
        consumer.get("matrix"),
        consumer.get("location"),
    )


def build_biosphere_exchanges(
    parsed: ParsedMethod,
    flows: list[FlowRecord],
    locations: set[str],
):
    flow_by_name_unit: dict[tuple[str, str], list[FlowRecord]] = defaultdict(list)
    flow_by_cas_unit: dict[tuple[str, str], list[FlowRecord]] = defaultdict(list)
    for flow in flows:
        flow_by_name_unit[(flow.name, flow.unit)].append(flow)
        if normalize_cas(flow.cas):
            flow_by_cas_unit[(normalize_cas(flow.cas), flow.unit)].append(flow)

    location_vocabulary = _source_location_vocabulary(parsed.rows, locations)
    regional_bases: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in parsed.rows:
        group = (row.category, row.compartment, row.subcompartment, row.cas, row.unit)
        for location in sorted(location_vocabulary, key=len, reverse=True):
            suffix = f", {location}"
            if row.substance.endswith(suffix):
                regional_bases[group].add(row.substance[: -len(suffix)])
                break
    source_names_by_cas: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(
        set
    )
    for row in parsed.rows:
        if row.compartment == "Non mat." or not normalize_cas(row.cas):
            continue
        base_name, _ = split_source_location(row, regional_bases, location_vocabulary)
        source_names_by_cas[
            (
                row.category,
                row.compartment,
                row.subcompartment,
                normalize_cas(row.cas),
                UNIT_MAP[row.unit],
            )
        ].add(base_name)
    candidates_by_category_signature: dict[
        tuple[str, tuple], list[tuple[int, SourceRow, dict]]
    ] = defaultdict(list)
    dispositions: dict[str, dict] = {}

    for row in parsed.rows:
        if row.compartment == "Non mat.":
            continue
        base_name, location = split_source_location(
            row, regional_bases, location_vocabulary
        )
        unit = UNIT_MAP[row.unit]
        target_units = {unit}
        if base_name in STANDARD_CUBIC_METER_SOURCE_NAMES:
            target_units.add("standard cubic meter")
        matches: dict[tuple, FlowRecord] = {}
        for candidate_name in _name_candidates(base_name):
            for target_unit in target_units:
                for flow in flow_by_name_unit.get((candidate_name, target_unit), ()):
                    if _allowed_flow(row, flow):
                        matches[(flow.name, flow.categories, flow.unit)] = flow

        resolution = "exact-name"
        cas_group = (
            row.category,
            row.compartment,
            row.subcompartment,
            normalize_cas(row.cas),
            unit,
        )
        if (
            not matches
            and normalize_cas(row.cas)
            and len(source_names_by_cas[cas_group]) == 1
        ):
            cas_matches = [
                flow
                for flow in flow_by_cas_unit.get((normalize_cas(row.cas), unit), ())
                if _allowed_flow(row, flow)
            ]
            names = {flow.name for flow in cas_matches}
            if len(names) == 1:
                for flow in cas_matches:
                    matches[(flow.name, flow.categories, flow.unit)] = flow
                resolution = "cas-alias"

        dispositions[row.row_id] = {
            "row_id": row.row_id,
            "line": row.line,
            "category": row.category,
            "source_name": row.substance,
            "base_name": base_name,
            "location": location,
            "resolution": resolution if matches else "not-in-supported-ecoinvent",
            "target_signatures": len(matches),
        }

        priority = 0 if row.subcompartment == "(unspecified)" else 1
        for flow in matches.values():
            exchange = {
                "supplier": {
                    "name": flow.name,
                    "categories": list(flow.categories),
                    "unit": flow.unit,
                    "matrix": "biosphere",
                },
                "consumer": {"matrix": "technosphere"},
                "value": row.value,
            }
            if location:
                exchange["consumer"]["location"] = location
            signature = edge_signature(exchange)
            candidates_by_category_signature[(row.category, signature)].append(
                (priority, row, exchange)
            )

    by_category: dict[str, list[dict]] = defaultdict(list)
    conflicts = []
    for (category, signature), candidates in candidates_by_category_signature.items():
        best_priority = max(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_priority]
        values = {item[1].value for item in best}
        if len(values) != 1:
            conflicts.append(
                {
                    "category": category,
                    "signature": signature,
                    "rows": [item[1].row_id for item in best],
                    "values": sorted(values),
                }
            )
            continue
        by_category[category].append(best[0][2])
    if conflicts:
        raise UBPError(f"Conflicting inherited CFs: {conflicts[:5]}")

    for category in by_category:
        by_category[category].sort(key=lambda exc: repr(edge_signature(exc)))
    return by_category, dispositions


def build_noise_exchanges(
    parsed: ParsedMethod, leaf_noise: dict[tuple, tuple[str, float]]
):
    source_noise = {
        row.substance: row for row in parsed.rows if row.compartment == "Non mat."
    }
    if set(source_noise) != set(NOISE_SOURCE_NAMES):
        raise UBPError(
            "Noise source rows differ from the reviewed six-row mapping: "
            f"{sorted(source_noise)}"
        )
    for name, family in NOISE_SOURCE_NAMES.items():
        row = source_noise[name]
        if row.unit != NOISE_SOURCE_UNITS[name]:
            raise UBPError(f"Unexpected source Noise unit for {name!r}: {row.unit!r}")
        if family != "lorry" and row.value != NOISE_FACTORS[family]:
            raise UBPError(f"Unexpected source Noise factor for {name!r}: {row.value}")
        if family == "lorry" and row.value != 180.0:
            raise UBPError(f"Unexpected source lorry Noise factor: {row.value}")

    exchanges = []
    families = Counter()
    for (name, product, unit), (family, factor) in sorted(leaf_noise.items()):
        exchanges.append(
            {
                "supplier": {
                    "name": name,
                    "reference product": product,
                    "unit": unit,
                    "matrix": "technosphere",
                },
                "consumer": {"matrix": "technosphere"},
                "value": factor,
            }
        )
        families[family] += 1
    missing_families = set(NOISE_SOURCE_NAMES.values()) - set(families)
    if missing_families:
        raise UBPError(
            f"No eligible ecoinvent leaf activities for {sorted(missing_families)}"
        )
    return exchanges, dict(sorted(families.items()))


def consolidate_exchanges(exchanges: Iterable[tuple[dict, float]]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    values: dict[tuple, float] = defaultdict(float)
    for exchange, multiplier in exchanges:
        signature = edge_signature(exchange)
        grouped.setdefault(signature, exchange)
        values[signature] += float(exchange["value"]) * multiplier
    output = []
    for signature in sorted(grouped, key=repr):
        exchange = json.loads(json.dumps(grouped[signature]))
        value = values[signature]
        exchange["value"] = 0.0 if value == 0 else value
        output.append(exchange)
    return output


def method_payload(
    parsed: ParsedMethod,
    category: str,
    exchanges: list[dict],
    *,
    description: str,
) -> dict:
    return {
        "name": f"{METHOD_NAME} - {category}",
        "unit": METHOD_UNIT,
        "version": parsed.version,
        "edges version": EDGES_FORMAT_VERSION,
        "description": description,
        "source": {
            "format": "SimaPro CSV 9.0.0",
            "exported_with": "SimaPro 10.4.0.0",
            "sha256": parsed.source_sha256,
            "category": category,
        },
        "reference": (
            "BAFU (Hrsg.) 2021: Ökofaktoren Schweiz 2021 gemäss der Methode "
            "der ökologischen Knappheit. Umwelt-Wissen Nr. 2121."
        ),
        "strategies": STRATEGIES,
        "exchanges": exchanges,
    }


def safe_category_filename(category: str) -> str:
    return f"{METHOD_NAME}_{category}_UBP.json"


def build_methods(parsed: ParsedMethod, flows, locations, leaf_noise):
    by_category, dispositions = build_biosphere_exchanges(parsed, flows, locations)
    noise_exchanges, noise_families = build_noise_exchanges(parsed, leaf_noise)
    by_category["Noise"] = noise_exchanges

    category_payloads = {}
    for category in parsed.category_units:
        exchanges = consolidate_exchanges(
            (exc, 1.0) for exc in by_category.get(category, [])
        )
        if not exchanges:
            raise UBPError(
                f"No executable exchanges generated for category {category!r}"
            )
        category_payloads[category] = method_payload(
            parsed,
            category,
            exchanges,
            description=(
                f"{category} category of Swiss Ecological Scarcity 2021, "
                "converted from the reviewed SimaPro 10.4 method export."
            ),
        )

    total = consolidate_exchanges(
        (exchange, parsed.category_weights[category])
        for category, payload in category_payloads.items()
        for exchange in payload["exchanges"]
    )
    total_payload = method_payload(
        parsed,
        "total",
        total,
        description=(
            "Total Swiss Ecological Scarcity 2021 score. Category eco-factors are "
            "added using the SimaPro weighting set; Noise is characterized on "
            "technosphere transport-service edges."
        ),
    )
    return category_payloads, total_payload, dispositions, noise_families


def find_duplicate_signatures(payload: dict) -> list[tuple]:
    signatures = [edge_signature(exchange) for exchange in payload["exchanges"]]
    return [signature for signature, count in Counter(signatures).items() if count > 1]


def summarize_source_only_rows(
    parsed: ParsedMethod, dispositions: dict[str, dict]
) -> list[dict]:
    """Group audited SimaPro rows which have no supported ecoinvent flow."""
    rows_by_id = {row.row_id: row for row in parsed.rows}
    grouped: dict[tuple, dict] = {}
    for row_id, disposition in dispositions.items():
        if disposition["resolution"] != "not-in-supported-ecoinvent":
            continue
        row = rows_by_id[row_id]
        key = (
            row.category,
            row.compartment,
            row.subcompartment,
            disposition["base_name"],
            row.unit,
        )
        record = grouped.setdefault(
            key,
            {
                "category": row.category,
                "compartment": row.compartment,
                "subcompartment": row.subcompartment,
                "base_name": disposition["base_name"],
                "source_unit": row.unit,
                "rows": 0,
                "nonzero_rows": 0,
                "locations": [],
            },
        )
        record["rows"] += 1
        record["nonzero_rows"] += int(row.value != 0)
        if (
            disposition["location"]
            and disposition["location"] not in record["locations"]
        ):
            if len(record["locations"]) < 8:
                record["locations"].append(disposition["location"])
    return [grouped[key] for key in sorted(grouped)]


def validate_generated(
    parsed: ParsedMethod,
    category_payloads: dict[str, dict],
    total_payload: dict,
    dispositions: dict[str, dict],
    flows: list[FlowRecord],
    activities: list[ActivityRecord],
    target_metadata: list[dict],
    noise_families: dict[str, int],
    reference_signatures: dict[str, set[tuple]],
) -> dict:
    payloads = {**category_payloads, "total": total_payload}
    duplicate_counts = {}
    try:
        from edges.matching_signatures import find_duplicate_clips_signatures
        from edges.utils import format_data
    except ImportError:
        find_duplicate_clips_signatures = None
        format_data = None
    for category, payload in payloads.items():
        duplicates = find_duplicate_signatures(payload)
        effective_duplicates = (
            find_duplicate_clips_signatures(payload["exchanges"])
            if find_duplicate_clips_signatures
            else []
        )
        duplicate_counts[category] = len(duplicates)
        if duplicates or effective_duplicates:
            raise UBPError(
                f"Duplicate effective signatures in {category!r}: "
                f"{(effective_duplicates or duplicates)[:3]}"
            )
        if format_data:
            try:
                format_data(copy.deepcopy(payload), weight="none")
            except (TypeError, ValueError) as exc:
                raise UBPError(
                    f"Invalid Edges method payload for {category!r}"
                ) from exc

    flow_signatures = {(flow.name, flow.categories, flow.unit) for flow in flows}
    activity_signatures = {
        (activity.name, activity.reference_product, activity.unit)
        for activity in activities
    }
    flow_versions = {
        (flow.name, flow.categories, flow.unit): set(flow.versions) for flow in flows
    }
    activity_versions = {
        (activity.name, activity.reference_product, activity.unit): set(
            activity.versions
        )
        for activity in activities
    }
    unreachable = []
    for category, payload in payloads.items():
        for index, exchange in enumerate(payload["exchanges"]):
            supplier = exchange["supplier"]
            if supplier["matrix"] == "biosphere":
                reachable = (
                    supplier["name"],
                    tuple(supplier.get("categories") or ()),
                    supplier["unit"],
                ) in flow_signatures
            else:
                reachable = (
                    supplier["name"],
                    supplier["reference product"],
                    supplier["unit"],
                ) in activity_signatures
            if not reachable:
                unreachable.append((category, index, edge_signature(exchange)))
    if unreachable:
        raise UBPError(
            f"Generated rules unreachable in all target versions: {unreachable[:5]}"
        )

    target_versions = [target["version"] for target in target_metadata]
    rule_reachability_by_version = {}
    for version in target_versions:
        counts = Counter()
        for exchange in total_payload["exchanges"]:
            supplier = exchange["supplier"]
            matrix = supplier["matrix"]
            if matrix == "biosphere":
                signature = (
                    supplier["name"],
                    tuple(supplier.get("categories") or ()),
                    supplier["unit"],
                )
                versions = flow_versions.get(signature, set())
            else:
                signature = (
                    supplier["name"],
                    supplier["reference product"],
                    supplier["unit"],
                )
                versions = activity_versions.get(signature, set())
            counts[f"{matrix}_rules_total"] += 1
            counts[
                (
                    f"{matrix}_rules_reachable"
                    if version in versions
                    else f"{matrix}_rules_absent"
                )
            ] += 1
        rule_reachability_by_version[version] = dict(sorted(counts.items()))

    reference_missing = {}
    for category, expected in reference_signatures.items():
        actual = {
            (
                exchange["supplier"]["name"],
                tuple(exchange["supplier"].get("categories") or ()),
                exchange["supplier"]["unit"],
            )
            for exchange in category_payloads[category]["exchanges"]
            if exchange["supplier"]["matrix"] == "biosphere"
        }
        missing = sorted(expected - actual)
        if missing:
            reference_missing[category] = missing
    reference_gap_rows = [
        (category, name, categories, unit)
        for category, signatures in sorted(reference_missing.items())
        for name, categories, unit in signatures
    ]
    reference_gap_sha256 = hashlib.sha256(
        json.dumps(reference_gap_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if reference_gap_sha256 != EXPECTED_REFERENCE_GAPS_SHA256:
        examples = {
            category: signatures[:8]
            for category, signatures in reference_missing.items()
        }
        raise UBPError(
            "The reviewed ecoinvent 3.12 reference-coverage gap changed: "
            f"count={len(reference_gap_rows)}, sha256={reference_gap_sha256}, "
            f"examples={examples}"
        )

    source_only = [
        record
        for record in dispositions.values()
        if record["resolution"] == "not-in-supported-ecoinvent"
    ]
    matched_rows = len(dispositions) - len(source_only)
    source_rows_by_id = {row.row_id: row for row in parsed.rows}
    noise_rows = sum(row.compartment == "Non mat." for row in parsed.rows)
    if len(dispositions) + noise_rows != len(parsed.rows):
        raise UBPError("Not every SimaPro source row has a validated disposition")
    source_only_nonzero = sum(
        source_rows_by_id[record["row_id"]].value != 0 for record in source_only
    )
    source_only_groups = summarize_source_only_rows(parsed, dispositions)
    source_only_groups_sha256 = hashlib.sha256(
        json.dumps(
            source_only_groups,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if source_only_groups_sha256 != EXPECTED_SOURCE_ONLY_GROUPS_SHA256:
        raise UBPError(
            "The reviewed source-only disposition changed: "
            f"groups={len(source_only_groups)}, sha256={source_only_groups_sha256}"
        )
    summary = {
        "method": METHOD_NAME,
        "version": parsed.version,
        "source_sha256": parsed.source_sha256,
        "source": {
            "categories": len(parsed.category_units),
            "rows": len(parsed.rows),
            "nonzero_rows": sum(row.value != 0 for row in parsed.rows),
            "zero_rows": sum(row.value == 0 for row in parsed.rows),
            "matched_biosphere_rows": matched_rows,
            "source_only_biosphere_rows": len(source_only),
            "source_only_nonzero_rows": source_only_nonzero,
            "source_only_reason": (
                "No compatible name/CAS, compartment and unit signature exists "
                "in the union of ecoinvent cutoff 3.10, 3.11 and 3.12."
            ),
            "source_only_groups": source_only_groups,
            "source_only_groups_sha256": source_only_groups_sha256,
            "noise_technosphere_rows": noise_rows,
            "accounted_rows": len(dispositions) + noise_rows,
        },
        "targets": target_metadata,
        "generated": {
            "category_methods": len(category_payloads),
            "category_exchange_counts": {
                category: len(payload["exchanges"])
                for category, payload in category_payloads.items()
            },
            "total_exchanges": len(total_payload["exchanges"]),
            "total_biosphere_exchanges": sum(
                exchange["supplier"]["matrix"] == "biosphere"
                for exchange in total_payload["exchanges"]
            ),
            "total_technosphere_exchanges": sum(
                exchange["supplier"]["matrix"] == "technosphere"
                for exchange in total_payload["exchanges"]
            ),
            "noise_leaf_signatures": noise_families,
            "duplicate_signature_counts": duplicate_counts,
            "unreachable_rules": 0,
            "rule_reachability_by_version": rule_reachability_by_version,
            "ecoinvent_312_reference_signatures": sum(
                len(signatures) for signatures in reference_signatures.values()
            ),
            "ecoinvent_312_reference_only_signatures": len(reference_gap_rows),
            "ecoinvent_312_reference_gap_sha256": reference_gap_sha256,
        },
        "status": "pass",
    }
    return summary


def render_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def output_files(
    category_payloads: dict[str, dict], total_payload: dict
) -> dict[str, str]:
    files = {
        safe_category_filename(category): render_json(payload)
        for category, payload in category_payloads.items()
    }
    files[safe_category_filename("total")] = render_json(total_payload)
    return files


def write_or_check(files: dict[str, str], output_dir: Path, *, check: bool) -> None:
    mismatches = []
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        path = output_dir / filename
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                mismatches.append(str(path))
        else:
            path.write_text(content, encoding="utf-8")
    if mismatches:
        raise UBPError("Generated files are stale or missing: " + ", ".join(mismatches))


def console_summary(summary: dict) -> dict:
    """Return the compact validation result suitable for terminal output."""
    return {
        "method": summary["method"],
        "version": summary["version"],
        "source_sha256": summary["source_sha256"],
        "source": {
            key: value
            for key, value in summary["source"].items()
            if key != "source_only_groups"
        },
        "targets": summary["targets"],
        "generated": summary["generated"],
        "status": summary["status"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "check"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("Ecological-Scarcity-21.CSV"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "edges" / "data",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).with_name("ubp_validation_summary.json"),
    )
    parser.add_argument("--allow-source-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    parsed = parse_simapro(
        args.source, verify_fingerprint=not args.allow_source_mismatch
    )
    flows, activities, locations, leaf_noise, _, target_metadata = (
        load_brightway_targets()
    )
    category_payloads, total_payload, dispositions, noise_families = build_methods(
        parsed, flows, locations, leaf_noise
    )
    reference_signatures = load_ecoinvent_312_reference_signatures()
    summary = validate_generated(
        parsed,
        category_payloads,
        total_payload,
        dispositions,
        flows,
        activities,
        target_metadata,
        noise_families,
        reference_signatures,
    )
    files = output_files(category_payloads, total_payload)

    if args.command == "build":
        write_or_check(files, args.output_dir, check=False)
        args.summary.write_text(render_json(summary), encoding="utf-8")
    elif args.command == "check":
        write_or_check(files, args.output_dir, check=True)
        if not args.summary.is_file() or args.summary.read_text(
            encoding="utf-8"
        ) != render_json(summary):
            raise UBPError(f"Validation summary is stale or missing: {args.summary}")

    print(render_json(console_summary(summary)), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UBPError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
