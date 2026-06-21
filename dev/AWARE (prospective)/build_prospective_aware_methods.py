"""Build prospective AWARE 2.0 country methods for Edges.

The generator uses:
- prospective ecoinvent-country CFs and country weights for deterministic values;
- prospective basin CFs and basin weights for stochastic distributions;
- the AWARE basin GeoPackage plus Natural Earth country polygons to map basins
  to country-level distributions.

Run with the ``edges`` conda environment, which has geopandas/pyogrio installed:

    conda run -n edges python "dev/AWARE (prospective)/build_prospective_aware_methods.py"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "dev" / "AWARE (prospective)"
LEGACY_DIR = ROOT / "dev" / "AWARE" / "AWARE 2.0"
DATA_DIR = ROOT / "edges" / "data"

COUNTRY_AVERAGE_XLSX = (
    SRC_DIR / "CF_ensemblemean_fullyprospective_1GHM_ecoinvent3-10_2019_2049.xlsx"
)
BASIN_CF_XLSX = SRC_DIR / "CF_ensemblemean_fullyprospective_1GHM_basin_2019_2049.xlsx"
BASIN_WEIGHT_XLSX = (
    SRC_DIR / "CFweights_ensemblemean_fullyprospective_1GHM_basin_2019_2049.xlsx"
)
BASIN_GPKG = SRC_DIR / "AWARE20_Native_CFs_geospatial.gpkg"
COUNTRY_CLASSIFICATION_XLSX = LEGACY_DIR / "AWARE20_Countries_and_Regions.xlsx"
NATURAL_EARTH_COUNTRIES = (
    LEGACY_DIR
    / "ne_110m_admin_0_countries"
    / "ne_110m_admin_0_countries.shp"
)

YEARS = ["2019", "2024", "2029", "2034", "2039", "2044", "2049"]
SCENARIOS = ["SSP126", "SSP370", "SSP585"]
BASELINE_SCENARIO = "SSP126"
BASELINE_YEAR = "2019"
AREA_CRS = "ESRI:54009"
MIN_INTERSECTION_AREA_KM2 = 1e-4
METHOD_VERSION = "1.0.0"
PROSPECTIVE_AWARE_REFERENCE = (
    "Reference: Seitfudem, G., Berger, M., and Boulay, A.-M. (2026), "
    "Harnessing model ensembles to assess uncertainty and provide prospective "
    "characterization factors for AWARE2.0."
)

STRATEGIES = [
    "map_exchanges",
    "map_aggregate_locations",
    "map_dynamic_locations",
    "map_contained_locations",
    "map_remaining_locations_to_global",
]

CATEGORY_SPECS = {
    "irri": {
        "label": "irrigation",
        "cf_sheet": "CFs_agri",
        "weight_sheet": "weights_agri",
        "basin_weighting": "annual_agri",
        "basin_weight_sheet": "weight_agri",
        "cpc": ["01"],
    },
    "non_irri": {
        "label": "non-irrigation",
        "cf_sheet": "CFs_nonagri",
        "weight_sheet": "weights_nonagri",
        "basin_weighting": "annual_nonagri",
        "basin_weight_sheet": "weight_nonagri",
        "cpc": ["02", "03", "04", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    },
    "unspecified": {
        "label": "unspecified",
        "cf_sheet": "CFs_unspecified",
        "weight_sheet": "weights_unspecified",
        "basin_weighting": "annual_unspecified",
        "basin_weight_sheet": "weight_unspecified",
        "cpc": None,
    },
}

WATER_FLOWS = [
    (-1, "Water", ("water",)),
    (-1, "Water", ("water", "surface water")),
    (-1, "Water", ("water", "ground-")),
    (1, "Water, cooling, unspecified natural origin", ("natural resource", "in water")),
    (1, "Water, lake", ("natural resource", "in water")),
    (1, "Water, river", ("natural resource", "in water")),
    (
        1,
        "Water, turbine use, unspecified natural origin",
        ("natural resource", "in water"),
    ),
    (1, "Water, unspecified natural origin", ("natural resource", "in water")),
    (1, "Water, unspecified natural origin", ("natural resource", "in ground")),
    (1, "Water, well, in ground", ("natural resource", "in water")),
]


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not text:
        text = "loc"
    if text[0].isdigit():
        text = f"loc_{text}"
    return text


def compact_float(value, digits: int = 12):
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")


def read_country_metadata() -> pd.DataFrame:
    df = pd.read_excel(
        COUNTRY_CLASSIFICATION_XLSX,
        sheet_name="CFs_unspecified",
        keep_default_na=False,
    )
    df = df[df["ecoinvent_collection"] == "countries"].copy()
    df = df.dropna(subset=["ecoinvent_shortname"])
    df["ecoinvent_shortname"] = df["ecoinvent_shortname"].astype(str)
    return df[
        [
            "ecoinvent_shortname",
            "ecoinvent_country_name",
            "GLAM_country_name",
            "GLAM_ISO3",
        ]
    ].drop_duplicates("ecoinvent_shortname")


def make_location_slugs(locations: list[str]) -> dict[str, str]:
    used = {}
    out = {}
    for loc in sorted(locations):
        base = slugify(loc)
        slug = base
        index = 2
        while slug in used and used[slug] != loc:
            slug = f"{base}_{index}"
            index += 1
        used[slug] = loc
        out[loc] = slug
    return out


def read_country_tables(country_locations: set[str]) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for category, spec in CATEGORY_SPECS.items():
        cf = pd.read_excel(
            COUNTRY_AVERAGE_XLSX, sheet_name=spec["cf_sheet"], keep_default_na=False
        )
        wt = pd.read_excel(
            COUNTRY_AVERAGE_XLSX,
            sheet_name=spec["weight_sheet"],
            keep_default_na=False,
        )

        for frame in (cf, wt):
            frame["ecoinvent_shortname"] = frame["ecoinvent_shortname"].astype(str)
            frame["month"] = frame["month"].astype(str)
            frame["scenario"] = frame["scenario"].astype(str)

        cf = cf[
            (cf["month"] == "annual")
            & (cf["ecoinvent_shortname"].isin(country_locations))
        ].copy()
        wt = wt[
            (wt["month"] == "annual")
            & (wt["ecoinvent_shortname"].isin(country_locations))
        ].copy()

        for year in YEARS:
            cf[year] = pd.to_numeric(cf[int(year)], errors="coerce")
            wt[year] = pd.to_numeric(wt[int(year)], errors="coerce")

        result[category] = {
            "cf": cf[["scenario", "ecoinvent_shortname", *YEARS]],
            "weight": wt[["scenario", "ecoinvent_shortname", *YEARS]],
        }
    return result


def read_basin_cf_tables() -> dict[str, pd.DataFrame]:
    raw = pd.read_excel(BASIN_CF_XLSX, sheet_name="annual_CFs")
    raw["scenario"] = raw["scenario"].astype(str)
    raw["weighting"] = raw["weighting"].astype(str)
    raw["Basin_ID"] = raw["Basin_ID"].astype(int)
    for year in YEARS:
        raw[year] = pd.to_numeric(raw[int(year)], errors="coerce")

    out = {}
    for category, spec in CATEGORY_SPECS.items():
        subset = raw[raw["weighting"] == spec["basin_weighting"]].copy()
        out[category] = subset[["scenario", "Basin_ID", *YEARS]]
    return out


def read_basin_weight_tables() -> dict[str, pd.DataFrame]:
    out = {}
    for category, spec in CATEGORY_SPECS.items():
        raw = pd.read_excel(BASIN_WEIGHT_XLSX, sheet_name=spec["basin_weight_sheet"])
        raw["scenario"] = raw["scenario"].astype(str)
        raw["Basin_ID"] = raw["Basin_ID"].astype(int)
        for year in YEARS:
            raw[year] = pd.to_numeric(raw[int(year)], errors="coerce")
        annual = raw.groupby(["scenario", "Basin_ID"], as_index=False)[YEARS].sum()
        out[category] = annual
    return out


def load_country_geometries(country_meta: pd.DataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    ne = gpd.read_file(NATURAL_EARTH_COUNTRIES)
    ne = ne[~ne.geometry.is_empty & ne.geometry.notna()].copy()
    ne["geometry"] = ne.geometry.make_valid()

    by_iso2 = {
        str(row["ISO_A2"]): row.geometry
        for _, row in ne.iterrows()
        if row.get("ISO_A2") not in (None, "-99")
    }
    by_iso2_eh = {
        str(row["ISO_A2_EH"]): row.geometry
        for _, row in ne.iterrows()
        if row.get("ISO_A2_EH") not in (None, "-99")
    }
    by_iso3 = {
        str(row["ISO_A3"]): row.geometry
        for _, row in ne.iterrows()
        if row.get("ISO_A3") not in (None, "-99")
    }
    by_names = {}
    for _, row in ne.iterrows():
        for col in ("ADMIN", "NAME", "NAME_LONG", "SOVEREIGNT", "GEOUNIT"):
            value = row.get(col)
            if value not in (None, ""):
                by_names[str(value).casefold()] = row.geometry

    records = []
    methods = defaultdict(int)
    missing = []
    for row in country_meta.to_dict("records"):
        loc = str(row["ecoinvent_shortname"])
        geom = None
        method = None
        if loc in by_iso2:
            geom = by_iso2[loc]
            method = "ISO_A2"
        elif loc in by_iso2_eh:
            geom = by_iso2_eh[loc]
            method = "ISO_A2_EH"
        else:
            iso3 = row.get("GLAM_ISO3")
            if pd.notna(iso3) and str(iso3) in by_iso3:
                geom = by_iso3[str(iso3)]
                method = "GLAM_ISO3"
            else:
                for name_col in ("ecoinvent_country_name", "GLAM_country_name"):
                    name = row.get(name_col)
                    if pd.notna(name) and str(name).casefold() in by_names:
                        geom = by_names[str(name).casefold()]
                        method = name_col
                        break

        if geom is None:
            missing.append(loc)
            continue
        records.append({"location": loc, "geometry": geom})
        methods[method] += 1

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=ne.crs)
    if not gdf.empty:
        gdf = gdf.dissolve(by="location", as_index=False)
        gdf["geometry"] = gdf.geometry.make_valid()

    summary = {
        "country_geometries": int(len(gdf)),
        "country_geometry_methods": dict(methods),
        "countries_without_geometry": sorted(missing),
    }
    return gdf, summary


def build_basin_country_shares(
    country_gdf: gpd.GeoDataFrame, basin_ids: set[int]
) -> tuple[pd.DataFrame, dict]:
    basins = gpd.read_file(
        BASIN_GPKG, layer="AWARE20_Native_CFs_geospatial", fid_as_index=True
    )
    if "Basin_ID" not in basins.columns:
        basins = basins.reset_index().rename(columns={"fid": "Basin_ID"})
    basins = basins[basins["Basin_ID"].isin(basin_ids)].copy()
    basins = basins[~basins.geometry.is_empty & basins.geometry.notna()].copy()
    basins["Basin_ID"] = basins["Basin_ID"].astype(int)
    basins["geometry"] = basins.geometry.make_valid()

    countries_area = country_gdf.to_crs(AREA_CRS)
    basins_area = basins[["Basin_ID", "geometry"]].to_crs(AREA_CRS)

    intersections = gpd.overlay(
        countries_area[["location", "geometry"]],
        basins_area[["Basin_ID", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    intersections = intersections[
        ~intersections.geometry.is_empty & intersections.geometry.notna()
    ].copy()
    intersections["area_km2"] = intersections.geometry.area / 1e6
    intersections = intersections[
        intersections["area_km2"] >= MIN_INTERSECTION_AREA_KM2
    ].copy()

    total_by_basin = intersections.groupby("Basin_ID")["area_km2"].transform("sum")
    intersections["basin_country_area_share"] = intersections["area_km2"] / total_by_basin

    shares = intersections[
        ["location", "Basin_ID", "area_km2", "basin_country_area_share"]
    ].copy()
    summary = {
        "basins_with_geometry": int(len(basins)),
        "basin_country_intersections": int(len(shares)),
        "countries_with_basin_intersections": int(shares["location"].nunique()),
        "basins_intersecting_countries": int(shares["Basin_ID"].nunique()),
    }
    return shares, summary


def table_to_lookup(df: pd.DataFrame, key_cols: list[str]) -> dict:
    lookup = {}
    for row in df.to_dict("records"):
        key = tuple(row[col] for col in key_cols)
        lookup[key] = {year: compact_float(row[year]) for year in YEARS}
    return lookup


def build_method_data(
    category: str,
    locations: list[str],
    slugs: dict[str, str],
    country_tables: dict[str, dict[str, pd.DataFrame]],
    uncertainty_refs: dict[str, dict],
    include_cpc: bool,
) -> dict:
    spec = CATEGORY_SPECS[category]
    cf_lookup = table_to_lookup(country_tables[category]["cf"], ["scenario", "ecoinvent_shortname"])
    wt_lookup = table_to_lookup(
        country_tables[category]["weight"], ["scenario", "ecoinvent_shortname"]
    )

    valid_locations = []
    for loc in locations:
        if all(
            (scenario, loc) in cf_lookup
            and all(cf_lookup[(scenario, loc)][year] is not None for year in YEARS)
            for scenario in SCENARIOS
        ):
            valid_locations.append(loc)

    parameters = {scenario: {} for scenario in SCENARIOS}
    for loc in valid_locations:
        slug = slugs[loc]
        cf_param = f"cf_{category}_{slug}"
        wt_param = f"wt_{category}_{slug}"
        for scenario in SCENARIOS:
            parameters[scenario][cf_param] = cf_lookup[(scenario, loc)]
            parameters[scenario][wt_param] = wt_lookup.get((scenario, loc), {})

    exchanges = []
    for loc in valid_locations:
        slug = slugs[loc]
        cf_param = f"cf_{category}_{slug}"
        wt_param = f"wt_{category}_{slug}"
        baseline_value = cf_lookup[(BASELINE_SCENARIO, loc)].get(BASELINE_YEAR)
        baseline_value = 0.0 if baseline_value is None else float(baseline_value)
        baseline_weight = wt_lookup.get((BASELINE_SCENARIO, loc), {}).get(BASELINE_YEAR)
        baseline_weight = 0.0 if baseline_weight is None else baseline_weight
        uncertainty_ref = f"{category}__{slug}"
        has_uncertainty = uncertainty_ref in uncertainty_refs

        for sign, name, categories in WATER_FLOWS:
            consumer = {"location": loc, "matrix": "technosphere"}
            if include_cpc and spec["cpc"]:
                consumer["classifications"] = {"CPC": spec["cpc"]}

            exchange = {
                "supplier": {
                    "name": name,
                    "categories": list(categories),
                    "matrix": "biosphere",
                },
                "consumer": consumer,
                "value": baseline_value if sign > 0 else -baseline_value,
                "value_expression": cf_param if sign > 0 else f"-{cf_param}",
                "weight": baseline_weight,
                "weight_expression": wt_param,
            }
            if has_uncertainty:
                exchange["uncertainty_ref"] = uncertainty_ref
                exchange["uncertainty_negative"] = 1 if sign < 0 else 0
            exchanges.append(exchange)

    return {
        "name": f"ecoinvent 3.10/3.11 - AWARE 2.0 prospective_Country_{category}_yearly",
        "unit": "m3 deprived water-eq.",
        "version": METHOD_VERSION,
        "description": (
            "Prospective AWARE 2.0 country-average yearly CFs for SSP126, "
            "SSP370, and SSP585. Country averages and weights come from the "
            "ecoinvent 3.10 aggregation workbook; stochastic distributions use "
            "basin CFs and basin weights allocated to countries by basin-country "
            f"geometry intersections. {PROSPECTIVE_AWARE_REFERENCE}"
        ),
        "strategies": STRATEGIES,
        "parameters": parameters,
        "uncertainties": {
            key: uncertainty_refs[key]
            for key in sorted(uncertainty_refs)
            if key.startswith(f"{category}__")
        },
        "exchanges": exchanges,
    }


def build_uncertainty_refs(
    locations: list[str],
    slugs: dict[str, str],
    country_tables: dict[str, dict[str, pd.DataFrame]],
    basin_cf_tables: dict[str, pd.DataFrame],
    basin_weight_tables: dict[str, pd.DataFrame],
    basin_country_shares: pd.DataFrame,
) -> tuple[dict[str, dict], dict]:
    refs = {}
    validation = []

    shares_by_country = {
        loc: group[["Basin_ID", "basin_country_area_share"]].copy()
        for loc, group in basin_country_shares.groupby("location")
    }

    for category in CATEGORY_SPECS:
        country_cf = table_to_lookup(
            country_tables[category]["cf"], ["scenario", "ecoinvent_shortname"]
        )
        country_weight = table_to_lookup(
            country_tables[category]["weight"], ["scenario", "ecoinvent_shortname"]
        )
        basin_cf = basin_cf_tables[category].set_index(["scenario", "Basin_ID"])
        basin_weight = basin_weight_tables[category].set_index(["scenario", "Basin_ID"])

        for loc in locations:
            if loc not in shares_by_country:
                continue
            slug = slugs[loc]
            ref = f"{category}__{slug}"
            country_basins = shares_by_country[loc]
            values_by_scenario = {}
            weights_by_scenario = {}
            valid_points = 0
            max_relative_error = 0.0

            for scenario in SCENARIOS:
                values_by_year = {}
                weights_by_year = {}
                for year in YEARS:
                    official_cf = country_cf.get((scenario, loc), {}).get(year)
                    official_weight = country_weight.get((scenario, loc), {}).get(year)
                    if official_cf is None:
                        continue

                    values = []
                    weights = []
                    for row in country_basins.itertuples(index=False):
                        key = (scenario, int(row.Basin_ID))
                        if key not in basin_cf.index or key not in basin_weight.index:
                            continue
                        cf_value = compact_float(basin_cf.at[key, year])
                        raw_weight = compact_float(basin_weight.at[key, year])
                        if cf_value is None or raw_weight is None or raw_weight <= 0:
                            continue
                        allocated_weight = raw_weight * float(row.basin_country_area_share)
                        if allocated_weight <= 0:
                            continue
                        values.append(cf_value)
                        weights.append(allocated_weight)

                    if not values:
                        continue

                    weight_sum = sum(weights)
                    if official_weight is not None and official_weight > 0 and weight_sum > 0:
                        scale = official_weight / weight_sum
                        weights = [w * scale for w in weights]

                    weighted_mean = float(np.average(values, weights=weights))
                    denominator = max(abs(float(official_cf)), 1e-12)
                    relative_error = abs(weighted_mean - float(official_cf)) / denominator
                    max_relative_error = max(max_relative_error, relative_error)

                    values_by_year[year] = [compact_float(v) for v in values]
                    weights_by_year[year] = [compact_float(w) for w in weights]
                    valid_points += 1
                    validation.append(
                        {
                            "category": category,
                            "location": loc,
                            "scenario": scenario,
                            "year": year,
                            "n_basins": len(values),
                            "official_cf": compact_float(official_cf),
                            "weighted_basin_mean": compact_float(weighted_mean),
                            "relative_error": compact_float(relative_error, 8),
                        }
                    )

                if values_by_year:
                    values_by_scenario[scenario] = values_by_year
                    weights_by_scenario[scenario] = weights_by_year

            if valid_points:
                refs[ref] = {
                    "distribution": "discrete_empirical",
                    "parameters": {
                        "values_by_scenario": values_by_scenario,
                        "weights_by_scenario": weights_by_scenario,
                    },
                    "metadata": {
                        "location": loc,
                        "category": category,
                        "max_weighted_mean_relative_error": compact_float(
                            max_relative_error, 8
                        ),
                    },
                }

    validation_df = pd.DataFrame(validation)
    if validation_df.empty:
        validation_summary = {
            "distributions": 0,
            "validation_points": 0,
        }
    else:
        validation_summary = {
            "distributions": int(len(refs)),
            "validation_points": int(len(validation_df)),
            "median_relative_error": compact_float(
                validation_df["relative_error"].median(), 8
            ),
            "p95_relative_error": compact_float(
                validation_df["relative_error"].quantile(0.95), 8
            ),
            "max_relative_error": compact_float(
                validation_df["relative_error"].max(), 8
            ),
            "points_above_10pct_error": int((validation_df["relative_error"] > 0.10).sum()),
            "points_above_25pct_error": int((validation_df["relative_error"] > 0.25).sum()),
        }

    return refs, {"summary": validation_summary, "points": validation}


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory where Edges method JSON files are written.",
    )
    parser.add_argument(
        "--write-validation-points",
        action="store_true",
        help="Write detailed per-country/year validation CSV.",
    )
    args = parser.parse_args()

    country_meta = read_country_metadata()
    country_locations = set(country_meta["ecoinvent_shortname"])
    slugs = make_location_slugs(sorted(country_locations))

    country_tables = read_country_tables(country_locations)
    basin_cf_tables = read_basin_cf_tables()
    basin_weight_tables = read_basin_weight_tables()
    basin_ids = set(basin_cf_tables["unspecified"]["Basin_ID"].astype(int))

    country_gdf, geometry_summary = load_country_geometries(country_meta)
    basin_country_shares, overlay_summary = build_basin_country_shares(
        country_gdf, basin_ids
    )
    uncertainty_refs, validation = build_uncertainty_refs(
        sorted(country_locations),
        slugs,
        country_tables,
        basin_cf_tables,
        basin_weight_tables,
        basin_country_shares,
    )

    outputs = {}
    for category in CATEGORY_SPECS:
        data = build_method_data(
            category=category,
            locations=sorted(country_locations),
            slugs=slugs,
            country_tables=country_tables,
            uncertainty_refs=uncertainty_refs,
            include_cpc=False,
        )
        filename = f"AWARE 2.0 prospective_Country_{category}_yearly.json"
        write_json(args.output_dir / filename, data)
        outputs[filename] = {
            "exchanges": len(data["exchanges"]),
            "parameters_per_scenario": {
                scenario: len(params) for scenario, params in data["parameters"].items()
            },
            "uncertainties": len(data["uncertainties"]),
        }

    all_uncertainties = {}
    all_exchanges = []
    all_parameters = {scenario: {} for scenario in SCENARIOS}
    for category in CATEGORY_SPECS:
        data = build_method_data(
            category=category,
            locations=sorted(country_locations),
            slugs=slugs,
            country_tables=country_tables,
            uncertainty_refs=uncertainty_refs,
            include_cpc=True,
        )
        all_exchanges.extend(data["exchanges"])
        all_uncertainties.update(data["uncertainties"])
        for scenario in SCENARIOS:
            all_parameters[scenario].update(data["parameters"][scenario])

    all_data = {
        "name": "ecoinvent 3.10/3.11 - AWARE 2.0 prospective_Country_all_yearly",
        "unit": "m3 deprived water-eq.",
        "version": METHOD_VERSION,
        "description": (
            "Prospective AWARE 2.0 country-average yearly CFs with CPC "
            "discrimination for agricultural, non-agricultural, and unspecified "
            f"water consumption. {PROSPECTIVE_AWARE_REFERENCE}"
        ),
        "strategies": STRATEGIES,
        "parameters": all_parameters,
        "uncertainties": all_uncertainties,
        "exchanges": all_exchanges,
    }
    all_filename = "AWARE 2.0 prospective_Country_all_yearly.json"
    write_json(args.output_dir / all_filename, all_data)
    outputs[all_filename] = {
        "exchanges": len(all_data["exchanges"]),
        "parameters_per_scenario": {
            scenario: len(params) for scenario, params in all_parameters.items()
        },
        "uncertainties": len(all_uncertainties),
    }

    summary = {
        "country_locations": len(country_locations),
        "geometry": geometry_summary,
        "overlay": overlay_summary,
        "validation": validation["summary"],
        "outputs": outputs,
    }
    write_json(SRC_DIR / "prospective_aware_generation_summary.json", summary)

    if args.write_validation_points and validation["points"]:
        pd.DataFrame(validation["points"]).to_csv(
            SRC_DIR / "prospective_aware_validation_points.csv", index=False
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
