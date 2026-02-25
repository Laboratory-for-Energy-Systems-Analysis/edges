from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import bw2data
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import edges
from edges import EdgeLCIA, setup_package_logging


PROJECT_NAME = "bw25_ei310"
ECOINVENT_DB = "ecoinvent-3.10.1-cutoff"
ACTIVITY_NAME = "apple production"
ACTIVITY_LOCATION = "IT"

ROOT = Path(__file__).resolve().parents[2]
PLOT_FILE = ROOT / "examples" / "various examples" / "figure_prospective_gwp100_apple_IT.png"


def get_activity():
    matches = [
        a
        for a in bw2data.Database(ECOINVENT_DB)
        if a["name"] == ACTIVITY_NAME and a.get("location") == ACTIVITY_LOCATION
    ]
    if not matches:
        raise RuntimeError(
            f"Activity not found in '{ECOINVENT_DB}': "
            f"name='{ACTIVITY_NAME}', location='{ACTIVITY_LOCATION}'"
        )
    if len(matches) > 1:
        print(
            f"Found {len(matches)} matching datasets for '{ACTIVITY_NAME}' in "
            f"'{ACTIVITY_LOCATION}'. Using the first one."
        )
    return matches[0]


def select_scenarios(all_scenarios: list[str], per_model: int = 2) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for scen in sorted(all_scenarios):
        model = scen.split("_-_")[0] if "_-_" in scen else scen.split("-")[0]
        grouped[model].append(scen)

    message = sorted(grouped.get("MESSAGE", []))
    if message:
        return message

    # Fallback for datasets without MESSAGE scenarios.
    selected: list[str] = []
    for model in sorted(grouped):
        selected.extend(grouped[model][:per_model])
    return selected[: min(6, len(selected))]


def get_scenario_pathway(scenario: str) -> str:
    if "_-_" in scenario:
        return scenario.split("_-_", 1)[1]
    if "-" in scenario:
        return scenario.split("-", 1)[1]
    return scenario


def get_pathway_family(scenario: str) -> str:
    pathway = get_scenario_pathway(scenario)

    if "rollBack" in pathway or pathway.endswith("-H") or "_H" in pathway:
        return "high forcing"
    if "NPi" in pathway or pathway.endswith("-M") or "_M" in pathway:
        return "medium forcing"
    if "NDC" in pathway or "PkBudg1000" in pathway or "LO" in pathway:
        return "low forcing"
    # Collapse very-low targets into the low-forcing family for plotting.
    if "PkBudg650" in pathway or "VL" in pathway:
        return "low forcing"
    if pathway.endswith("-L") or "_L" in pathway:
        return "low forcing"
    # Default fallback kept inside the requested high/medium/low taxonomy.
    return "medium forcing"


def get_model_name(scenario: str) -> str:
    return scenario.split("_-_")[0] if "_-_" in scenario else scenario.split("-")[0]


def build_results_dataframe(lcia: EdgeLCIA) -> pd.DataFrame:
    scenarios = select_scenarios(list(lcia.parameters.keys()), per_model=2)
    years = sorted(int(y) for y in lcia.parameters[scenarios[0]]["CF_CH4"].keys())
    years = [y for y in years if 2005 <= y <= 2100]

    results = []
    for scenario in scenarios:
        for year in years:
            lcia.evaluate_cfs(scenario=scenario, scenario_idx=str(year))
            lcia.lcia()
            results.append(
                {
                    "Scenario": scenario,
                    "Model": get_model_name(scenario),
                    "Year": year,
                    "GWP100 [kg CO2-eq/unit apple production]": lcia.score,
                }
            )

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame) -> None:
    forcing_colors = {
        "high forcing": "#d73027",
        "medium forcing": "#fc8d59",
        "low forcing": "#1a9850",
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    scenario_handles = {}

    for scenario in sorted(df["Scenario"].unique()):
        data = df[df["Scenario"] == scenario].sort_values("Year")
        forcing = get_pathway_family(scenario)
        color = forcing_colors[forcing]

        (line,) = ax.plot(
            data["Year"],
            data["GWP100 [kg CO2-eq/unit apple production]"],
            color=color,
            linestyle="-",
            linewidth=2,
            label=scenario,
        )
        scenario_handles[scenario] = line

        # Label each curve at its end to make curve identity explicit.
        end = data.iloc[-1]
        ax.text(
            end["Year"] + 0.6,
            end["GWP100 [kg CO2-eq/unit apple production]"],
            get_scenario_pathway(scenario),
            color=color,
            fontsize=7,
            va="center",
        )

    ax.set_title(
        "Prospective GWP100 of apple production (IT)"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(r"GWP100 [kg CO$_2$-eq/unit apple production]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(df["Year"].min(), df["Year"].max() + 8)

    forcing_legend = [
        Line2D([0], [0], color=forcing_colors[f], lw=2, label=f)
        for f in ["high forcing", "medium forcing", "low forcing"]
    ]
    ax.legend(
        handles=forcing_legend,
        title="Forcing level (color)",
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    setup_package_logging()
    bw2data.projects.set_current(PROJECT_NAME)

    if ECOINVENT_DB not in bw2data.databases:
        raise RuntimeError(
            f"Database '{ECOINVENT_DB}' not found in project '{PROJECT_NAME}'."
        )

    act = get_activity()

    method =  ('Prospective', 'GWP100')

    lcia = EdgeLCIA(demand={act: 1}, method=method)
    lcia.lci()
    lcia.map_exchanges()

    df = build_results_dataframe(lcia)
    print("\nCurves shown in the plot:")
    for scen in sorted(df["Scenario"].unique()):
        print(
            f"- {scen}: model={get_model_name(scen)}, "
            f"pathway={get_scenario_pathway(scen)}, family={get_pathway_family(scen)}"
        )
    plot_results(df)
    print(f"Saved plot to: {PLOT_FILE}")


if __name__ == "__main__":
    main()
