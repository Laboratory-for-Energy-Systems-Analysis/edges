"""
Module copied from bw2analyzer.comparisons to allow for custom modifications.
https://docs.brightway.dev/en/legacy/_modules/bw2analyzer/comparisons.html#compare_activities_by_grouped_leaves

"""

from os.path import commonprefix
import pandas as pd
import operator
import tabulate
from bw2analyzer.comparisons import get_value_for_cpc
import bw2data as bd
import numpy as np

from . import EdgeLCIA


def _activity_demand_key(activity):
    return activity.id if hasattr(activity, "id") else activity


def find_leaves(
    activity,
    lcia_method,
    results=None,
    lca_obj=None,
    amount=1,
    total_score=None,
    level=0,
    max_level=3,
    cutoff=2.5e-2,
    cache=None,
):
    """Traverse the supply chain of an activity to find leaves - places where the impact of that
    component falls below a threshold value.

    Returns a list of ``(impact of this activity, amount consumed, Activity instance)`` tuples.
    """
    first_level = results is None

    activity = bd.get_activity(activity)

    k = (lcia_method, activity["database"], activity["code"])

    if first_level:
        level = 0
        results = []

        total_score = lca_obj.score
        cache[k] = lca_obj.score
    else:
        if k not in cache:
            lca_obj.lcia({_activity_demand_key(activity): amount})
            cache[k] = lca_obj.score
            sub_score = lca_obj.score
        else:
            sub_score = cache[k]

        # If this is a leaf, add the leaf and return
        if abs(sub_score) <= abs(total_score * cutoff) or level >= max_level:
            # Only add leaves with scores that matter
            if abs(sub_score) > abs(total_score * 1e-4):
                results.append((sub_score, amount, activity))

            return results, cache

        # Add direct emissions from this demand
        idx = np.argwhere(lca_obj.demand_array)[0][-1]
        direct = (
            lca_obj.characterization_matrix[:, idx].T
            * lca_obj.biosphere_matrix
            * lca_obj.demand_array
        ).sum()
        if abs(direct) >= abs(total_score * 1e-4):
            results.append((direct, amount, activity))

    for exc in activity.technosphere():
        _, cache = find_leaves(
            activity=exc.input,
            lcia_method=lcia_method,
            results=results,
            lca_obj=lca_obj,
            amount=amount * exc["amount"],
            total_score=total_score,
            level=level + 1,
            max_level=max_level,
            cutoff=cutoff,
            cache=cache,
        )

    return sorted(results, reverse=True), cache


def group_leaves(leaves):
    """Group elements in ``leaves`` by their `CPC (Central Product Classification)
    <https://unstats.un.org/unsd/classifications/Econ/cpc>`__ code.

    Returns a list of ``(fraction of total impact, specific impact, amount,
    Activity instance)`` tuples.
    """
    results = {}

    for leaf in leaves:
        cpc = get_isic(leaf[2])
        if cpc not in results:
            results[cpc] = np.zeros((2,))
        results[cpc] += np.array(leaf[:2])

    return sorted([v.tolist() + [k] for k, v in results.items()], reverse=True)


def get_isic(activity):
    try:
        return next(
            cl[1] for cl in activity.get("classifications", []) if "ISIC" in cl[0]
        )
    except StopIteration:
        return


def compare_activities_by_grouped_leaves(
    activities,
    lcia_method,
    mode="relative",
    max_level=4,
    cutoff=7.5e-3,
    output_format="list",
    str_length=50,
    cache=None,
):
    """
    Compare activities by the impact of their different inputs, aggregated
    by the product classification of those inputs.

    Args:
        activities: list of ``Activity`` instances.
        lcia_method: tuple. LCIA method to use when traversing supply chain graph.
        mode: str. If "relative" (default), results are returned as a fraction of total input.
        Otherwise, results are absolute impact per input exchange.
        max_level: int. Maximum level in supply chain to examine.
        cutoff: float. Fraction of total impact to cutoff supply chain graph traversal at.
        output_format: str. See below.
        str_length; int. If ``output_format`` is ``html``, this controls how many
        characters each column label can have.

    Raises:
        ValueError: ``activities`` is malformed.

    Returns:
        Depends on ``output_format``:

        * ``list``: Tuple of ``(column labels, data)``
        * ``html``: HTML string that will print nicely in Jupyter notebooks.
        * ``pandas``: a pandas ``DataFrame``.

    """
    for act in activities:
        if not isinstance(act, bd.backends.proxies.Activity):
            raise ValueError("`activities` must be an iterable of `Activity` instances")

    lca = EdgeLCIA({act: 1 for act in activities}, lcia_method)
    lca.lci(factorize=True)
    lca.lcia()

    objs, cache = [], cache or {}

    for act in activities:
        leaves, cache = find_leaves(
            activity=act,
            lcia_method=lcia_method,
            max_level=max_level,
            cutoff=cutoff,
            lca_obj=lca,
            cache=cache,
        )

        grouped_leaves = group_leaves(leaves)

        objs.append(grouped_leaves)

    sorted_keys = sorted(
        [
            (max([el[0] for obj in objs for el in obj if el[2] == key]), key)
            for key in {el[2] for obj in objs for el in obj}
        ],
        reverse=True,
    )
    name_common = commonprefix([act["name"] for act in activities])

    if " " not in name_common:
        name_common = ""
    else:
        last_space = len(name_common) - operator.indexOf(reversed(name_common), " ")
        name_common = name_common[:last_space]
        print("Omitting activity name common prefix: '{}'".format(name_common))

    product_common = commonprefix(
        [act.get("reference product", "") for act in activities]
    )

    lca = EdgeLCIA({act: 1 for act in activities}, lcia_method)
    lca.lci(factorize=True)
    lca.lcia()

    labels = [
        "activity",
        "product",
        "location",
        "unit",
        "total",
        "direct emissions",
    ] + [key for _, key in sorted_keys]

    data = []
    for act, lst in zip(activities, objs):
        lca.redo_lcia({_activity_demand_key(act): 1})
        idx = np.argwhere(lca.demand_array)[0][-1]
        data.append(
            [
                act["name"].replace(name_common, ""),
                act.get("reference product", "").replace(product_common, ""),
                act.get("location", "")[:25],
                act.get("unit", ""),
                lca.score,
            ]
            + [
                (
                    lca.characterization_matrix[:, idx].T
                    * lca.biosphere_matrix
                    * lca.demand_array
                ).sum()
            ]
            + [get_value_for_cpc(lst, key) for _, key in sorted_keys]
        )

    data.sort(key=lambda x: x[4], reverse=True)

    if mode == "relative":
        for row in data:
            for index, point in enumerate(row[5:]):
                row[index + 5] = point / row[4]

    if output_format == "list":
        return labels, data
    elif output_format == "pandas":
        return pd.DataFrame(data, columns=labels)
    elif output_format == "html":
        return tabulate.tabulate(
            data,
            [x[:str_length] for x in labels],
            tablefmt="html",
            floatfmt=".3f",
        )
