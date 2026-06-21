"""
Utilities for uncertainty handling: RNG derivation, cache keys, canonicalization,
and sampling of characterization-factor (CF) uncertainty distributions.
"""

import numpy as np
import json
from copy import deepcopy
from scipy import stats
import hashlib
import logging

from edges.utils import safe_eval

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_rng_for_key(key: str, base_seed: int) -> np.random.Generator:
    """
    Derive a reproducible RNG from a base seed and a string key.

    :param key: Arbitrary identifier (e.g., CF uncertainty fingerprint).
    :param base_seed: Base integer seed.
    :return: Numpy random.Generator instance initialized from derived seed.
    """
    key_digest = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    seed = base_seed + key_digest
    logger.debug("Creating RNG with derived seed %d for key %s", seed, key)
    return np.random.default_rng(seed)


def make_distribution_key(cf):
    """
    Generate a stable, hashable cache key for a CF's uncertainty block.

    :param cf: CF dictionary potentially containing an 'uncertainty' entry.
    :return: JSON string key without 'negative' flag, or None if no uncertainty.
    """
    unc_ref = cf.get("uncertainty_ref")
    if unc_ref is not None:
        return f"ref:{unc_ref}"

    unc = cf.get("uncertainty")
    if unc:
        if unc.get("ref") is not None:
            return f"ref:{unc['ref']}"
        unc_copy = dict(unc)  # shallow copy
        unc_copy.pop("negative", None)  # remove if present
        return json.dumps(unc_copy, sort_keys=True)
    else:
        # No uncertainty block → return None = skip caching
        logger.debug("No uncertainty block present; skipping cache key.")
        return None


def _canon_atom(item):
    """
    Canonicalize a mixture atom and return its fingerprint.

    - If ``item`` is a distribution dict, remove 'negative'; for
      ``discrete_empirical`` recursively canonicalize children and merge duplicates.
    - If scalar or expression string, return as-is.

    :param item: Scalar, expression string, or distribution dict.
    :return: Tuple (canonical_object, fingerprint_string).
    """
    if isinstance(item, dict) and "distribution" in item:
        clean = deepcopy(item)
        clean.pop("negative", None)
        dist = clean.get("distribution")
        params = clean.get("parameters", {})

        if dist == "discrete_empirical":
            vals = list(params.get("values", []))
            wts = list(params.get("weights", []))
            canon_pairs = []
            for v, w in zip(vals, wts):
                v_clean, v_fp = _canon_atom(v)
                canon_pairs.append((v_fp, float(w), v_clean))

            # merge equal atoms (same fingerprint) and normalize; sort by fp
            merged = {}
            obj_for = {}
            for fp, w, v_clean in canon_pairs:
                merged[fp] = merged.get(fp, 0.0) + float(w)
                obj_for[fp] = v_clean
            tot = sum(merged.values()) or 1.0
            items = sorted(merged.items(), key=lambda kv: kv[0])
            clean["parameters"] = {
                "values": [obj_for[fp] for fp, _ in items],
                "weights": [w / tot for _, w in items],
            }

        # canonical fingerprint: JSON with sorted keys
        fp = json.dumps(clean, sort_keys=True)
        return clean, f"dist:{fp}"

    # scalar or expression
    if isinstance(item, str):
        val_repr = item.strip()
    else:
        val_repr = item
    fp = f"const:{repr(val_repr)}"
    return val_repr, fp


def _sample_bounded_continuous_distribution(draw_fn, n, minimum, maximum):
    """
    Draw ``n`` samples within inclusive bounds via rejection sampling.

    This preserves the shape of the distribution inside the valid interval and
    avoids artificially stacking probability mass at the boundaries, which
    happens when simply clipping out-of-range draws.

    :param draw_fn: Callable accepting ``size=...`` and returning raw samples.
    :param n: Number of accepted samples to return.
    :param minimum: Inclusive lower bound.
    :param maximum: Inclusive upper bound.
    :return: NumPy array of shape ``(n,)``.

    :raises ValueError: If valid samples cannot be obtained after repeated tries.
    """
    if minimum > maximum:
        raise ValueError(
            f"Invalid bounds for bounded sampling: minimum={minimum} > maximum={maximum}"
        )

    accepted = []
    remaining = n
    batch_size = max(remaining, 128)
    attempts = 0

    while remaining > 0:
        attempts += 1
        if attempts > 100:
            raise ValueError(
                "Unable to obtain enough in-bounds samples after repeated draws; "
                f"minimum={minimum}, maximum={maximum}, remaining={remaining}"
            )

        draws = np.asarray(draw_fn(size=batch_size), dtype=float)
        valid = draws[(draws >= minimum) & (draws <= maximum)]

        if valid.size == 0:
            batch_size *= 2
            continue

        take = min(remaining, valid.size)
        accepted.append(valid[:take])
        remaining -= take

        if remaining:
            acceptance_rate = valid.size / draws.size
            batch_size = max(int(np.ceil(remaining / acceptance_rate * 1.2)), 128)

    return np.concatenate(accepted)


def sample_cf_distribution(
    cf: dict,
    n: int,
    parameters: dict,
    random_state: np.random._generator.Generator,
    use_distributions: bool = True,
    SAFE_GLOBALS: dict = None,
    scenario_idx: int | str = 0,
    scenario_name: str | None = None,
) -> np.ndarray:
    """
    Draw samples from the CF's uncertainty distribution (or constant fallback).

    If no uncertainty or distributions are disabled, returns a length-``n`` array
    filled with the deterministic CF value. ``value_expression`` is preferred
    over ``value`` when present, allowing methods to keep a numeric baseline
    value while evaluating scenario-dependent values at runtime.

    ``discrete_empirical`` distributions can define ``values_by_scenario`` and
    ``weights_by_scenario`` mappings of ``scenario -> year/index -> sequence``.
    These scenario/year-specific arrays are selected using ``scenario_name`` and
    ``scenario_idx``.

    :param cf: CF dictionary with 'value' and optional 'uncertainty' specification.
    :param n: Number of samples to generate.
    :param parameters: Parameter dict for evaluating expression atoms.
    :param random_state: RNG to use for sampling.
    :param use_distributions: If False, bypass uncertainty and return constants.
    :param SAFE_GLOBALS: Safe globals for expression evaluation.
    :param scenario_idx: Scenario year/index used for scenario-aware uncertainty data.
    :param scenario_name: Optional scenario name used for scenario-aware uncertainty data.
    :return: NumPy array of shape (n,) with sampled CF values.

    :raises ValueError: If sampling fails due to invalid distribution parameters.
    """

    def _eval_atom(item):
        if isinstance(item, str):
            expr = item.strip()
            if expr in parameters:
                return parameters[expr]
            if len(expr) > 1 and expr[0] in "+-" and expr[1:] in parameters:
                value = float(parameters[expr[1:]])
                return -value if expr[0] == "-" else value
            return float(
                safe_eval(
                    expr=expr,
                    parameters=parameters,
                    scenario_idx=scenario_idx,
                    SAFE_GLOBALS=SAFE_GLOBALS,
                )
            )
        return item

    def _deterministic_value() -> float:
        value = cf.get("value_expression")
        if value in (None, ""):
            value = cf["value"]
        return float(_eval_atom(value))

    if not use_distributions or cf.get("uncertainty") is None:
        return np.full(n, _deterministic_value(), dtype=float)

    unc = cf["uncertainty"]
    dist_name = unc["distribution"]
    params = unc["parameters"]

    def _select_scenario_values(base_name: str):
        """Select scenario/year-specific distribution arrays when available."""
        by_scenario = params.get(f"{base_name}_by_scenario")
        if not isinstance(by_scenario, dict):
            return None

        selected_name = scenario_name
        if selected_name not in by_scenario:
            selected_name = next(iter(by_scenario), None)
        if selected_name is None:
            return None

        by_index = by_scenario.get(selected_name)
        if not isinstance(by_index, dict):
            return by_index

        idx = str(scenario_idx)
        if idx in by_index:
            return by_index[idx]
        if by_index:
            return list(by_index.values())[-1]
        return None

    try:
        if dist_name == "discrete_empirical":
            values = _select_scenario_values("values")
            if values is None:
                values = params["values"]

            raw_weights = _select_scenario_values("weights")
            if raw_weights is None:
                raw_weights = params["weights"]

            weights = np.array([_eval_atom(w) for w in raw_weights], dtype=float)
            if weights.sum() == 0:
                logger.warning(
                    "All weights are zero in discrete_empirical; using equal weights."
                )
                weights = np.ones_like(weights, dtype=float) / len(weights)
            else:
                weights = weights / weights.sum()

            chosen_indices = random_state.choice(len(values), size=n, p=weights)

            samples = np.empty(n)

            for i, idx in enumerate(chosen_indices):
                item = values[idx]
                if isinstance(item, dict) and "distribution" in item:
                    # Recursively sample this distribution
                    samples[i] = sample_cf_distribution(
                        cf={"value": 0, "uncertainty": item},
                        n=1,
                        parameters=parameters,
                        random_state=random_state,
                        use_distributions=use_distributions,
                        SAFE_GLOBALS=SAFE_GLOBALS,
                        scenario_idx=scenario_idx,
                        scenario_name=scenario_name,
                    )[0]
                else:
                    samples[i] = _eval_atom(item)

        elif dist_name == "uniform":
            samples = random_state.uniform(params["minimum"], params["maximum"], size=n)

        elif dist_name == "triang":
            left = params["minimum"]
            mode = params["loc"]
            right = params["maximum"]
            samples = random_state.triangular(left, mode, right, size=n)

        elif dist_name == "normal":
            samples = _sample_bounded_continuous_distribution(
                draw_fn=lambda size: random_state.normal(
                    loc=params["loc"], scale=params["scale"], size=size
                ),
                n=n,
                minimum=params["minimum"],
                maximum=params["maximum"],
            )

        elif dist_name == "lognorm":
            s = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = _sample_bounded_continuous_distribution(
                draw_fn=lambda size: stats.lognorm.rvs(
                    s=s, loc=loc, scale=scale, size=size, random_state=random_state
                ),
                n=n,
                minimum=params["minimum"],
                maximum=params["maximum"],
            )

        elif dist_name == "beta":
            a = params["shape_a"]
            b = params["shape_b"]
            samples = _sample_bounded_continuous_distribution(
                draw_fn=lambda size: params["loc"]
                + random_state.beta(a, b, size=size) * params["scale"],
                n=n,
                minimum=params["minimum"],
                maximum=params["maximum"],
            )

        elif dist_name == "gamma":
            samples = _sample_bounded_continuous_distribution(
                draw_fn=lambda size: random_state.gamma(
                    params["shape_a"], params["scale"], size=size
                )
                + params["loc"],
                n=n,
                minimum=params["minimum"],
                maximum=params["maximum"],
            )

        elif dist_name == "weibull_min":
            c = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = _sample_bounded_continuous_distribution(
                draw_fn=lambda size: stats.weibull_min.rvs(
                    c=c, loc=loc, scale=scale, size=size, random_state=random_state
                ),
                n=n,
                minimum=params["minimum"],
                maximum=params["maximum"],
            )

        else:
            logger.warning(
                "Unknown distribution '%s'; falling back to constant value.", dist_name
            )
            samples = np.full(n, _deterministic_value(), dtype=float)

    except ValueError as e:
        logger.error(
            "Error sampling distribution '%s' with parameters %s", dist_name, params
        )
        raise ValueError(
            f"Error sampling distribution '{dist_name}' with parameters {params}: {e}"
        )
    return samples
