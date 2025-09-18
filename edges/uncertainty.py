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
    unc = cf.get("uncertainty")
    if unc:
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


def sample_cf_distribution(
    cf: dict,
    n: int,
    parameters: dict,
    random_state: np.random._generator.Generator,
    use_distributions: bool = True,
    SAFE_GLOBALS: dict = None,
) -> np.ndarray:
    """
    Draw samples from the CF's uncertainty distribution (or constant fallback).

    If no uncertainty or distributions are disabled, returns a length-``n`` array
    filled with the (possibly evaluated) deterministic CF value.

    :param cf: CF dictionary with 'value' and optional 'uncertainty' specification.
    :param n: Number of samples to generate.
    :param parameters: Parameter dict for evaluating expression atoms.
    :param random_state: RNG to use for sampling.
    :param use_distributions: If False, bypass uncertainty and return constants.
    :param SAFE_GLOBALS: Safe globals for expression evaluation.
    :return: NumPy array of shape (n,) with sampled CF values.

    :raises ValueError: If sampling fails due to invalid distribution parameters.
    """
    if not use_distributions or cf.get("uncertainty") is None:
        # If value is a string (expression), evaluate once
        value = cf["value"]
        if isinstance(value, str):
            value = safe_eval(
                expr=value,
                parameters=parameters,
                scenario_idx=0,
                SAFE_GLOBALS=SAFE_GLOBALS,
            )
        return np.full(n, value, dtype=float)

    unc = cf["uncertainty"]
    dist_name = unc["distribution"]
    params = unc["parameters"]

    try:
        if dist_name == "discrete_empirical":
            values = params["values"]
            weights = np.array(params["weights"])
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
                    )[0]
                else:
                    if isinstance(item, str):
                        # evaluate deterministic expression atom
                        item = safe_eval(
                            expr=item,
                            parameters=parameters,
                            scenario_idx=0,
                            SAFE_GLOBALS=SAFE_GLOBALS,
                        )
                    samples[i] = item

        elif dist_name == "uniform":
            samples = random_state.uniform(params["minimum"], params["maximum"], size=n)

        elif dist_name == "triang":
            left = params["minimum"]
            mode = params["loc"]
            right = params["maximum"]
            samples = random_state.triangular(left, mode, right, size=n)

        elif dist_name == "normal":
            samples = random_state.normal(
                loc=params["loc"], scale=params["scale"], size=n
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "lognorm":
            s = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = stats.lognorm.rvs(
                s=s, loc=loc, scale=scale, size=n, random_state=random_state
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "beta":
            a = params["shape_a"]
            b = params["shape_b"]
            x = random_state.beta(a, b, size=n)
            samples = params["loc"] + x * params["scale"]
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "gamma":
            samples = (
                random_state.gamma(params["shape_a"], params["scale"], size=n)
                + params["loc"]
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "weibull_min":
            c = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = stats.weibull_min.rvs(
                c=c, loc=loc, scale=scale, size=n, random_state=random_state
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        else:
            logger.warning(
                "Unknown distribution '%s'; falling back to constant value.", dist_name
            )
            samples = np.full(n, cf["value"], dtype=float)

    except ValueError as e:
        logger.error(
            "Error sampling distribution '%s' with parameters %s", dist_name, params
        )
        raise ValueError(
            f"Error sampling distribution '{dist_name}' with parameters {params}: {e}"
        )
    return samples
