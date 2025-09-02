"""
This module contains the Uncertainty class, which is responsible for handling
"""

import numpy as np
import json
from copy import deepcopy
from scipy import stats
import hashlib

from edges.utils import safe_eval


def get_rng_for_key(key: str, base_seed: int) -> np.random.Generator:
    key_digest = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(base_seed + key_digest)


def make_distribution_key(cf):
    """Generate a hashable cache key for CF uncertainty sampling."""
    unc = cf.get("uncertainty")
    if unc:
        unc_copy = dict(unc)  # shallow copy
        unc_copy.pop("negative", None)  # remove if present
        return json.dumps(unc_copy, sort_keys=True)
    else:
        # No uncertainty block → return None = skip caching
        return None


def _canon_atom(item):
    """
    Return (canonical_object, fingerprint_string) for a mixture atom.
    - If item is a nested distribution dict, remove 'negative', and for
      discrete_empirical recursively canonicalize and sort by fingerprint.
    - If item is a scalar or expression string, return as-is with a const fingerprint.
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
    Generate n random CF values from the distribution info in the 'uncertainty' key.
    Falls back to a constant value if no uncertainty. If 'negative' == 1, samples are negated.
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
            weights = weights / weights.sum() if weights.sum() != 0 else weights

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
            samples = np.full(n, cf["value"], dtype=float)

    except ValueError as e:
        raise ValueError(
            f"Error sampling distribution '{dist_name}' with parameters {params}: {e}"
        )
    return samples
