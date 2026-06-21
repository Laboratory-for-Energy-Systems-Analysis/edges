import pytest
import numpy as np
from edges.uncertainty import sample_cf_distribution

INTERPOLATION_POLICY = {
    "axis": "scenario_idx",
    "axis_type": "year",
    "method": "linear",
    "extrapolation": "nearest",
    "source_years": ["2024", "2029"],
}


def test_sample_constant_cf():
    cf = {"value": 42}
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=100, random_state=random_state
    )
    assert np.all(samples == 42)


def test_sample_expression_cf():
    cf = {"value": "A * 2"}
    parameters = {"A": 5}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=100, random_state=random_state
    )
    assert np.all(samples == 10)


def test_sample_cf_prefers_value_expression_over_baseline_value():
    cf = {"value": 1.0, "value_expression": "A * 2"}
    parameters = {"A": 5}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=100, random_state=random_state
    )
    assert np.all(samples == 10)


def test_sample_uniform_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "uniform",
            "parameters": {"minimum": 2, "maximum": 5},
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert samples.shape == (1000,)
    assert np.all(samples >= 2)
    assert np.all(samples <= 5)


def test_sample_normal_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "normal",
            "parameters": {"loc": 5, "scale": 1, "minimum": 0, "maximum": 10},
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert samples.shape == (1000,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)
    assert abs(np.mean(samples) - 5) < 0.3


def test_sample_triangular_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "triang",
            "parameters": {"minimum": 1, "loc": 3, "maximum": 5},
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert np.all(samples >= 1)
    assert np.all(samples <= 5)
    assert abs(np.mean(samples) - 3) < 0.3


def test_sample_log_normal_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "lognorm",
            "parameters": {
                "shape_a": 0.5,
                "loc": 0,
                "scale": 1,
                "minimum": 0,
                "maximum": 10,
            },
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)


def test_sample_beta_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "beta",
            "parameters": {
                "shape_a": 2,
                "shape_b": 5,
                "loc": 0,
                "scale": 1,
                "minimum": 0,
                "maximum": 1,
            },
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_sample_gamma_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "gamma",
            "parameters": {
                "shape_a": 2,
                "scale": 1,
                "loc": 0,
                "minimum": 0,
                "maximum": 10,
            },
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)


def test_sample_weibull_distribution():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "weibull_min",
            "parameters": {
                "shape_a": 1.5,
                "loc": 0,
                "scale": 2,
                "minimum": 0,
                "maximum": 10,
            },
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=1000, random_state=random_state
    )
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)


def test_fallback_to_constant_on_unknown_distribution():
    cf = {
        "value": 7.5,
        "uncertainty": {"distribution": "unknown_dist", "parameters": {}},
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=100, random_state=random_state
    )
    assert np.allclose(samples, 7.5)


def test_sample_discrete_empirical_distribution_by_scenario_and_year():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP126": {"2019": [1.0, 2.0], "2024": [10.0, 20.0]},
                    "SSP585": {"2019": [3.0, 4.0], "2024": [30.0, 40.0]},
                },
                "weights_by_scenario": {
                    "SSP126": {"2019": [1.0, 0.0], "2024": [0.0, 1.0]},
                    "SSP585": {"2019": [1.0, 0.0], "2024": [0.0, 1.0]},
                },
            },
        },
    }
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf,
        parameters={},
        n=100,
        random_state=random_state,
        scenario_name="SSP585",
        scenario_idx="2024",
    )
    assert np.allclose(samples, 40.0)


def test_sample_discrete_empirical_interpolates_scenario_year_arrays():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP585": {"2024": [10.0, 20.0], "2029": [20.0, 40.0]},
                },
                "weights_by_scenario": {
                    "SSP585": {"2024": [1.0, 0.0], "2029": [0.0, 1.0]},
                },
            },
        },
    }
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf,
        parameters={},
        n=100,
        random_state=random_state,
        scenario_name="SSP585",
        scenario_idx="2026.5",
        interpolation_policy=INTERPOLATION_POLICY,
    )
    assert set(np.unique(samples)).issubset({15.0, 30.0})
    assert 0 < np.mean(samples) < 30.0


def test_sample_discrete_empirical_interpolates_by_basin_id_when_order_changes():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP585": {"2024": [10.0, 20.0], "2029": [40.0, 20.0]},
                },
                "weights_by_scenario": {
                    "SSP585": {"2024": [1.0, 0.0], "2029": [0.0, 1.0]},
                },
                "ids_by_scenario": {
                    "SSP585": {"2024": [1, 2], "2029": [2, 1]},
                },
            },
        },
    }
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf,
        parameters={},
        n=100,
        random_state=random_state,
        scenario_name="SSP585",
        scenario_idx="2026.5",
        interpolation_policy=INTERPOLATION_POLICY,
    )
    assert np.allclose(samples, 15.0)


def test_sample_discrete_empirical_clamps_year_outside_available_range():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP585": {"2024": [10.0], "2029": [20.0]},
                },
                "weights_by_scenario": {
                    "SSP585": {"2024": [1.0], "2029": [1.0]},
                },
            },
        },
    }
    random_state = np.random.default_rng(42)
    below = sample_cf_distribution(
        cf,
        parameters={},
        n=10,
        random_state=random_state,
        scenario_name="SSP585",
        scenario_idx="2010",
        interpolation_policy=INTERPOLATION_POLICY,
    )
    above = sample_cf_distribution(
        cf,
        parameters={},
        n=10,
        random_state=random_state,
        scenario_name="SSP585",
        scenario_idx="2050",
        interpolation_policy=INTERPOLATION_POLICY,
    )
    assert np.allclose(below, 10.0)
    assert np.allclose(above, 20.0)


def test_sample_discrete_empirical_keeps_legacy_fallback_without_policy():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP585": {"2024": [10.0], "2029": [20.0]},
                },
                "weights_by_scenario": {
                    "SSP585": {"2024": [1.0], "2029": [1.0]},
                },
            },
        },
    }

    samples = sample_cf_distribution(
        cf,
        parameters={},
        n=10,
        random_state=np.random.default_rng(42),
        scenario_name="SSP585",
        scenario_idx="2026",
    )

    assert np.allclose(samples, 20.0)


def test_sample_discrete_empirical_uses_same_fallback_scenario_for_weights():
    cf = {
        "value": 0,
        "uncertainty": {
            "distribution": "discrete_empirical",
            "parameters": {
                "values_by_scenario": {
                    "SSP126": {"2024": [10.0, 20.0]},
                    "SSP585": {"2024": [30.0, 40.0]},
                },
                "weights_by_scenario": {
                    "SSP585": {"2024": [0.0, 1.0]},
                    "SSP126": {"2024": [1.0, 0.0]},
                },
            },
        },
    }

    samples = sample_cf_distribution(
        cf,
        parameters={},
        n=20,
        random_state=np.random.default_rng(42),
        scenario_name="MISSING",
        scenario_idx="2024",
    )

    assert np.allclose(samples, 10.0)


def test_log_normal_distribution_avoids_boundary_pile_up():
    cf = {
        "value": 51.1,
        "uncertainty": {
            "distribution": "lognorm",
            "parameters": {
                "shape_a": 0.867,
                "loc": 0.1,
                "scale": 38.276,
                "minimum": 0.1,
                "maximum": 100.0,
            },
        },
    }
    parameters = {}
    random_state = np.random.default_rng(42)
    samples = sample_cf_distribution(
        cf, parameters=parameters, n=5000, random_state=random_state
    )
    assert np.all(samples >= 0.1)
    assert np.all(samples <= 100.0)
    assert np.count_nonzero(samples == 100.0) == 0
