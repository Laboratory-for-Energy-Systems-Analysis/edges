import pytest
import numpy as np
from edges.uncertainty import sample_cf_distribution


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
