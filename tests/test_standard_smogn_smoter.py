from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "renvsa_experiments"
    / "run_smogn_smoter_baselines.py"
)
MODULE_DIR = str(MODULE_PATH.parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
SPEC = importlib.util.spec_from_file_location("run_smogn_smoter_baselines", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_budget_matches_rensa_effective_budget() -> None:
    assert len(module.SEARCH_CONFIGS) == len(module.base.effective_rensa_configs()) == 6
    assert module.PIPELINE_CANDIDATES_PER_METHOD == 36
    assert module.SVR_GRID_SIZE == 36
    assert {config.relevance_threshold for config in module.SEARCH_CONFIGS} == {0.4, 0.5, 0.6}
    assert {config.k for config in module.SEARCH_CONFIGS} == {3, 5}
    assert {config.sampling_strategy for config in module.SEARCH_CONFIGS} == {"balance", "extreme"}
    assert {config.perturbation for config in module.SEARCH_CONFIGS} == {0.01, 0.02, 0.05}


def test_relevance_control_points_are_train_only_quantiles() -> None:
    y = np.linspace(-5.0, 8.0, 101)
    points = np.asarray(module.relevance_control_points(y))
    assert points.shape == (5, 3)
    assert np.allclose(points[:, 0], np.quantile(y, module.RELEVANCE_CONTROL_QUANTILES))
    assert np.array_equal(points[:, 1], np.asarray([1.0, 1.0, 0.0, 1.0, 1.0]))


def test_audited_standard_resamplers_are_finite_and_deterministic() -> None:
    module.verify_environment()
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 12))
    y = np.linspace(-3.0, 3.0, 30) + rng.normal(scale=0.02, size=30)
    for method in module.METHODS:
        result = module.resample_standard(method, X, y, module.SEARCH_CONFIGS[0])
        repeated = module.resample_standard(method, X, y, module.SEARCH_CONFIGS[0])
        assert result.X.shape[1] == X.shape[1]
        assert result.X.shape[0] == result.y.size == result.n_output
        assert np.isfinite(result.X).all()
        assert np.isfinite(result.y).all()
        assert np.array_equal(result.X, repeated.X)
        assert np.array_equal(result.y, repeated.y)


def test_ubl_balance_sampling_factors_and_counts() -> None:
    numeric = module.numeric_standard
    factors = numeric.sampling_factors(12, np.asarray([2, 8, 2]), "balance")
    assert np.array_equal(factors, np.asarray([2.0, 0.5, 2.0]))


def test_synthetic_predictors_do_not_depend_on_target_values() -> None:
    numeric = module.numeric_standard
    X = np.asarray([[0.0, 0.0], [0.2, 0.1], [2.0, 2.2], [2.2, 2.0]])
    anchors = np.asarray([0, 1, 2, 3])
    y_a = np.asarray([-5.0, -1.0, 1.0, 5.0])
    y_b = np.asarray([500.0, -100.0, 20.0, -700.0])
    out_a = numeric._generate_synthetic(
        "smoter", X, y_a, anchors, 2, 0.02, np.random.default_rng(19)
    )
    out_b = numeric._generate_synthetic(
        "smoter", X, y_b, anchors, 2, 0.02, np.random.default_rng(19)
    )
    assert np.array_equal(out_a[0], out_b[0])
