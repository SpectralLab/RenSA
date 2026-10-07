from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from respond_spectra import (
    CARSFeatureSelector,
    CorrelationIntervalSelector,
    ResponseDrivenAugmenter,
    compare_regressors_with_train_augmentation,
    default_tuned_tolerant_regressors,
    evaluate_plsr_cv_with_train_augmentation,
    load_spectrum_csv,
    msc,
    snv,
)


def _toy_data(seed: int = 3):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 80)
    y = np.linspace(1.0, 3.0, 16)
    X = []
    for value in y:
        peak = np.exp(-0.5 * ((grid - 0.45) / 0.09) ** 2)
        shoulder = np.exp(-0.5 * ((grid - 0.72) / 0.13) ** 2)
        X.append(peak * value + shoulder * (3.4 - value) + rng.normal(0.0, 0.01, grid.size))
    return np.asarray(X), y


def test_snv_row_standardizes_spectra():
    X, _ = _toy_data()
    out = snv(X)
    assert np.allclose(out.mean(axis=1), 0.0)
    assert np.allclose(out.std(axis=1, ddof=1), 1.0)


def test_msc_preserves_shape():
    X, _ = _toy_data()
    out = msc(X)
    assert out.shape == X.shape
    assert np.isfinite(out).all()


def test_response_driven_augmenter_adds_requested_samples():
    X, y = _toy_data()
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=20,
        response_bins=4,
        neighbors=4,
        noise_scale=0.0,
        random_state=11,
    )
    result = augmenter.fit_resample(X, y)
    assert result.n_original == X.shape[0]
    assert result.n_synthetic == 20
    assert result.X.shape == (X.shape[0] + 20, X.shape[1])
    assert result.y.shape == (y.shape[0] + 20,)
    assert np.all(result.y[result.synthetic_mask] >= y.min())
    assert np.all(result.y[result.synthetic_mask] <= y.max())


def test_load_coal_heating_value_csv():
    if not Path("data/coal_Q.csv").is_file():
        pytest.skip("coal_Q.csv is not distributed with the code-only release")
    wavelengths, X, y = load_spectrum_csv("data/coal_Q.csv")
    assert wavelengths.shape == (1500,)
    assert X.shape == (133, 1500)
    assert y.shape == (133,)
    assert np.allclose(wavelengths[:3], [1.0, 2.0, 3.0])
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_load_spectrum_csv_supports_custom_prefix(tmp_path):
    path = tmp_path / "spectra.csv"
    path.write_text(
        "NIR_2,NIR_1,octane\n"
        "0.2,0.1,85.3\n"
        "0.4,0.3,87.9\n"
    )
    wavelengths, X, y = load_spectrum_csv(path, target_column="octane", spectral_prefix="NIR_")
    assert np.allclose(wavelengths, [1.0, 2.0])
    assert np.allclose(X, [[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(y, [85.3, 87.9])


def test_load_spectrum_csv_supports_headerless_last_column_target(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text(
        "0.2,0.1,85.3\n"
        "0.4,0.3,87.9\n"
    )
    wavelengths, X, y = load_spectrum_csv(path)
    assert np.allclose(wavelengths, [1.0, 2.0])
    assert np.allclose(X, [[0.2, 0.1], [0.4, 0.3]])
    assert np.allclose(y, [85.3, 87.9])


def test_fold_augmentation_evaluates_on_original_validation_samples():
    X, y = _toy_data()
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=8,
        response_bins=4,
        neighbors=4,
        noise_scale=0.0,
        random_state=13,
    )
    result = evaluate_plsr_cv_with_train_augmentation(
        X,
        y,
        augmenter,
        max_components=3,
        cv=4,
    )
    assert result["best"]["n_components"] in {1, 2, 3}
    assert len(result["folds"]) == 4
    assert sum(fold["validation_original"] for fold in result["folds"]) == X.shape[0]
    assert all(fold["synthetic_accepted"] == 8 for fold in result["folds"])


def test_tolerant_regressor_comparison_reports_deltas():
    from sklearn.ensemble import ExtraTreesRegressor

    X, y = _toy_data()
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=6,
        response_bins=4,
        neighbors=4,
        noise_scale=0.0,
        random_state=17,
    )
    result = compare_regressors_with_train_augmentation(
        X,
        y,
        augmenter,
        transformer=CorrelationIntervalSelector(n_intervals=8, n_select=3),
        regressors={
            "extra_trees": ExtraTreesRegressor(
                n_estimators=20,
                max_features="sqrt",
                random_state=17,
            )
        },
        cv=4,
    )
    assert set(result) == {"extra_trees"}
    assert {"baseline", "fold_train_augmented", "delta_rmse", "delta_r2"} <= set(
        result["extra_trees"]
    )
    assert len(result["extra_trees"]["fold_train_augmented"]["folds"]) == 4


def test_cars_feature_selector_reduces_variables():
    X, y = _toy_data()
    selector = CARSFeatureSelector(
        n_sampling=5,
        min_features=6,
        max_components=3,
        cv=3,
        random_state=19,
    )
    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape[0] == X.shape[0]
    assert 6 <= X_selected.shape[1] <= X.shape[1]
    assert selector.rmse_path_.shape == (5,)
    assert np.all(np.diff(selector.selected_indices_) > 0)


def test_tuned_tolerant_regressors_are_grid_searches():
    models = default_tuned_tolerant_regressors(random_state=7, inner_cv=2)
    assert set(models) == {"extra_trees_tuned", "random_forest_tuned", "rbf_svr_tuned"}
    assert all(hasattr(model, "param_grid") for model in models.values())


def test_shallow_1d_cnn_regressor_smoke():
    if os.environ.get("RESPOND_SPECTRA_RUN_TORCH_TESTS") != "1":
        pytest.skip("Set RESPOND_SPECTRA_RUN_TORCH_TESTS=1 to run PyTorch smoke tests.")

    from respond_spectra.deep import Shallow1DCNNRegressor

    X, y = _toy_data()
    model = Shallow1DCNNRegressor(
        epochs=2,
        batch_size=8,
        validation_fraction=0.0,
        patience=2,
        random_state=23,
        device="cpu",
    )
    model.fit(X, y)
    pred = model.predict(X[:3])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()
