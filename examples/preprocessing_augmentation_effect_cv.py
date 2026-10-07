from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing_sensitivity_cv import (
    PREPROCESS_METHODS,
    Endpoint,
    endpoints_for_repo,
    fit_spectral_preprocessor,
    load_endpoint,
)
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter


SUMMARY_FIELDS = [
    "dataset",
    "target",
    "n_samples",
    "n_variables",
    "preprocess",
    "model",
    "cv_folds",
    "inner_cv",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "baseline_rmse",
    "augmented_rmse",
    "delta_rmse",
    "baseline_r2",
    "augmented_r2",
    "delta_r2",
    "delta_rmse_95ci",
    "delta_r2_95ci",
    "mean_selected_features",
    "mean_synthetic_accepted",
    "mean_acceptance_rate",
    "is_rmse_improved",
    "is_r2_improved",
    "svr_grid_size",
]

FOLD_FIELDS = [
    "dataset",
    "target",
    "preprocess",
    "fold",
    "train_size",
    "test_size",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "baseline_rmse",
    "augmented_rmse",
    "delta_rmse",
    "baseline_r2",
    "augmented_r2",
    "delta_r2",
    "selected_features",
    "synthetic_accepted",
    "acceptance_rate",
    "baseline_best_params",
    "augmented_best_params",
]


@dataclass(frozen=True)
class AugSpec:
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated ReNVSA effectiveness under different spectral preprocessing methods "
            "for the five main endpoints."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "publication_solid")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "table_preprocessing_augmentation_effect_cv_summary.csv"
    folds_path = args.output_dir / "table_preprocessing_augmentation_effect_cv_folds.csv"

    endpoints = endpoints_for_repo(Path(__file__).resolve().parents[1])
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    for endpoint in endpoints:
        X, y = load_endpoint(endpoint)
        spec = augmentation_specs()[endpoint.dataset]
        print(
            "{}: {} samples, {} variables, ReNVSA n={} {}".format(
                endpoint.dataset,
                X.shape[0],
                X.shape[1],
                spec.n_synthetic,
                spec.perturbation_mode,
            ),
            flush=True,
        )
        for preprocess in PREPROCESS_METHODS:
            row, rows = evaluate_preprocess_augmentation(endpoint, X, y, preprocess, spec, args, param_grid)
            summary_rows.append(row)
            fold_rows.extend(rows)
            print(
                "  {:6s} R2 {:.4f}->{:.4f} delta={:+.4f}; RMSE {:.4f}->{:.4f}".format(
                    preprocess,
                    row["baseline_r2"],
                    row["augmented_r2"],
                    row["delta_r2"],
                    row["baseline_rmse"],
                    row["augmented_rmse"],
                ),
                flush=True,
            )

    write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
    write_rows(folds_path, FOLD_FIELDS, fold_rows)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {folds_path}", flush=True)


def augmentation_specs() -> dict[str, AugSpec]:
    return {
        "coal_Q": AugSpec(n_synthetic=80, perturbation_mode="none", noise_scale=0.0),
        "coal_ash": AugSpec(n_synthetic=40, perturbation_mode="local_std", noise_scale=0.008),
        "soil_SOC": AugSpec(n_synthetic=80, perturbation_mode="local_std", noise_scale=0.008),
        "diesel_CN": AugSpec(n_synthetic=80, perturbation_mode="none", noise_scale=0.0),
        "diesel_FREEZE": AugSpec(n_synthetic=40, perturbation_mode="local_std", noise_scale=0.008),
    }


def evaluate_preprocess_augmentation(
    endpoint: Endpoint,
    X: np.ndarray,
    y: np.ndarray,
    preprocess: str,
    spec: AugSpec,
    args: argparse.Namespace,
    param_grid: dict,
) -> tuple[dict, list[dict]]:
    splitter = KFold(n_splits=min(args.cv, X.shape[0]), shuffle=True, random_state=args.random_state)
    baseline_pred = np.empty_like(y, dtype=float)
    augmented_pred = np.empty_like(y, dtype=float)
    fold_rows = []
    selected_features = []
    synthetic_accepted = []
    acceptance_rates = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        preprocessor = fit_spectral_preprocessor(X_train, preprocess)
        X_train_pre = preprocessor.transform(X_train)
        X_test_pre = preprocessor.transform(X_test)

        selector = CARSFeatureSelector(
            n_sampling=args.cars_sampling,
            min_features=args.cars_min_features,
            max_components=args.cars_components,
            cv=min(args.inner_cv, 5),
            random_state=int(args.random_state) + fold_idx,
        )
        Z_train = selector.fit_transform(X_train_pre, y_train)
        Z_test = selector.transform(X_test_pre)

        baseline_search = make_svr_search(param_grid, args, Z_train.shape[0], fold_idx)
        baseline_search.fit(Z_train, y_train)
        fold_baseline = baseline_search.predict(Z_test).reshape(-1)

        augmenter = ResponseDrivenAugmenter(
            n_synthetic=spec.n_synthetic,
            response_bins=6,
            response_bin_strategy="quantile",
            neighbors=5,
            alpha_min=0.15,
            alpha_max=0.85,
            noise_scale=spec.noise_scale,
            random_state=int(args.random_state) + fold_idx,
            neighbor_space="response",
            perturbation_mode=spec.perturbation_mode,
        )
        aug = augmenter.fit_resample(Z_train, y_train)
        augmented_search = make_svr_search(param_grid, args, aug.X.shape[0], fold_idx + 1000)
        augmented_search.fit(aug.X, aug.y)
        fold_augmented = augmented_search.predict(Z_test).reshape(-1)

        baseline_pred[test_idx] = fold_baseline
        augmented_pred[test_idx] = fold_augmented

        baseline_rmse = float(mean_squared_error(y_test, fold_baseline, squared=False))
        augmented_rmse = float(mean_squared_error(y_test, fold_augmented, squared=False))
        baseline_r2 = float(r2_score(y_test, fold_baseline))
        augmented_r2 = float(r2_score(y_test, fold_augmented))
        selected_features.append(int(Z_train.shape[1]))
        synthetic_accepted.append(int(aug.n_synthetic))
        acceptance_rates.append(float(aug.metadata["acceptance_rate"]))
        fold_rows.append(
            {
                "dataset": endpoint.dataset,
                "target": endpoint.target,
                "preprocess": preprocess,
                "fold": fold_idx,
                "train_size": int(train_idx.size),
                "test_size": int(test_idx.size),
                "n_synthetic": spec.n_synthetic,
                "perturbation_mode": spec.perturbation_mode,
                "noise_scale": spec.noise_scale,
                "baseline_rmse": baseline_rmse,
                "augmented_rmse": augmented_rmse,
                "delta_rmse": augmented_rmse - baseline_rmse,
                "baseline_r2": baseline_r2,
                "augmented_r2": augmented_r2,
                "delta_r2": augmented_r2 - baseline_r2,
                "selected_features": int(Z_train.shape[1]),
                "synthetic_accepted": int(aug.n_synthetic),
                "acceptance_rate": float(aug.metadata["acceptance_rate"]),
                "baseline_best_params": baseline_search.best_params_,
                "augmented_best_params": augmented_search.best_params_,
            }
        )

    baseline_rmse = float(mean_squared_error(y, baseline_pred, squared=False))
    augmented_rmse = float(mean_squared_error(y, augmented_pred, squared=False))
    baseline_r2 = float(r2_score(y, baseline_pred))
    augmented_r2 = float(r2_score(y, augmented_pred))
    stats = bootstrap_deltas(y, baseline_pred, augmented_pred, args.bootstrap, args.random_state)
    row = {
        "dataset": endpoint.dataset,
        "target": endpoint.target,
        "n_samples": int(X.shape[0]),
        "n_variables": int(X.shape[1]),
        "preprocess": preprocess,
        "model": "CARS-RBF-SVR",
        "cv_folds": int(min(args.cv, X.shape[0])),
        "inner_cv": int(args.inner_cv),
        "n_synthetic": spec.n_synthetic,
        "perturbation_mode": spec.perturbation_mode,
        "noise_scale": spec.noise_scale,
        "baseline_rmse": baseline_rmse,
        "augmented_rmse": augmented_rmse,
        "delta_rmse": augmented_rmse - baseline_rmse,
        "baseline_r2": baseline_r2,
        "augmented_r2": augmented_r2,
        "delta_r2": augmented_r2 - baseline_r2,
        "delta_rmse_95ci": stats["delta_rmse_95ci"],
        "delta_r2_95ci": stats["delta_r2_95ci"],
        "mean_selected_features": float(np.mean(selected_features)),
        "mean_synthetic_accepted": float(np.mean(synthetic_accepted)),
        "mean_acceptance_rate": float(np.mean(acceptance_rates)),
        "is_rmse_improved": int(augmented_rmse < baseline_rmse),
        "is_r2_improved": int(augmented_r2 > baseline_r2),
        "svr_grid_size": len(ParameterGrid(param_grid)),
    }
    return row, fold_rows


def make_svr_search(param_grid: dict, args: argparse.Namespace, n_samples: int, fold_idx: int) -> GridSearchCV:
    inner = KFold(
        n_splits=min(args.inner_cv, n_samples),
        shuffle=True,
        random_state=int(args.random_state) + int(fold_idx),
    )
    return GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner,
        n_jobs=1,
    )


def bootstrap_deltas(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    augmented_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, str]:
    if int(n_bootstrap) <= 0:
        return {"delta_rmse_95ci": "", "delta_r2_95ci": ""}
    rng = np.random.default_rng(random_state)
    delta_rmse = []
    delta_r2 = []
    n = y_true.shape[0]
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        if np.allclose(y_true[idx].min(), y_true[idx].max()):
            continue
        base_rmse = mean_squared_error(y_true[idx], baseline_pred[idx], squared=False)
        aug_rmse = mean_squared_error(y_true[idx], augmented_pred[idx], squared=False)
        base_r2 = r2_score(y_true[idx], baseline_pred[idx])
        aug_r2 = r2_score(y_true[idx], augmented_pred[idx])
        delta_rmse.append(float(aug_rmse - base_rmse))
        delta_r2.append(float(aug_r2 - base_r2))
    return {
        "delta_rmse_95ci": _ci(delta_rmse),
        "delta_r2_95ci": _ci(delta_r2),
    }


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: list[float]) -> str:
    if not values:
        return ""
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return f"[{low:.6g}, {high:.6g}]"


def _parse_gamma(value: str) -> float | str:
    if value in {"scale", "auto"}:
        return value
    return float(value)


if __name__ == "__main__":
    main()
