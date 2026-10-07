from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from run_coal_q_augmentation_demo import preprocess_spectra


SUMMARY_FIELDS = [
    "run_id",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "train_size",
    "test_count",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "baseline_rmse",
    "baseline_r2",
    "augmented_rmse",
    "augmented_r2",
    "delta_rmse",
    "delta_r2",
    "bootstrap_delta_rmse_ci_low",
    "bootstrap_delta_rmse_ci_high",
    "bootstrap_delta_r2_ci_low",
    "bootstrap_delta_r2_ci_high",
    "bootstrap_p_rmse_improved",
    "bootstrap_p_r2_improved",
    "baseline_best_params",
    "augmented_best_params",
    "selected_features",
    "synthetic_accepted",
    "acceptance_rate",
    "predictions_path",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Final KS/SPXY holdout CARS-SVR predictions with paired bootstrap intervals."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--loader", choices=["xprefix", "diesel"], default="xprefix")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--summary-output", type=Path, default=root / "results" / "final_holdout_summary.csv")
    parser.add_argument("--predictions-output", type=Path, default=root / "results" / "final_holdout_predictions.csv")
    parser.add_argument("--preprocess", default="raw", choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"])
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--spxy-y-weight", type=float, default=1.0)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, default=60)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--response-bin-strategy", default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--response-bins", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha-min", type=float, default=0.15)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)

    _, X, y = load_dataset(args)
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem
    train_idx, test_idx = representative_split(
        X,
        y,
        test_size=args.test_size,
        method=args.split_method,
        spxy_y_weight=args.spxy_y_weight,
    )
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }

    baseline = fit_predict(X, y, train_idx, test_idx, args, param_grid, augmenter=None)
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=args.response_bins,
        response_bin_strategy=args.response_bin_strategy,
        neighbors=args.neighbors,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=args.perturbation_mode,
    )
    augmented = fit_predict(X, y, train_idx, test_idx, args, param_grid, augmenter=augmenter)
    stats = bootstrap_deltas(
        baseline["y_true"],
        baseline["pred"],
        augmented["pred"],
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )

    write_predictions(args.predictions_output, dataset, test_idx, baseline["y_true"], baseline["pred"], augmented["pred"])
    row = {
        "run_id": int(time.time()),
        "dataset": dataset,
        "preprocess": args.preprocess,
        "split_method": args.split_method,
        "test_size": args.test_size,
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "n_synthetic": args.n_synthetic,
        "perturbation_mode": args.perturbation_mode,
        "noise_scale": args.noise_scale,
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "augmented_rmse": augmented["rmse"],
        "augmented_r2": augmented["r2"],
        "delta_rmse": augmented["rmse"] - baseline["rmse"],
        "delta_r2": augmented["r2"] - baseline["r2"],
        **stats,
        "baseline_best_params": json.dumps(baseline["best_params"]),
        "augmented_best_params": json.dumps(augmented["best_params"]),
        "selected_features": baseline["selected_features"],
        "synthetic_accepted": augmented["synthetic_accepted"],
        "acceptance_rate": augmented["acceptance_rate"],
        "predictions_path": str(args.predictions_output),
    }
    write_summary(args.summary_output, row)
    print(
        "baseline_r2={:.4f} augmented_r2={:.4f} delta={:+.4f} "
        "rmse={:.4f}->{:.4f}".format(
            row["baseline_r2"],
            row["augmented_r2"],
            row["delta_r2"],
            row["baseline_rmse"],
            row["augmented_rmse"],
        ),
        flush=True,
    )
    print(
        "bootstrap delta_r2 95% CI [{:.4f}, {:.4f}], delta_rmse 95% CI [{:.4f}, {:.4f}]".format(
            row["bootstrap_delta_r2_ci_low"],
            row["bootstrap_delta_r2_ci_high"],
            row["bootstrap_delta_rmse_ci_low"],
            row["bootstrap_delta_rmse_ci_high"],
        ),
        flush=True,
    )


def fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    augmenter: ResponseDrivenAugmenter | None,
) -> dict:
    transformer = CARSFeatureSelector(
        n_sampling=args.cars_sampling,
        min_features=args.cars_min_features,
        max_components=args.cars_components,
        cv=min(args.inner_cv, 5),
        random_state=args.random_state,
    )
    X_train = transformer.fit_transform(X[train_idx], y[train_idx])
    X_test = transformer.transform(X[test_idx])
    y_train = y[train_idx]
    synthetic_accepted = 0
    acceptance_rate = 0.0
    if augmenter is not None:
        result = _clone_augmenter_for_fold(augmenter, 0).fit_resample(X_train, y_train)
        X_fit, y_fit = result.X, result.y
        synthetic_accepted = result.n_synthetic
        acceptance_rate = float(result.metadata["acceptance_rate"])
    else:
        X_fit, y_fit = X_train, y_train

    inner = KFold(n_splits=min(args.inner_cv, X_fit.shape[0]), shuffle=True, random_state=args.random_state)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner,
        n_jobs=1,
    )
    search.fit(X_fit, y_fit)
    pred = search.predict(X_test).reshape(-1)
    y_true = y[test_idx]
    return {
        "y_true": y_true,
        "pred": pred,
        "rmse": float(mean_squared_error(y_true, pred, squared=False)),
        "r2": float(r2_score(y_true, pred)),
        "best_params": search.best_params_,
        "selected_features": int(X_train.shape[1]),
        "synthetic_accepted": synthetic_accepted,
        "acceptance_rate": acceptance_rate,
    }


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.loader == "xprefix":
        return load_spectrum_csv(
            args.data,
            target_column=args.target_column,
            spectral_prefix=args.spectral_prefix,
        )
    frame = pd.read_csv(args.data)
    y = frame[args.target_column].to_numpy(dtype=float).reshape(-1)
    spectral_columns = [
        column
        for column in frame.columns
        if column not in {"Label", args.target_column} and _is_float_like(column)
    ]
    X = frame.loc[:, spectral_columns].to_numpy(dtype=float)
    wavelengths = np.asarray([float(column) for column in spectral_columns], dtype=float)
    return wavelengths, X, y


def bootstrap_deltas(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    augmented_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    rng = np.random.default_rng(random_state)
    n = y_true.shape[0]
    delta_rmse = []
    delta_r2 = []
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
    delta_rmse_arr = np.asarray(delta_rmse, dtype=float)
    delta_r2_arr = np.asarray(delta_r2, dtype=float)
    return {
        "bootstrap_delta_rmse_ci_low": float(np.quantile(delta_rmse_arr, 0.025)),
        "bootstrap_delta_rmse_ci_high": float(np.quantile(delta_rmse_arr, 0.975)),
        "bootstrap_delta_r2_ci_low": float(np.quantile(delta_r2_arr, 0.025)),
        "bootstrap_delta_r2_ci_high": float(np.quantile(delta_r2_arr, 0.975)),
        "bootstrap_p_rmse_improved": float(np.mean(delta_rmse_arr < 0.0)),
        "bootstrap_p_r2_improved": float(np.mean(delta_r2_arr > 0.0)),
    }


def write_predictions(
    path: Path,
    dataset: str,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    augmented_pred: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_index", "y_true", "baseline_pred", "augmented_pred", "baseline_error", "augmented_error"],
        )
        writer.writeheader()
        for sample_index, truth, base, aug in zip(test_idx, y_true, baseline_pred, augmented_pred):
            writer.writerow(
                {
                    "dataset": dataset,
                    "sample_index": int(sample_index),
                    "y_true": float(truth),
                    "baseline_pred": float(base),
                    "augmented_pred": float(aug),
                    "baseline_error": float(base - truth),
                    "augmented_error": float(aug - truth),
                }
            )


def write_summary(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def _is_float_like(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
