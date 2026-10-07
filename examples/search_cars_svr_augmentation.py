from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import (
    _clone_augmenter_for_fold,
    _validated_cv_inputs,
)
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "preprocess",
    "cv",
    "inner_cv",
    "n_synthetic",
    "neighbor_space",
    "perturbation_mode",
    "response_consistency",
    "cars_sampling",
    "cars_min_features",
    "cars_components",
    "svr_grid_size",
    "baseline_rmse",
    "baseline_r2",
    "augmented_rmse",
    "augmented_r2",
    "delta_rmse",
    "delta_r2",
    "baseline_best_params_by_fold",
    "augmented_best_params_by_fold",
    "baseline_selected_features_by_fold",
    "augmented_selected_features_by_fold",
    "augmented_synthetic_accepted_by_fold",
    "augmented_acceptance_rate_mean",
    "augmented_acceptance_rate_by_fold",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Focused tuned RBF-SVR + CARS + response-neighbor augmentation search."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--output", type=Path, default=root / "results" / "tuned_svr_cars_search.csv")
    parser.add_argument("--preprocess", default="raw", choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--n-synthetic", type=int, default=80)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--response-consistency", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--min-derivative-corr", type=float, default=0.70)
    parser.add_argument("--max-spectral-angle", type=float, default=0.18)
    parser.add_argument("--envelope-margin", type=float, default=0.08)
    parser.add_argument("--cars-sampling", type=int, nargs="+", default=[40, 60])
    parser.add_argument("--cars-min-features", type=int, nargs="+", default=[20, 30, 40, 60, 80, 120])
    parser.add_argument("--cars-components", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--svr-C", type=float, nargs="+", default=[10, 30, 100, 300, 1000])
    parser.add_argument(
        "--svr-gamma",
        nargs="+",
        default=["scale", "0.001", "0.003", "0.01", "0.03", "0.1"],
        help="Use 'scale' or numeric gamma values.",
    )
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1])
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on new CARS combinations. 0 means run all requested combinations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip CARS combinations already present in the output CSV.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)
    param_grid = {
        "svr__C": [float(v) for v in args.svr_C],
        "svr__gamma": [_parse_gamma(v) for v in args.svr_gamma],
        "svr__epsilon": [float(v) for v in args.svr_epsilon],
    }
    combos = list(product(args.cars_sampling, args.cars_min_features, args.cars_components))
    completed = _completed_keys(args.output) if args.resume else set()
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    n_new = 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for cars_sampling, cars_min_features, cars_components in combos:
            key = (cars_sampling, cars_min_features, cars_components)
            if key in completed:
                print(f"skip completed CARS={key}", flush=True)
                continue
            if args.max_runs and n_new >= args.max_runs:
                break

            n_new += 1
            print(f"run {n_new}: CARS={key}", flush=True)
            started = time.perf_counter()
            row = evaluate_combo(
                X=X,
                y=y,
                args=args,
                param_grid=param_grid,
                cars_sampling=cars_sampling,
                cars_min_features=cars_min_features,
                cars_components=cars_components,
            )
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            row["run_id"] = int(time.time())
            writer.writerow(row)
            handle.flush()
            print(
                "  baseline_r2={:.4f} augmented_r2={:.4f} delta_r2={:+.4f}".format(
                    row["baseline_r2"],
                    row["augmented_r2"],
                    row["delta_r2"],
                ),
                flush=True,
            )

    print(f"wrote {args.output}", flush=True)


def evaluate_combo(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    cars_sampling: int,
    cars_min_features: int,
    cars_components: int,
) -> dict:
    transformer = CARSFeatureSelector(
        n_sampling=cars_sampling,
        min_features=cars_min_features,
        max_components=cars_components,
        cv=min(args.cv, 5),
        random_state=args.random_state,
    )
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=6,
        neighbors=5,
        noise_scale=args.noise_scale,
        min_derivative_corr=args.min_derivative_corr,
        max_spectral_angle=args.max_spectral_angle,
        envelope_margin=args.envelope_margin,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=args.perturbation_mode,
        response_consistency=args.response_consistency,
    )

    baseline = evaluate_tuned_svr_cv(
        X,
        y,
        transformer=transformer,
        augmenter=None,
        param_grid=param_grid,
        cv=args.cv,
        inner_cv=args.inner_cv,
        random_state=args.random_state,
    )
    augmented = evaluate_tuned_svr_cv(
        X,
        y,
        transformer=transformer,
        augmenter=augmenter,
        param_grid=param_grid,
        cv=args.cv,
        inner_cv=args.inner_cv,
        random_state=args.random_state,
    )

    acceptance_rates = [fold["acceptance_rate"] for fold in augmented["folds"]]
    return {
        "elapsed_sec": "",
        "run_id": "",
        "preprocess": args.preprocess,
        "cv": args.cv,
        "inner_cv": args.inner_cv,
        "n_synthetic": args.n_synthetic,
        "neighbor_space": args.neighbor_space,
        "perturbation_mode": args.perturbation_mode,
        "response_consistency": bool(args.response_consistency),
        "cars_sampling": cars_sampling,
        "cars_min_features": cars_min_features,
        "cars_components": cars_components,
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "augmented_rmse": augmented["rmse"],
        "augmented_r2": augmented["r2"],
        "delta_rmse": augmented["rmse"] - baseline["rmse"],
        "delta_r2": augmented["r2"] - baseline["r2"],
        "baseline_best_params_by_fold": json.dumps(baseline["best_params_by_fold"]),
        "augmented_best_params_by_fold": json.dumps(augmented["best_params_by_fold"]),
        "baseline_selected_features_by_fold": json.dumps(baseline["selected_features_by_fold"]),
        "augmented_selected_features_by_fold": json.dumps(augmented["selected_features_by_fold"]),
        "augmented_synthetic_accepted_by_fold": json.dumps(
            [fold["synthetic_accepted"] for fold in augmented["folds"]]
        ),
        "augmented_acceptance_rate_mean": float(np.mean(acceptance_rates)),
        "augmented_acceptance_rate_by_fold": json.dumps(acceptance_rates),
    }


def evaluate_tuned_svr_cv(
    X: np.ndarray,
    y: np.ndarray,
    transformer,
    augmenter: ResponseDrivenAugmenter | None,
    param_grid: dict,
    cv: int,
    inner_cv: int,
    random_state: int,
) -> dict:
    X, y, splitter = _validated_cv_inputs(X, y, cv, random_state)
    inner = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    pred = np.empty_like(y, dtype=float)
    best_params = []
    selected_features = []
    fold_metadata = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        fold_transformer = clone(transformer)
        X_train = fold_transformer.fit_transform(X[train_idx], y[train_idx])
        X_test = fold_transformer.transform(X[test_idx])
        y_train = y[train_idx]
        selected_features.append(int(X_train.shape[1]))

        if augmenter is not None:
            fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
            result = fold_augmenter.fit_resample(X_train, y_train)
            X_fit, y_fit = result.X, result.y
            fold_metadata.append(
                {
                    "fold": fold_idx,
                    "synthetic_accepted": result.n_synthetic,
                    "acceptance_rate": float(result.metadata["acceptance_rate"]),
                }
            )
        else:
            X_fit, y_fit = X_train, y_train

        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        )
        search.fit(X_fit, y_fit)
        best_params.append(search.best_params_)
        pred[test_idx] = search.predict(X_test).reshape(-1)

    return {
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "r2": float(r2_score(y, pred)),
        "best_params_by_fold": best_params,
        "selected_features_by_fold": selected_features,
        "folds": fold_metadata,
    }


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def _completed_keys(path: Path) -> set[tuple[int, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                int(row["cars_sampling"]),
                int(row["cars_min_features"]),
                int(row["cars_components"]),
            )
            for row in reader
        }


if __name__ == "__main__":
    main()
