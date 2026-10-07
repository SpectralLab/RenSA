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

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "train_size",
    "test_count",
    "spxy_y_weight",
    "n_synthetic",
    "neighbor_space",
    "response_bin_strategy",
    "response_bins",
    "neighbors",
    "alpha_min",
    "alpha_max",
    "perturbation_mode",
    "noise_scale",
    "cars_sampling",
    "cars_min_features",
    "cars_components",
    "svr_grid_size",
    "baseline_inner_cv_rmse",
    "baseline_inner_cv_r2",
    "baseline_test_rmse",
    "baseline_test_r2",
    "augmented_inner_cv_rmse",
    "augmented_inner_cv_r2",
    "augmented_test_rmse",
    "augmented_test_r2",
    "delta_test_rmse",
    "delta_test_r2",
    "baseline_best_params",
    "augmented_best_params",
    "baseline_selected_features",
    "augmented_selected_features",
    "synthetic_accepted",
    "acceptance_rate",
    "train_indices",
    "test_indices",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="KS/SPXY holdout CARS-SVR with train-only response-driven augmentation."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_cars_svr_ks_spxy.csv")
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, nargs="+", default=[0.2, 0.25, 0.3])
    parser.add_argument("--spxy-y-weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, nargs="+", default=[0, 40, 60, 80])
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--response-bin-strategy", default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--response-bins", type=int, nargs="+", default=[6])
    parser.add_argument("--neighbors", type=int, nargs="+", default=[5])
    parser.add_argument("--alpha-min", type=float, nargs="+", default=[0.15])
    parser.add_argument("--alpha-max", type=float, nargs="+", default=[0.85])
    parser.add_argument("--perturbation-mode", nargs="+", default=["local_std"], choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.008])
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }

    completed = _completed_keys(args.output) if args.resume else set()
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    n_new = 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for config in iter_configs(args):
            key = config_key(config)
            if key in completed:
                print(f"skip completed {key}", flush=True)
                continue
            if args.max_runs and n_new >= args.max_runs:
                break
            n_new += 1
            train_idx, test_idx = representative_split(
                X,
                y,
                test_size=config["test_size"],
                method=config["split_method"],
                spxy_y_weight=config["spxy_y_weight"],
            )
            started = time.perf_counter()
            print(
                "run {}: split={} test={} n={} mode={} noise={}".format(
                    n_new,
                    config["split_method"],
                    config["test_size"],
                    config["n_synthetic"],
                    config["perturbation_mode"],
                    config["noise_scale"],
                ),
                flush=True,
            )
            row = evaluate_config(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                args=args,
                dataset=dataset,
                param_grid=param_grid,
                config=config,
            )
            row["run_id"] = int(time.time())
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            writer.writerow(row)
            handle.flush()
            print(
                "  baseline_r2={:.4f} augmented_r2={:.4f} delta={:+.4f}".format(
                    row["baseline_test_r2"],
                    row["augmented_test_r2"],
                    row["delta_test_r2"],
                ),
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)


def iter_configs(args: argparse.Namespace):
    for (
        test_size,
        spxy_y_weight,
        n_synthetic,
        response_bins,
        neighbors,
        alpha_min,
        alpha_max,
        perturbation_mode,
        noise_scale,
    ) in product(
        args.test_size,
        args.spxy_y_weight,
        args.n_synthetic,
        args.response_bins,
        args.neighbors,
        args.alpha_min,
        args.alpha_max,
        args.perturbation_mode,
        args.noise_scale,
    ):
        if alpha_min >= alpha_max:
            continue
        if n_synthetic == 0 and (
            response_bins != args.response_bins[0]
            or neighbors != args.neighbors[0]
            or alpha_min != args.alpha_min[0]
            or alpha_max != args.alpha_max[0]
            or perturbation_mode != args.perturbation_mode[0]
            or noise_scale != args.noise_scale[0]
        ):
            continue
        yield {
            "split_method": args.split_method,
            "test_size": float(test_size),
            "spxy_y_weight": float(spxy_y_weight),
            "n_synthetic": int(n_synthetic),
            "response_bins": int(response_bins),
            "neighbors": int(neighbors),
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "perturbation_mode": str(perturbation_mode),
            "noise_scale": float(noise_scale),
        }


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    param_grid: dict,
    config: dict,
) -> dict:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    baseline = inner_select_and_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        args=args,
        param_grid=param_grid,
        augmenter=None,
    )
    augmenter = None
    if config["n_synthetic"] > 0:
        augmenter = ResponseDrivenAugmenter(
            n_synthetic=config["n_synthetic"],
            response_bins=config["response_bins"],
            response_bin_strategy=args.response_bin_strategy,
            neighbors=config["neighbors"],
            alpha_min=config["alpha_min"],
            alpha_max=config["alpha_max"],
            noise_scale=config["noise_scale"],
            random_state=args.random_state,
            neighbor_space=args.neighbor_space,
            perturbation_mode=config["perturbation_mode"],
        )
    augmented = inner_select_and_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        args=args,
        param_grid=param_grid,
        augmenter=augmenter,
    )
    return {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "split_method": config["split_method"],
        "test_size": config["test_size"],
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "spxy_y_weight": config["spxy_y_weight"],
        "n_synthetic": config["n_synthetic"],
        "neighbor_space": args.neighbor_space,
        "response_bin_strategy": args.response_bin_strategy,
        "response_bins": config["response_bins"],
        "neighbors": config["neighbors"],
        "alpha_min": config["alpha_min"],
        "alpha_max": config["alpha_max"],
        "perturbation_mode": config["perturbation_mode"],
        "noise_scale": config["noise_scale"],
        "cars_sampling": args.cars_sampling,
        "cars_min_features": args.cars_min_features,
        "cars_components": args.cars_components,
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "baseline_inner_cv_rmse": baseline["inner_cv_rmse"],
        "baseline_inner_cv_r2": baseline["inner_cv_r2"],
        "baseline_test_rmse": baseline["test_rmse"],
        "baseline_test_r2": baseline["test_r2"],
        "augmented_inner_cv_rmse": augmented["inner_cv_rmse"],
        "augmented_inner_cv_r2": augmented["inner_cv_r2"],
        "augmented_test_rmse": augmented["test_rmse"],
        "augmented_test_r2": augmented["test_r2"],
        "delta_test_rmse": augmented["test_rmse"] - baseline["test_rmse"],
        "delta_test_r2": augmented["test_r2"] - baseline["test_r2"],
        "baseline_best_params": json.dumps(baseline["best_params"]),
        "augmented_best_params": json.dumps(augmented["best_params"]),
        "baseline_selected_features": baseline["selected_features"],
        "augmented_selected_features": augmented["selected_features"],
        "synthetic_accepted": augmented["synthetic_accepted"],
        "acceptance_rate": augmented["acceptance_rate"],
        "train_indices": json.dumps(train_idx.tolist()),
        "test_indices": json.dumps(test_idx.tolist()),
    }


def inner_select_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
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
    Z_train = transformer.fit_transform(X_train, y_train)
    Z_test = transformer.transform(X_test)
    synthetic_accepted = 0
    acceptance_rate = 0.0
    if augmenter is not None:
        result = _clone_augmenter_for_fold(augmenter, 0).fit_resample(Z_train, y_train)
        Z_fit, y_fit = result.X, result.y
        synthetic_accepted = result.n_synthetic
        acceptance_rate = float(result.metadata["acceptance_rate"])
    else:
        Z_fit, y_fit = Z_train, y_train

    inner = KFold(n_splits=min(args.inner_cv, Z_fit.shape[0]), shuffle=True, random_state=args.random_state)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner,
        n_jobs=1,
    )
    search.fit(Z_fit, y_fit)
    train_pred = search.predict(Z_fit).reshape(-1)
    test_pred = search.predict(Z_test).reshape(-1)
    return {
        "inner_cv_rmse": float(-search.best_score_),
        "inner_cv_r2": float(r2_score(y_fit, train_pred)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "best_params": search.best_params_,
        "selected_features": int(Z_train.shape[1]),
        "synthetic_accepted": synthetic_accepted,
        "acceptance_rate": acceptance_rate,
    }


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def config_key(config: dict) -> tuple:
    return (
        config["split_method"],
        float(config["test_size"]),
        float(config["spxy_y_weight"]),
        int(config["n_synthetic"]),
        int(config["response_bins"]),
        int(config["neighbors"]),
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        str(config["perturbation_mode"]),
        float(config["noise_scale"]),
    )


def _completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            config_key(
                {
                    "split_method": row["split_method"],
                    "test_size": row["test_size"],
                    "spxy_y_weight": row["spxy_y_weight"],
                    "n_synthetic": row["n_synthetic"],
                    "response_bins": row["response_bins"],
                    "neighbors": row["neighbors"],
                    "alpha_min": row["alpha_min"],
                    "alpha_max": row["alpha_max"],
                    "perturbation_mode": row["perturbation_mode"],
                    "noise_scale": row["noise_scale"],
                }
            )
            for row in csv.DictReader(handle)
        }


if __name__ == "__main__":
    main()
