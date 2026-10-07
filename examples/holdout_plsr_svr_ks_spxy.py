from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "model",
    "n_synthetic",
    "best_params",
    "baseline_test_rmse",
    "baseline_test_r2",
    "augmented_test_rmse",
    "augmented_test_r2",
    "delta_test_rmse",
    "delta_test_r2",
    "synthetic_accepted",
    "acceptance_rate",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="KS/SPXY holdout PLSR/SVR augmentation comparison.")
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_plsr_svr.csv")
    parser.add_argument("--preprocess", default="raw", choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"])
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--models", nargs="+", default=["plsr", "rbf-svr"], choices=["plsr", "rbf-svr"])
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--n-synthetic", type=int, default=60)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--response-bins", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)
    train_idx, test_idx = representative_split(X, y, args.test_size, args.split_method)
    dataset = args.dataset_name or args.data.stem

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for model in args.models:
            started = time.perf_counter()
            print(f"run model={model}", flush=True)
            row = evaluate_model(X, y, train_idx, test_idx, args, dataset, model)
            row["run_id"] = int(time.time())
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            writer.writerow(row)
            handle.flush()
            print(
                "  baseline_r2={:.4f} augmented_r2={:.4f} delta={:+.4f}".format(
                    row["baseline_test_r2"], row["augmented_test_r2"], row["delta_test_r2"]
                ),
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
) -> dict:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    baseline = inner_select_and_test(X_train, y_train, X_test, y_test, args, model_name, augmenter=None)
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=args.response_bins,
        neighbors=args.neighbors,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=args.perturbation_mode,
    )
    augmented = inner_select_and_test(X_train, y_train, X_test, y_test, args, model_name, augmenter=augmenter)
    return {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "split_method": args.split_method,
        "test_size": args.test_size,
        "model": model_name,
        "n_synthetic": args.n_synthetic,
        "best_params": json.dumps(augmented["best_params"]),
        "baseline_test_rmse": baseline["test_rmse"],
        "baseline_test_r2": baseline["test_r2"],
        "augmented_test_rmse": augmented["test_rmse"],
        "augmented_test_r2": augmented["test_r2"],
        "delta_test_rmse": augmented["test_rmse"] - baseline["test_rmse"],
        "delta_test_r2": augmented["test_r2"] - baseline["test_r2"],
        "synthetic_accepted": augmented["synthetic_accepted"],
        "acceptance_rate": augmented["acceptance_rate"],
    }


def inner_select_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    model_name: str,
    augmenter: ResponseDrivenAugmenter | None,
) -> dict:
    X_fit, y_fit = X_train, y_train
    synthetic_accepted = 0
    acceptance_rate = 0.0
    if augmenter is not None:
        result = _clone_augmenter_for_fold(augmenter, 0).fit_resample(X_train, y_train)
        X_fit, y_fit = result.X, result.y
        synthetic_accepted = result.n_synthetic
        acceptance_rate = float(result.metadata["acceptance_rate"])

    inner = KFold(n_splits=min(args.inner_cv, X_fit.shape[0]), shuffle=True, random_state=args.random_state)
    if model_name == "plsr":
        best = None
        upper = min(args.max_components, X_fit.shape[1], X_fit.shape[0] - 1)
        for n_components in range(1, upper + 1):
            pred = np.empty_like(y_fit, dtype=float)
            for inner_train, inner_valid in inner.split(X_fit):
                n = min(n_components, inner_train.size - 1, X_fit.shape[1])
                estimator = PLSRegression(n_components=n)
                estimator.fit(X_fit[inner_train], y_fit[inner_train])
                pred[inner_valid] = estimator.predict(X_fit[inner_valid]).reshape(-1)
            rmse = float(mean_squared_error(y_fit, pred, squared=False))
            if best is None or rmse < best["rmse"]:
                best = {"rmse": rmse, "n_components": n_components}
        final = PLSRegression(n_components=best["n_components"])
        final.fit(X_fit, y_fit)
        test_pred = final.predict(X_test).reshape(-1)
        best_params = {"n_components": best["n_components"]}
    else:
        param_grid = {
            "svr__C": [float(value) for value in args.svr_C],
            "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
            "svr__epsilon": [float(value) for value in args.svr_epsilon],
        }
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        )
        search.fit(X_fit, y_fit)
        test_pred = search.predict(X_test).reshape(-1)
        best_params = search.best_params_
    return {
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "best_params": best_params,
        "synthetic_accepted": synthetic_accepted,
        "acceptance_rate": acceptance_rate,
    }


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


if __name__ == "__main__":
    main()
