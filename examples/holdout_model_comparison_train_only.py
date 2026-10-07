from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, load_spectrum_csv
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDS = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "spxy_y_weight",
    "train_size",
    "test_count",
    "inner_cv",
    "model",
    "inner_cv_rmse",
    "inner_cv_r2",
    "test_rmse",
    "test_r2",
    "best_params",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train-only holdout comparison across baseline regressors.")
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_model_compare_trainonly.csv")
    parser.add_argument("--preprocess", default="raw", choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"])
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--spxy-y-weight", type=float, default=1.0)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["plsr", "rbf-svr", "random-forest", "extra-trees", "cars-rbf-svr"],
        choices=["plsr", "rbf-svr", "random-forest", "extra-trees", "cars-rbf-svr"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column, spectral_prefix=args.spectral_prefix)
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem
    train_idx, test_idx = representative_split(
        X,
        y,
        test_size=args.test_size,
        method=args.split_method,
        spxy_y_weight=args.spxy_y_weight,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for model_name in args.models:
            started = time.perf_counter()
            print(f"run {dataset} preprocess={args.preprocess} model={model_name}", flush=True)
            row = evaluate_model(model_name, X_train, y_train, X_test, y_test, args)
            row.update(
                {
                    "run_id": int(time.time()),
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "dataset": dataset,
                    "preprocess": args.preprocess,
                    "split_method": args.split_method,
                    "test_size": args.test_size,
                    "spxy_y_weight": args.spxy_y_weight,
                    "train_size": int(train_idx.size),
                    "test_count": int(test_idx.size),
                    "inner_cv": args.inner_cv,
                    "model": model_name,
                }
            )
            writer.writerow(row)
            handle.flush()
            print("  inner_rmse={:.4f} test_r2={:.4f}".format(row["inner_cv_rmse"], row["test_r2"]), flush=True)
    print(f"wrote {args.output}", flush=True)


def evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    if model_name == "plsr":
        return evaluate_plsr(X_train, y_train, X_test, y_test, args)
    estimator, grid = estimator_and_grid(model_name, args)
    search = GridSearchCV(
        estimator,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_splitter(args, X_train.shape[0]),
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test).reshape(-1)
    inner_pred = search.predict(X_train).reshape(-1)
    return {
        "inner_cv_rmse": float(-search.best_score_),
        "inner_cv_r2": float(r2_score(y_train, inner_pred)),
        "test_rmse": float(mean_squared_error(y_test, pred, squared=False)),
        "test_r2": float(r2_score(y_test, pred)),
        "best_params": json.dumps(search.best_params_),
    }


def evaluate_plsr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    splitter = inner_splitter(args, X_train.shape[0])
    upper = min(20, X_train.shape[0] - 2, X_train.shape[1])
    candidates = [n for n in [2, 4, 6, 8, 10, 12, 15, 20] if n <= upper]
    best = None
    for n_components in candidates:
        pred = np.empty_like(y_train, dtype=float)
        for fit_idx, valid_idx in splitter.split(X_train):
            model = PLSRegression(n_components=min(n_components, fit_idx.size - 1, X_train.shape[1]))
            model.fit(X_train[fit_idx], y_train[fit_idx])
            pred[valid_idx] = model.predict(X_train[valid_idx]).reshape(-1)
        rmse = float(mean_squared_error(y_train, pred, squared=False))
        r2 = float(r2_score(y_train, pred))
        if best is None or rmse < best["inner_cv_rmse"]:
            best = {"n_components": n_components, "inner_cv_rmse": rmse, "inner_cv_r2": r2}
    final = PLSRegression(n_components=best["n_components"])
    final.fit(X_train, y_train)
    test_pred = final.predict(X_test).reshape(-1)
    return {
        "inner_cv_rmse": best["inner_cv_rmse"],
        "inner_cv_r2": best["inner_cv_r2"],
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "best_params": json.dumps({"n_components": best["n_components"]}),
    }


def estimator_and_grid(model_name: str, args: argparse.Namespace):
    if model_name == "rbf-svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf")), {
            "svr__C": [100.0, 300.0, 1000.0, 3000.0],
            "svr__gamma": ["scale", 0.001, 0.003, 0.01],
            "svr__epsilon": [0.05, 0.1, 0.2],
        }
    if model_name == "random-forest":
        return RandomForestRegressor(random_state=args.random_state, n_jobs=1), {
            "n_estimators": [300],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 12],
        }
    if model_name == "extra-trees":
        return ExtraTreesRegressor(random_state=args.random_state, n_jobs=1), {
            "n_estimators": [300],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 12],
        }
    if model_name == "cars-rbf-svr":
        return make_pipeline(
            CARSFeatureSelector(
                n_sampling=40,
                min_features=40,
                max_components=8,
                cv=min(args.inner_cv, 5),
                random_state=args.random_state,
            ),
            StandardScaler(),
            SVR(kernel="rbf"),
        ), {
            "svr__C": [1000.0, 3000.0, 10000.0],
            "svr__gamma": [0.001, 0.0015, 0.003],
            "svr__epsilon": [0.05, 0.1],
        }
    raise ValueError(f"Unknown model: {model_name}")


def inner_splitter(args: argparse.Namespace, n_samples: int) -> KFold:
    return KFold(n_splits=min(args.inner_cv, n_samples), shuffle=True, random_state=args.random_state)


if __name__ == "__main__":
    main()
