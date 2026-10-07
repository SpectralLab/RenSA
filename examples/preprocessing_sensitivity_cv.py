from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import CARSFeatureSelector, load_spectrum_csv
from respond_spectra.preprocessing import msc, savgol, snv


PREPROCESS_METHODS = ["raw", "snv", "sg0", "sg1", "sg2", "snv-sg0", "msc", "msc-sg0"]

SUMMARY_FIELDS = [
    "dataset",
    "target",
    "n_samples",
    "n_variables",
    "preprocess",
    "model",
    "cv_folds",
    "inner_cv",
    "pooled_rmse",
    "pooled_r2",
    "mean_fold_rmse",
    "std_fold_rmse",
    "mean_fold_r2",
    "std_fold_r2",
    "mean_selected_features",
    "std_selected_features",
    "rank_by_pooled_rmse",
    "rank_by_pooled_r2",
    "svr_grid_size",
]

FOLD_FIELDS = [
    "dataset",
    "target",
    "preprocess",
    "fold",
    "train_size",
    "test_size",
    "rmse",
    "r2",
    "selected_features",
    "best_params",
]


@dataclass(frozen=True)
class Endpoint:
    dataset: str
    target: str
    path: Path
    target_column: str
    loader: str = "xprefix"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated preprocessing sensitivity screen for the five "
            "main solid/publication endpoints under a common CARS-RBF-SVR pipeline."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "publication_solid")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--inner-cv", type=int, default=3)
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
    summary_path = args.output_dir / "table_preprocessing_sensitivity_cv_summary.csv"
    folds_path = args.output_dir / "table_preprocessing_sensitivity_cv_folds.csv"

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
        print(
            f"{endpoint.dataset}: {X.shape[0]} samples, {X.shape[1]} variables, target={endpoint.target}",
            flush=True,
        )
        for preprocess in PREPROCESS_METHODS:
            row, rows = evaluate_preprocess(endpoint, X, y, preprocess, args, param_grid)
            summary_rows.append(row)
            fold_rows.extend(rows)
            print(
                "  {:6s} pooled_RMSE={:.4f} pooled_R2={:.4f} mean_features={:.1f}".format(
                    preprocess,
                    row["pooled_rmse"],
                    row["pooled_r2"],
                    row["mean_selected_features"],
                ),
                flush=True,
            )

    add_ranks(summary_rows)
    write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
    write_rows(folds_path, FOLD_FIELDS, fold_rows)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {folds_path}", flush=True)


def endpoints_for_repo(root: Path) -> list[Endpoint]:
    data = root / "data"
    return [
        Endpoint("coal_Q", "Heating value", data / "coal_Q.csv", "y"),
        Endpoint("coal_ash", "Ash content", data / "coal_ash.csv", "y"),
        Endpoint("soil_SOC", "Soil organic carbon", data / "soil_SOC.csv", "y"),
        Endpoint("diesel_CN", "Cetane number", data / "diesel_CN.csv", "CN", loader="diesel"),
        Endpoint("diesel_FREEZE", "Freezing point", data / "diesel_FREEZE.csv", "FREEZE", loader="diesel"),
    ]


def load_endpoint(endpoint: Endpoint) -> tuple[np.ndarray, np.ndarray]:
    if endpoint.loader == "xprefix":
        _, X, y = load_spectrum_csv(endpoint.path, target_column=endpoint.target_column, spectral_prefix="x")
        return X, y
    if endpoint.loader == "diesel":
        frame = pd.read_csv(endpoint.path)
        y = frame[endpoint.target_column].to_numpy(dtype=float).reshape(-1)
        spectral_columns = [
            column
            for column in frame.columns
            if column not in {"Label", endpoint.target_column} and _is_float_like(column)
        ]
        X = frame.loc[:, spectral_columns].to_numpy(dtype=float)
        return X, y
    raise ValueError(f"Unknown endpoint loader: {endpoint.loader}")


def evaluate_preprocess(
    endpoint: Endpoint,
    X: np.ndarray,
    y: np.ndarray,
    preprocess: str,
    args: argparse.Namespace,
    param_grid: dict,
) -> tuple[dict, list[dict]]:
    splitter = KFold(n_splits=min(args.cv, X.shape[0]), shuffle=True, random_state=args.random_state)
    pred = np.empty_like(y, dtype=float)
    fold_rows = []
    fold_rmse = []
    fold_r2 = []
    selected_features = []

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

        search = make_svr_search(param_grid, args, Z_train.shape[0], fold_idx)
        search.fit(Z_train, y_train)
        fold_pred = search.predict(Z_test).reshape(-1)
        pred[test_idx] = fold_pred

        rmse = float(mean_squared_error(y_test, fold_pred, squared=False))
        r2 = float(r2_score(y_test, fold_pred))
        n_selected = int(Z_train.shape[1])
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        selected_features.append(n_selected)
        fold_rows.append(
            {
                "dataset": endpoint.dataset,
                "target": endpoint.target,
                "preprocess": preprocess,
                "fold": fold_idx,
                "train_size": int(train_idx.size),
                "test_size": int(test_idx.size),
                "rmse": rmse,
                "r2": r2,
                "selected_features": n_selected,
                "best_params": search.best_params_,
            }
        )

    row = {
        "dataset": endpoint.dataset,
        "target": endpoint.target,
        "n_samples": int(X.shape[0]),
        "n_variables": int(X.shape[1]),
        "preprocess": preprocess,
        "model": "CARS-RBF-SVR",
        "cv_folds": int(min(args.cv, X.shape[0])),
        "inner_cv": int(args.inner_cv),
        "pooled_rmse": float(mean_squared_error(y, pred, squared=False)),
        "pooled_r2": float(r2_score(y, pred)),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "std_fold_rmse": float(np.std(fold_rmse, ddof=1)),
        "mean_fold_r2": float(np.mean(fold_r2)),
        "std_fold_r2": float(np.std(fold_r2, ddof=1)),
        "mean_selected_features": float(np.mean(selected_features)),
        "std_selected_features": float(np.std(selected_features, ddof=1)),
        "rank_by_pooled_rmse": "",
        "rank_by_pooled_r2": "",
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


@dataclass
class SpectralPreprocessor:
    method: str
    msc_reference: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        window_length = min(21, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
        if self.method == "raw":
            return X
        if self.method == "snv":
            return snv(X)
        if self.method == "sg0":
            return savgol(X, window_length=window_length, polyorder=2, deriv=0)
        if self.method == "sg1":
            return savgol(X, window_length=window_length, polyorder=2, deriv=1)
        if self.method == "sg2":
            return savgol(X, window_length=window_length, polyorder=2, deriv=2)
        if self.method == "snv-sg0":
            return savgol(snv(X), window_length=window_length, polyorder=2, deriv=0)
        if self.method == "msc":
            return msc(X, reference=self.msc_reference)
        if self.method == "msc-sg0":
            return savgol(msc(X, reference=self.msc_reference), window_length=window_length, polyorder=2, deriv=0)
        raise ValueError(f"Unknown preprocessing method: {self.method}")


def fit_spectral_preprocessor(X_train: np.ndarray, method: str) -> SpectralPreprocessor:
    reference = X_train.mean(axis=0) if method in {"msc", "msc-sg0"} else None
    return SpectralPreprocessor(method=method, msc_reference=reference)


def add_ranks(rows: list[dict]) -> None:
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        for rank, row in enumerate(sorted(group, key=lambda item: item["pooled_rmse"]), start=1):
            row["rank_by_pooled_rmse"] = rank
        for rank, row in enumerate(sorted(group, key=lambda item: item["pooled_r2"], reverse=True), start=1):
            row["rank_by_pooled_r2"] = rank


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_gamma(value: str) -> float | str:
    if value in {"scale", "auto"}:
        return value
    return float(value)


def _is_float_like(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
