from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from run_coal_q_augmentation_demo import preprocess_spectra


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    data: str
    preprocess: str


@dataclass(frozen=True)
class AugConfig:
    label: str
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float

    @property
    def is_augmented(self) -> bool:
        return self.n_synthetic > 0


TASKS = [
    TaskSpec("coal_Q", "Coal GCV", "coal_Q.csv", "raw"),
    TaskSpec("coal_ash", "Coal ash", "coal_ash.csv", "raw"),
    TaskSpec("soil_SOC", "Soil SOC", "soil_SOC.csv", "snv-sg"),
    TaskSpec("diesel_CN", "Diesel CN", "diesel_CN.csv", "snv-sg"),
    TaskSpec("diesel_FREEZE", "Diesel freeze", "diesel_FREEZE.csv", "snv"),
]


AUG_CONFIGS = [
    AugConfig("baseline", 0, "none", 0.0),
    AugConfig("n40_none", 40, "none", 0.0),
    AugConfig("n80_none", 80, "none", 0.0),
    AugConfig("n40_localstd", 40, "local_std", 0.008),
    AugConfig("n80_localstd", 80, "local_std", 0.008),
]


PIPELINES = ["PLSR", "Ridge", "RBF-SVR", "CARS-PLSR", "CARS-RBF-SVR", "CARS-RF"]
PIPELINE_COLORS = {
    "PLSR": "#3568A8",
    "Ridge": "#777777",
    "RBF-SVR": "#7B5BB4",
    "CARS-PLSR": "#44A6A6",
    "CARS-RBF-SVR": "#C65A2E",
    "CARS-RF": "#3E8A5A",
}


FIELDNAMES = [
    "dataset",
    "task_label",
    "preprocess",
    "train_size",
    "test_count",
    "pipeline",
    "baseline_inner_cv_rmse",
    "baseline_inner_cv_r2",
    "baseline_best_params",
    "selected_aug_config",
    "selected_n_synthetic",
    "selected_perturbation_mode",
    "selected_noise_scale",
    "renvsa_inner_cv_rmse",
    "renvsa_inner_cv_r2",
    "renvsa_best_params",
    "delta_inner_cv_r2",
    "delta_inner_cv_rmse",
]


DETAIL_FIELDS = [
    "dataset",
    "task_label",
    "preprocess",
    "pipeline",
    "aug_config",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "inner_cv_rmse",
    "inner_cv_r2",
    "best_params",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "publication_coal_diesel"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "table_modeling_pipeline_sensitivity_inner_cv.csv"
    detail_path = output / "table_modeling_pipeline_sensitivity_inner_cv_details.csv"

    all_summary_rows: list[dict[str, str]] = []
    all_detail_rows: list[dict[str, str]] = []
    started = time.perf_counter()

    for task in TASKS:
        print(f"task {task.label}: preprocess={task.preprocess}", flush=True)
        summary_rows, detail_rows = evaluate_task(root, task)
        all_summary_rows.extend(summary_rows)
        all_detail_rows.extend(detail_rows)
        write_csv(summary_path, all_summary_rows, FIELDNAMES)
        write_csv(detail_path, all_detail_rows, DETAIL_FIELDS)

    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {detail_path}", flush=True)
    print(f"elapsed_sec={time.perf_counter() - started:.1f}", flush=True)


def evaluate_task(root: Path, task: TaskSpec) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    X, y = load_task_data(root / "data" / task.data)
    X = preprocess_spectra(X, task.preprocess)
    train_idx, test_idx = representative_split(X, y, test_size=0.25, method="spxy", spxy_y_weight=1.0)
    X_train = X[train_idx]
    y_train = y[train_idx]

    fold_cache: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {}
    for config in AUG_CONFIGS:
        print(f"  folds {config.label}", flush=True)
        raw_folds = build_raw_folds(X_train, y_train, config)
        cars_folds = build_cars_folds(raw_folds)
        fold_cache[config.label] = {"raw": raw_folds, "cars": cars_folds}

    detail_rows: list[dict[str, str]] = []
    scores: dict[tuple[str, str], dict[str, object]] = {}
    for pipeline in PIPELINES:
        fold_kind = "cars" if pipeline.startswith("CARS-") else "raw"
        print(f"  pipeline {pipeline}", flush=True)
        for config in AUG_CONFIGS:
            result = select_by_inner_cv(pipeline, fold_cache[config.label][fold_kind])
            scores[(pipeline, config.label)] = result
            detail_rows.append(
                {
                    "dataset": task.key,
                    "task_label": task.label,
                    "preprocess": task.preprocess,
                    "pipeline": pipeline,
                    "aug_config": config.label,
                    "n_synthetic": str(config.n_synthetic),
                    "perturbation_mode": config.perturbation_mode,
                    "noise_scale": str(config.noise_scale),
                    "inner_cv_rmse": fmt(result["rmse"]),
                    "inner_cv_r2": fmt(result["r2"]),
                    "best_params": str(result["params"]),
                }
            )

    summary_rows: list[dict[str, str]] = []
    for pipeline in PIPELINES:
        baseline = scores[(pipeline, "baseline")]
        candidates = [
            (config, scores[(pipeline, config.label)])
            for config in AUG_CONFIGS
            if config.is_augmented
        ]
        selected_config, selected = min(candidates, key=lambda item: float(item[1]["rmse"]))
        summary_rows.append(
            {
                "dataset": task.key,
                "task_label": task.label,
                "preprocess": task.preprocess,
                "train_size": str(int(train_idx.size)),
                "test_count": str(int(test_idx.size)),
                "pipeline": pipeline,
                "baseline_inner_cv_rmse": fmt(baseline["rmse"]),
                "baseline_inner_cv_r2": fmt(baseline["r2"]),
                "baseline_best_params": str(baseline["params"]),
                "selected_aug_config": selected_config.label,
                "selected_n_synthetic": str(selected_config.n_synthetic),
                "selected_perturbation_mode": selected_config.perturbation_mode,
                "selected_noise_scale": str(selected_config.noise_scale),
                "renvsa_inner_cv_rmse": fmt(selected["rmse"]),
                "renvsa_inner_cv_r2": fmt(selected["r2"]),
                "renvsa_best_params": str(selected["params"]),
                "delta_inner_cv_r2": fmt(selected["r2"] - baseline["r2"]),
                "delta_inner_cv_rmse": fmt(selected["rmse"] - baseline["rmse"]),
            }
        )
    return summary_rows, detail_rows


def load_task_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        _, X, y = load_spectrum_csv(path)
        return X, y
    except ValueError:
        frame = pd.read_csv(path)
        return frame.iloc[:, 2:].to_numpy(dtype=float), frame.iloc[:, 1].to_numpy(dtype=float).reshape(-1)


def build_raw_folds(
    X: np.ndarray,
    y: np.ndarray,
    config: AugConfig,
    cv: int = 5,
    random_state: int = 42,
) -> list[dict[str, np.ndarray]]:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    folds = []
    for fold_id, (fit_idx, valid_idx) in enumerate(splitter.split(X)):
        X_fit = X[fit_idx]
        y_fit = y[fit_idx]
        if config.is_augmented:
            augmenter = ResponseDrivenAugmenter(
                n_synthetic=config.n_synthetic,
                response_bins=6,
                neighbors=5,
                alpha_min=0.15,
                alpha_max=0.85,
                noise_scale=config.noise_scale,
                random_state=random_state + fold_id,
                neighbor_space="response",
                perturbation_mode=config.perturbation_mode,
            )
            result = augmenter.fit_resample(X_fit, y_fit)
            X_fit = result.X
            y_fit = result.y
        folds.append(
            {
                "X_fit": X_fit,
                "y_fit": y_fit,
                "X_valid": X[valid_idx],
                "y_valid": y[valid_idx],
            }
        )
    return folds


def build_cars_folds(raw_folds: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    cars_folds = []
    for fold_id, fold in enumerate(raw_folds):
        selector = CARSFeatureSelector(
            n_sampling=12,
            min_features=40,
            max_components=8,
            cv=3,
            random_state=42 + fold_id,
        )
        X_fit = selector.fit_transform(fold["X_fit"], fold["y_fit"])
        X_valid = selector.transform(fold["X_valid"])
        cars_folds.append(
            {
                "X_fit": X_fit,
                "y_fit": fold["y_fit"],
                "X_valid": X_valid,
                "y_valid": fold["y_valid"],
            }
        )
    return cars_folds


def select_by_inner_cv(pipeline: str, folds: list[dict[str, np.ndarray]]) -> dict[str, object]:
    best = None
    for params in parameter_grid(pipeline, folds):
        y_true_parts = []
        y_pred_parts = []
        for fold in folds:
            estimator = estimator_for(pipeline, params)
            estimator.fit(fold["X_fit"], fold["y_fit"])
            pred = estimator.predict(fold["X_valid"]).reshape(-1)
            y_true_parts.append(fold["y_valid"])
            y_pred_parts.append(pred)
        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        r2 = float(r2_score(y_true, y_pred))
        if best is None or rmse < best["rmse"]:
            best = {"rmse": rmse, "r2": r2, "params": params_to_string(params)}
    if best is None:
        raise RuntimeError(f"No valid parameter setting for {pipeline}")
    return best


def parameter_grid(pipeline: str, folds: list[dict[str, np.ndarray]]) -> list[dict[str, object]]:
    min_fit_n = min(fold["X_fit"].shape[0] for fold in folds)
    min_features = min(fold["X_fit"].shape[1] for fold in folds)
    if pipeline.endswith("PLSR"):
        upper = min(20, min_fit_n - 1, min_features)
        candidates = [2, 4, 6, 8, 10, 12, 15, 20]
        return [{"n_components": n} for n in candidates if n <= upper]
    if pipeline == "Ridge":
        return [{"alpha": value} for value in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]
    if pipeline.endswith("RBF-SVR"):
        c_values = [100.0, 1000.0, 3000.0] if pipeline == "RBF-SVR" else [1000.0, 3000.0, 10000.0]
        gamma_values = ["scale", 0.001, 0.003] if pipeline == "RBF-SVR" else [0.001, 0.003]
        epsilon_values = [0.05, 0.1, 0.2] if pipeline == "RBF-SVR" else [0.05, 0.1]
        return [
            {"C": c, "gamma": gamma, "epsilon": epsilon}
            for c, gamma, epsilon in itertools.product(c_values, gamma_values, epsilon_values)
        ]
    if pipeline == "CARS-RF":
        return [
            {"max_features": max_features, "min_samples_leaf": min_leaf}
            for max_features, min_leaf in itertools.product(["sqrt", 0.5], [1, 2])
        ]
    raise ValueError(f"Unknown pipeline: {pipeline}")


def estimator_for(pipeline: str, params: dict[str, object]):
    if pipeline.endswith("PLSR"):
        return PLSRegression(n_components=int(params["n_components"]))
    if pipeline == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(params["alpha"])))
    if pipeline.endswith("RBF-SVR"):
        return make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=float(params["C"]),
                gamma=params["gamma"],
                epsilon=float(params["epsilon"]),
            ),
        )
    if pipeline == "CARS-RF":
        return RandomForestRegressor(
            n_estimators=100,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")




def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def params_to_string(params: dict[str, object]) -> str:
    return ";".join(f"{key}={value}" for key, value in params.items())


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
