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
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import ResponseDrivenAugmenter, load_spectrum_csv
from run_coal_q_augmentation_demo import preprocess_spectra


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    data: str
    preprocess: str
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float


TASKS = [
    TaskSpec("coal_Q", "Coal GCV", "coal_Q.csv", "raw", 80, "none", 0.0),
    TaskSpec("coal_ash", "Coal ash", "coal_ash.csv", "raw", 40, "local_std", 0.008),
    TaskSpec("soil_SOC", "Soil SOC", "soil_SOC.csv", "snv-sg", 80, "local_std", 0.008),
    TaskSpec("diesel_CN", "Diesel CN", "diesel_CN.csv", "snv-sg", 80, "none", 0.0),
    TaskSpec("diesel_FREEZE", "Diesel freeze", "diesel_FREEZE.csv", "snv", 40, "local_std", 0.008),
]


MODEL_ORDER = ["PLSR", "Ridge", "RBF-SVR", "RF", "ExtraTrees"]
MODEL_COLORS = {
    "PLSR": "#3568A8",
    "Ridge": "#6F6F6F",
    "RBF-SVR": "#7B5BB4",
    "RF": "#3E8A5A",
    "ExtraTrees": "#C65A2E",
}


FIELDNAMES = [
    "dataset",
    "task_label",
    "preprocess",
    "train_size",
    "test_count",
    "model",
    "baseline_inner_cv_rmse",
    "baseline_inner_cv_r2",
    "baseline_best_params",
    "renvsa_inner_cv_rmse",
    "renvsa_inner_cv_r2",
    "renvsa_best_params",
    "delta_inner_cv_r2",
    "delta_inner_cv_rmse",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "publication_coal_diesel"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "table_model_sensitivity_inner_cv.csv"

    started = time.perf_counter()
    rows = []
    for task in TASKS:
        print(f"task {task.label}: preprocess={task.preprocess}, n_syn={task.n_synthetic}", flush=True)
        rows.extend(evaluate_task(root, task))

    write_csv(csv_path, rows)
    print(f"wrote {csv_path}", flush=True)
    print(f"elapsed_sec={time.perf_counter() - started:.1f}", flush=True)


def evaluate_task(root: Path, task: TaskSpec) -> list[dict[str, str]]:
    X, y = load_task_data(root / "data" / task.data)
    X = preprocess_spectra(X, task.preprocess)
    train_idx, test_idx = representative_split(X, y, test_size=0.25, method="spxy", spxy_y_weight=1.0)
    X_train = X[train_idx]
    y_train = y[train_idx]

    baseline_folds = build_fold_sets(X_train, y_train, augmented=False, task=task)
    augmented_folds = build_fold_sets(X_train, y_train, augmented=True, task=task)

    rows = []
    for model_name in MODEL_ORDER:
        print(f"  model {model_name}", flush=True)
        baseline = select_by_inner_cv(model_name, baseline_folds)
        augmented = select_by_inner_cv(model_name, augmented_folds)
        rows.append(
            {
                "dataset": task.key,
                "task_label": task.label,
                "preprocess": task.preprocess,
                "train_size": str(int(train_idx.size)),
                "test_count": str(int(test_idx.size)),
                "model": model_name,
                "baseline_inner_cv_rmse": fmt(baseline["rmse"]),
                "baseline_inner_cv_r2": fmt(baseline["r2"]),
                "baseline_best_params": baseline["params"],
                "renvsa_inner_cv_rmse": fmt(augmented["rmse"]),
                "renvsa_inner_cv_r2": fmt(augmented["r2"]),
                "renvsa_best_params": augmented["params"],
                "delta_inner_cv_r2": fmt(augmented["r2"] - baseline["r2"]),
                "delta_inner_cv_rmse": fmt(augmented["rmse"] - baseline["rmse"]),
                "n_synthetic": str(task.n_synthetic),
                "perturbation_mode": task.perturbation_mode,
                "noise_scale": str(task.noise_scale),
            }
        )
    return rows


def load_task_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        _, X, y = load_spectrum_csv(path)
        return X, y
    except ValueError:
        frame = pd.read_csv(path)
        target = frame.iloc[:, 1].to_numpy(dtype=float).reshape(-1)
        spectra = frame.iloc[:, 2:].to_numpy(dtype=float)
        return spectra, target


def build_fold_sets(
    X: np.ndarray,
    y: np.ndarray,
    augmented: bool,
    task: TaskSpec,
    cv: int = 5,
    random_state: int = 42,
) -> list[dict[str, np.ndarray]]:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_sets = []
    for fold_id, (fit_idx, valid_idx) in enumerate(splitter.split(X)):
        X_fit = X[fit_idx]
        y_fit = y[fit_idx]
        if augmented:
            augmenter = ResponseDrivenAugmenter(
                n_synthetic=task.n_synthetic,
                response_bins=6,
                neighbors=5,
                alpha_min=0.15,
                alpha_max=0.85,
                noise_scale=task.noise_scale,
                random_state=random_state + fold_id,
                neighbor_space="response",
                perturbation_mode=task.perturbation_mode,
            )
            result = augmenter.fit_resample(X_fit, y_fit)
            X_fit = result.X
            y_fit = result.y
        fold_sets.append(
            {
                "X_fit": X_fit,
                "y_fit": y_fit,
                "X_valid": X[valid_idx],
                "y_valid": y[valid_idx],
                "valid_idx": valid_idx,
            }
        )
    return fold_sets


def select_by_inner_cv(model_name: str, folds: list[dict[str, np.ndarray]]) -> dict[str, object]:
    best = None
    for params in parameter_grid(model_name, folds):
        y_true_parts = []
        y_pred_parts = []
        for fold in folds:
            estimator = estimator_for(model_name, params)
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
        raise RuntimeError(f"No valid parameter setting for {model_name}")
    return best


def parameter_grid(model_name: str, folds: list[dict[str, np.ndarray]]) -> list[dict[str, object]]:
    min_fit_n = min(fold["X_fit"].shape[0] for fold in folds)
    n_features = min(fold["X_fit"].shape[1] for fold in folds)
    if model_name == "PLSR":
        candidates = [2, 4, 6, 8, 10, 12, 15, 20]
        upper = min(20, min_fit_n - 1, n_features)
        return [{"n_components": n} for n in candidates if n <= upper]
    if model_name == "Ridge":
        return [{"alpha": value} for value in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]
    if model_name == "RBF-SVR":
        return [
            {"C": c, "gamma": gamma, "epsilon": eps}
            for c, gamma, eps in itertools.product(
                [10.0, 100.0, 1000.0],
                ["scale", 0.001, 0.003, 0.01],
                [0.05, 0.1, 0.2],
            )
        ]
    if model_name in {"RF", "ExtraTrees"}:
        return [
            {"max_features": max_features, "min_samples_leaf": min_leaf}
            for max_features, min_leaf in itertools.product(["sqrt", 0.5], [1, 2])
        ]
    raise ValueError(f"Unknown model: {model_name}")


def estimator_for(model_name: str, params: dict[str, object]):
    if model_name == "PLSR":
        return PLSRegression(n_components=int(params["n_components"]))
    if model_name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(params["alpha"])))
    if model_name == "RBF-SVR":
        return make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=float(params["C"]),
                gamma=params["gamma"],
                epsilon=float(params["epsilon"]),
            ),
        )
    if model_name == "RF":
        return RandomForestRegressor(
            n_estimators=100,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=100,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")




def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
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
