"""Post-hoc 27-threshold sensitivity on the paper's outer SPXY holdouts.

This script evaluates, but never selects, threshold combinations using the
holdout set. Its outputs must not be described as pre-holdout parameter tuning.
All output is isolated from the manuscript's existing results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "examples"))

from holdout_bp_ks_spxy import representative_split
from run_coal_q_augmentation_demo import preprocess_spectra
from run_renvsa import ALGORITHM_SEED, load_task
from analyze_threshold_grid import BASE, GRID, write_csv_exclusive
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter


DEFAULT_OUTPUT = HERE / "results_analyze_threshold_holdout_20260929"


@dataclass(frozen=True)
class TaskProtocol:
    preprocess: str
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float
    svr_C: tuple[float, ...]
    svr_gamma: tuple[float, ...]
    svr_epsilon: tuple[float, ...]
    source_file: str
    source_dataset: str = ""


PROTOCOLS = {
    "coal_Q": TaskProtocol(
        "raw", 80, "none", 0.0,
        (3000.0, 5000.0, 10000.0), (0.0015, 0.002, 0.003, 0.005), (0.05, 0.1, 0.15),
        "holdout_unified_trainonly_primary.csv", "coal_Q",
    ),
    "coal_ash": TaskProtocol(
        "raw", 40, "local_std", 0.008,
        (3000.0, 5000.0, 10000.0), (0.0015, 0.002, 0.003, 0.005), (0.05, 0.1, 0.15),
        "holdout_unified_trainonly_coal_ash_raw.csv",
    ),
    "soil_SOC": TaskProtocol(
        "snv-sg", 80, "local_std", 0.008,
        (1000.0,), (0.003,), (0.1,),
        "final_soil_snvsg_spxy25_locked_protocol_summary.csv",
    ),
    "diesel_CN": TaskProtocol(
        "snv-sg", 80, "none", 0.0,
        (3000.0, 5000.0, 10000.0), (0.0005, 0.001, 0.0015, 0.002), (0.05, 0.1, 0.15),
        "final_diesel_cn_snvsg_n80_none_for_violin_summary.csv",
    ),
    "diesel_FREEZE": TaskProtocol(
        "snv", 40, "local_std", 0.008,
        (3000.0, 5000.0, 10000.0), (0.0015, 0.002, 0.003, 0.005), (0.05, 0.1, 0.15),
        "final_diesel_freeze_snv_cars_n40_localstd_for_violin_summary.csv",
    ),
}


def fit_svr(X: np.ndarray, y: np.ndarray, protocol: TaskProtocol) -> GridSearchCV:
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        {
            "svr__C": protocol.svr_C,
            "svr__gamma": protocol.svr_gamma,
            "svr__epsilon": protocol.svr_epsilon,
        },
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=ALGORITHM_SEED),
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    return search


def augment(Z: np.ndarray, y: np.ndarray, protocol: TaskProtocol,
            thresholds: dict[str, float]):
    return ResponseDrivenAugmenter(
        n_synthetic=protocol.n_synthetic,
        response_bins=6,
        response_bin_strategy="quantile",
        neighbors=5,
        alpha_min=0.15,
        alpha_max=0.85,
        noise_scale=protocol.noise_scale,
        max_spectral_angle=thresholds["theta_max"],
        min_derivative_corr=thresholds["rho_min"],
        envelope_margin=thresholds["delta_m"],
        random_state=ALGORITHM_SEED,
        neighbor_space="response",
        perturbation_mode=protocol.perturbation_mode,
    ).fit_resample(Z, y)


def archived_metrics(protocol: TaskProtocol, task: str) -> tuple[float, float]:
    with (PROJECT / "results" / protocol.source_file).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        item for item in rows
        if not protocol.source_dataset or item.get("dataset") == protocol.source_dataset
    )
    if "baseline_test_rmse" in row:
        return float(row["baseline_test_rmse"]), float(row["selected_test_rmse"])
    return float(row["baseline_rmse"]), float(row["augmented_rmse"])


def run_task(task: str) -> tuple[list[dict], list[dict], dict]:
    started = time.perf_counter()
    protocol = PROTOCOLS[task]
    _, X_raw, y = load_task(task)
    X = preprocess_spectra(X_raw, protocol.preprocess)
    cal_idx, hold_idx = representative_split(
        X, y, test_size=0.25, method="spxy", spxy_y_weight=1.0
    )
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_hold, y_hold = X[hold_idx], y[hold_idx]
    cars = CARSFeatureSelector(
        n_sampling=40, min_features=40, max_components=8,
        cv=5, random_state=ALGORITHM_SEED,
    )
    Z_cal = cars.fit_transform(X_cal, y_cal)
    Z_hold = cars.transform(X_hold)
    baseline_model = fit_svr(Z_cal, y_cal, protocol)
    baseline_prediction = baseline_model.predict(Z_hold)
    baseline_rmsep = mean_squared_error(y_hold, baseline_prediction) ** 0.5
    fitted_by_hash: dict[str, tuple[np.ndarray, dict]] = {}
    rows: list[dict] = []
    prediction_rows: list[dict] = []

    for grid_id, thresholds in enumerate(GRID, start=1):
        result = augment(Z_cal, y_cal, protocol, thresholds)
        key = hashlib.sha256(result.X.tobytes() + result.y.tobytes()).hexdigest()
        if key not in fitted_by_hash:
            model = fit_svr(result.X, result.y, protocol)
            fitted_by_hash[key] = (model.predict(Z_hold), model.best_params_)
        prediction, best_params = fitted_by_hash[key]
        rmsep = mean_squared_error(y_hold, prediction) ** 0.5
        rejected = result.metadata["rejections"]
        rows.append({
            "task": task,
            "grid_id": grid_id,
            **thresholds,
            "is_original_thresholds": int(thresholds == BASE),
            "n_calibration": len(cal_idx),
            "n_holdout": len(hold_idx),
            "preprocessing": protocol.preprocess,
            "n_synthetic_requested": protocol.n_synthetic,
            "perturbation_mode": protocol.perturbation_mode,
            "noise_scale": protocol.noise_scale,
            "n_synthetic_accepted": result.n_synthetic,
            "acceptance_rate": result.metadata["acceptance_rate"],
            "rejected_angle": rejected.get("spectral_angle", 0),
            "rejected_derivative": rejected.get("derivative_corr", 0),
            "rejected_envelope": rejected.get("envelope", 0),
            "n_cars_features": Z_cal.shape[1],
            "baseline_rmsep": baseline_rmsep,
            "rensa_rmsep": rmsep,
            "delta_rmsep_vs_measured_only": rmsep - baseline_rmsep,
            "baseline_r2": r2_score(y_hold, baseline_prediction),
            "rensa_r2": r2_score(y_hold, prediction),
            "svr_C": best_params["svr__C"],
            "svr_gamma": best_params["svr__gamma"],
            "svr_epsilon": best_params["svr__epsilon"],
        })
        for sample_idx, true, base, augmented in zip(hold_idx, y_hold, baseline_prediction, prediction):
            prediction_rows.append({
                "task": task, "grid_id": grid_id, "sample_index": int(sample_idx),
                "y_true": float(true), "baseline_prediction": float(base),
                "rensa_prediction": float(augmented),
            })

    original = next(row for row in rows if row["is_original_thresholds"] == 1)
    source_base, source_rensa = archived_metrics(protocol, task)
    validation = {
        "task": task,
        "source_file": protocol.source_file,
        "source_baseline_rmsep": source_base,
        "reproduced_baseline_rmsep": baseline_rmsep,
        "source_rensa_rmsep": source_rensa,
        "reproduced_rensa_rmsep": original["rensa_rmsep"],
        "baseline_abs_error": abs(baseline_rmsep - source_base),
        "rensa_abs_error": abs(original["rensa_rmsep"] - source_rensa),
        "n_unique_augmented_training_sets": len(fitted_by_hash),
    }
    print(
        f"{task}: 27 grids; unique fits={len(fitted_by_hash)}; "
        f"base={baseline_rmsep:.4f} vs {source_base:.4f}; "
        f"original RenSA={original['rensa_rmsep']:.4f} vs {source_rensa:.4f}; "
        f"{time.perf_counter()-started:.1f}s",
        flush=True,
    )
    return rows, prediction_rows, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "design": "post_hoc_outer_holdout_threshold_sensitivity_not_parameter_selection",
        "grid": GRID,
        "protocols": {task: asdict(config) for task, config in PROTOCOLS.items()},
        "split": "SPXY 75/25 on task-specific preprocessed measured data",
        "cars": "fit once on measured outer calibration samples and shared across baseline/grid",
        "svr": "tuned separately for each unique augmented calibration set",
        "warning": "The outer-holdout grid must not be used to claim pre-holdout threshold selection",
    }, indent=2), encoding="utf-8")
    all_rows: list[dict] = []
    validations: list[dict] = []
    for task in args.tasks:
        rows, predictions, validation = run_task(task)
        write_csv_exclusive(args.output_dir / f"{task}_summary.csv", rows)
        write_csv_exclusive(args.output_dir / f"{task}_predictions.csv", predictions)
        all_rows.extend(rows)
        validations.append(validation)
    write_csv_exclusive(args.output_dir / "summary.csv", all_rows)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validations, indent=2), encoding="utf-8"
    )
    print(f"completed {len(args.tasks)} tasks and {len(all_rows)} grid rows", flush=True)


if __name__ == "__main__":
    main()
