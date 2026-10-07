"""Post-hoc, calibration-only one-factor-at-a-time threshold sensitivity.

This is a diagnostic analysis, not evidence that the thresholds were originally
chosen before outer-holdout results were inspected. Existing results are read
only; outputs go to a new directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from run_renvsa import (
    ALGORITHM_SEED,
    DATASETS,
    INNER_CV,
    SVR_C,
    SVR_EPSILON,
    SVR_GAMMA,
    apply_fixed_cars_mask,
    fit_cars_measured_only,
    fit_preprocessor_measured_only,
    load_task,
    outer_split,
)
from respond_spectra import ResponseDrivenAugmenter


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results_threshold_oat_20260929"
BASE = {"theta_max": 0.18, "rho_min": 0.70, "delta_m": 0.08}
LEVELS = {
    "theta_max": (0.15, 0.18, 0.21),
    "rho_min": (0.65, 0.70, 0.75),
    "delta_m": (0.06, 0.08, 0.10),
}
# Calibration-only preprocessing choices from the existing full-spectrum SPXY run.
PREPROCESSING = {
    "coal_Q": "sg",
    "coal_ash": "msc",
    "soil_SOC": "sg",
    "diesel_CN": "snv-sg",
    "diesel_FREEZE": "sg",
}
N_SYNTHETIC = 80
NOISE_SCALE = 0.008


def conditions() -> list[tuple[str, float, dict[str, float]]]:
    output = [("baseline", 0.0, BASE.copy())]
    for name, values in LEVELS.items():
        for value in values:
            if value == BASE[name]:
                continue
            thresholds = BASE.copy()
            thresholds[name] = value
            output.append((name, value, thresholds))
    assert len(output) == 7
    return output


def make_augmenter(thresholds: dict[str, float]) -> ResponseDrivenAugmenter:
    return ResponseDrivenAugmenter(
        n_synthetic=N_SYNTHETIC,
        response_bins=6,
        response_bin_strategy="quantile",
        neighbors=5,
        alpha_min=0.15,
        alpha_max=0.85,
        noise_scale=NOISE_SCALE,
        max_spectral_angle=thresholds["theta_max"],
        min_derivative_corr=thresholds["rho_min"],
        envelope_margin=thresholds["delta_m"],
        random_state=ALGORITHM_SEED,
        neighbor_space="response",
        perturbation_mode="local_std",
    )


def baseline_search(X: np.ndarray, y: np.ndarray) -> GridSearchCV:
    cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=ALGORITHM_SEED)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        {
            "svr__C": list(SVR_C),
            "svr__gamma": list(SVR_GAMMA),
            "svr__epsilon": list(SVR_EPSILON),
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    return search


def run_task(task: str) -> list[dict]:
    _, X, y = load_task(task)
    cal_idx, _, _ = outer_split(X, y, "spxy", None)
    X_cal, y_cal = X[cal_idx].copy(), y[cal_idx].copy()
    preprocessing = PREPROCESSING[task]
    outer_cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=ALGORITHM_SEED)
    variants = conditions()
    predictions = {
        factor if factor == "baseline" else f"{factor}_{value:.2f}": np.full(y_cal.shape, np.nan)
        for factor, value, _ in variants
    }
    measured_only_prediction = np.full(y_cal.shape, np.nan)
    fold_rows = []

    for fold, (fit_idx, valid_idx) in enumerate(outer_cv.split(X_cal), start=1):
        started = time.perf_counter()
        X_fit_raw, y_fit = X_cal[fit_idx], y_cal[fit_idx]
        preprocessor = fit_preprocessor_measured_only(
            X_fit_raw, preprocessing, expected_measured_count=len(fit_idx)
        )
        X_fit = preprocessor.transform(X_fit_raw)
        X_valid = preprocessor.transform(X_cal[valid_idx])
        selector = fit_cars_measured_only(
            X_fit, y_fit,
            expected_measured_count=len(fit_idx),
            forbidden_synthetic_count=N_SYNTHETIC,
        )
        mask = np.asarray(selector.selected_indices_, dtype=int)
        Z_fit = X_fit[:, mask]
        Z_valid = X_valid[:, mask]
        search = baseline_search(Z_fit, y_fit)
        baseline_params = search.best_params_
        baseline_prediction = search.predict(Z_valid)
        measured_only_prediction[valid_idx] = baseline_prediction

        for factor, value, thresholds in variants:
            augmentation = make_augmenter(thresholds).fit_resample(X_fit, y_fit)
            X_syn = augmentation.X[augmentation.synthetic_mask]
            y_syn = augmentation.y[augmentation.synthetic_mask]
            Z_measured, Z_synthetic, Z_evaluation = apply_fixed_cars_mask(
                mask, X_fit, X_syn, X_valid, X.shape[1]
            )
            model = clone(search.best_estimator_)
            model.fit(
                np.vstack([Z_measured, Z_synthetic]),
                np.concatenate([y_fit, y_syn]),
            )
            prediction = model.predict(Z_evaluation)
            key = factor if factor == "baseline" else f"{factor}_{value:.2f}"
            predictions[key][valid_idx] = prediction
            rejection = augmentation.metadata["rejections"]
            fold_rows.append({
                "task": task,
                "fold": fold,
                "factor": factor,
                "tested_value": value,
                **thresholds,
                "n_calibration": len(cal_idx),
                "n_fit": len(fit_idx),
                "n_valid": len(valid_idx),
                "preprocessing": preprocessing,
                "n_synthetic_requested": N_SYNTHETIC,
                "n_synthetic_accepted": augmentation.n_synthetic,
                "attempted": augmentation.metadata["attempted"],
                "acceptance_rate": augmentation.metadata["acceptance_rate"],
                "rejected_angle": rejection.get("spectral_angle", 0),
                "rejected_derivative": rejection.get("derivative_corr", 0),
                "rejected_envelope": rejection.get("envelope", 0),
                "n_cars_features": len(mask),
                "svr_C": baseline_params["svr__C"],
                "svr_gamma": baseline_params["svr__gamma"],
                "svr_epsilon": baseline_params["svr__epsilon"],
                "fold_rmse": mean_squared_error(y_cal[valid_idx], prediction) ** 0.5,
                "fold_r2": r2_score(y_cal[valid_idx], prediction),
                "baseline_fold_rmse": mean_squared_error(y_cal[valid_idx], baseline_prediction) ** 0.5,
            })
        print(f"{task} fold {fold}/5 complete in {time.perf_counter()-started:.1f}s", flush=True)

    if any(not np.isfinite(value).all() for value in predictions.values()) or not np.isfinite(measured_only_prediction).all():
        raise RuntimeError(f"Missing out-of-fold predictions for {task}")
    base_rmse = mean_squared_error(y_cal, measured_only_prediction) ** 0.5
    rows = []
    for factor, value, thresholds in variants:
        key = factor if factor == "baseline" else f"{factor}_{value:.2f}"
        group = [row for row in fold_rows if row["factor"] == factor and row["tested_value"] == value]
        prediction = predictions[key]
        rmse = mean_squared_error(y_cal, prediction) ** 0.5
        rows.append({
            "task": task,
            "factor": factor,
            "tested_value": value,
            **thresholds,
            "n_calibration": len(cal_idx),
            "preprocessing": preprocessing,
            "n_synthetic_requested": N_SYNTHETIC,
            "noise_scale": NOISE_SCALE,
            "perturbation_mode": "local_std",
            "pooled_oof_rmse": rmse,
            "pooled_oof_r2": r2_score(y_cal, prediction),
            "delta_rmse_vs_base_thresholds": rmse - mean_squared_error(y_cal, predictions["baseline"]) ** 0.5,
            "measured_only_pooled_oof_rmse": base_rmse,
            "delta_rmse_vs_measured_only": rmse - base_rmse,
            "mean_acceptance_rate": float(np.mean([row["acceptance_rate"] for row in group])),
            "total_attempted": sum(row["attempted"] for row in group),
            "total_rejected_angle": sum(row["rejected_angle"] for row in group),
            "total_rejected_derivative": sum(row["rejected_derivative"] for row in group),
            "total_rejected_envelope": sum(row["rejected_envelope"] for row in group),
        })
    return rows, fold_rows


def write_csv_exclusive(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    manifest = {
        "design": "post_hoc_calibration_only_one_factor_at_a_time",
        "outer_split": "SPXY 75/25; holdout observations are never accessed after splitting",
        "levels": LEVELS,
        "base_thresholds": BASE,
        "preprocessing": PREPROCESSING,
        "n_synthetic": N_SYNTHETIC,
        "perturbation_mode": "local_std",
        "noise_scale": NOISE_SCALE,
        "selection_rule": "No thresholds are selected; compare pooled out-of-fold calibration RMSE, R2 and filter acceptance",
        "svr_rule": "Within each fold, tune on measured fit samples once; reuse those hyperparameters for all thresholds",
        "cars_rule": "Within each fold, fit on measured fit samples once; reuse the same mask for all thresholds",
        "algorithm_seed": ALGORITHM_SEED,
        "tasks": args.tasks,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_rows = []
    for task in args.tasks:
        rows, fold_rows = run_task(task)
        write_csv_exclusive(args.output_dir / f"{task}_summary.csv", rows)
        write_csv_exclusive(args.output_dir / f"{task}_folds.csv", fold_rows)
        all_rows.extend(rows)
    write_csv_exclusive(args.output_dir / "summary.csv", all_rows)
    print(f"completed {len(args.tasks)} tasks; wrote {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
