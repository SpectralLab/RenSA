"""Calibration-only 3 x 3 x 3 threshold sensitivity grid.

Post-hoc diagnostic. Reuses the same measured-only CARS mask and measured-only
SVR tuning within each fold. Writes only to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from run_renvsa import (
    ALGORITHM_SEED, DATASETS, INNER_CV, apply_fixed_cars_mask,
    fit_cars_measured_only, fit_preprocessor_measured_only, load_task, outer_split,
)
from analyze_threshold_one_at_a_time import (
    BASE, LEVELS, N_SYNTHETIC, NOISE_SCALE, PREPROCESSING,
    baseline_search, make_augmenter, write_csv_exclusive,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_threshold_grid27_20260929"
GRID = [
    {"theta_max": theta, "rho_min": rho, "delta_m": delta}
    for theta, rho, delta in product(
        LEVELS["theta_max"], LEVELS["rho_min"], LEVELS["delta_m"]
    )
]
assert len(GRID) == 27 and BASE in GRID


def run_task(task: str) -> tuple[list[dict], list[dict]]:
    _, X, y = load_task(task)
    cal_idx, _, _ = outer_split(X, y, "spxy", None)
    X_cal, y_cal = X[cal_idx].copy(), y[cal_idx].copy()
    preprocessing = PREPROCESSING[task]
    outer_cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=ALGORITHM_SEED)
    predictions = np.full((27, len(y_cal)), np.nan)
    measured_prediction = np.full(len(y_cal), np.nan)
    fold_rows: list[dict] = []

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
        Z_fit, Z_valid = X_fit[:, mask], X_valid[:, mask]
        search = baseline_search(Z_fit, y_fit)
        measured_prediction[valid_idx] = search.predict(Z_valid)

        for grid_id, thresholds in enumerate(GRID, start=1):
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
            pred = model.predict(Z_evaluation)
            predictions[grid_id - 1, valid_idx] = pred
            rejected = augmentation.metadata["rejections"]
            fold_rows.append({
                "task": task,
                "fold": fold,
                "grid_id": grid_id,
                **thresholds,
                "n_fit": len(fit_idx),
                "n_valid": len(valid_idx),
                "n_cars_features": len(mask),
                "n_synthetic_accepted": augmentation.n_synthetic,
                "attempted": augmentation.metadata["attempted"],
                "acceptance_rate": augmentation.metadata["acceptance_rate"],
                "rejected_angle": rejected.get("spectral_angle", 0),
                "rejected_derivative": rejected.get("derivative_corr", 0),
                "rejected_envelope": rejected.get("envelope", 0),
                "fold_rmse": mean_squared_error(y_cal[valid_idx], pred) ** 0.5,
            })
        print(f"{task} fold {fold}/5 completed in {time.perf_counter()-started:.1f}s", flush=True)

    if not np.isfinite(predictions).all() or not np.isfinite(measured_prediction).all():
        raise RuntimeError(f"Missing out-of-fold prediction for {task}")
    measured_rmse = mean_squared_error(y_cal, measured_prediction) ** 0.5
    base_id = next(i for i, thresholds in enumerate(GRID) if thresholds == BASE)
    base_rmse = mean_squared_error(y_cal, predictions[base_id]) ** 0.5
    summary: list[dict] = []
    for grid_id, thresholds in enumerate(GRID, start=1):
        group = [row for row in fold_rows if row["grid_id"] == grid_id]
        rmse = mean_squared_error(y_cal, predictions[grid_id - 1]) ** 0.5
        summary.append({
            "task": task,
            "grid_id": grid_id,
            **thresholds,
            "is_original_thresholds": int(thresholds == BASE),
            "n_calibration": len(cal_idx),
            "preprocessing": preprocessing,
            "n_synthetic_requested": N_SYNTHETIC,
            "noise_scale": NOISE_SCALE,
            "pooled_oof_rmse": rmse,
            "pooled_oof_r2": r2_score(y_cal, predictions[grid_id - 1]),
            "measured_only_pooled_oof_rmse": measured_rmse,
            "delta_rmse_vs_measured_only": rmse - measured_rmse,
            "delta_rmse_vs_original_thresholds": rmse - base_rmse,
            "mean_acceptance_rate": float(np.mean([row["acceptance_rate"] for row in group])),
            "total_attempted": sum(row["attempted"] for row in group),
            "total_rejected_angle": sum(row["rejected_angle"] for row in group),
            "total_rejected_derivative": sum(row["rejected_derivative"] for row in group),
            "total_rejected_envelope": sum(row["rejected_envelope"] for row in group),
        })
    return summary, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    manifest = {
        "design": "post_hoc_calibration_only_full_factorial_3x3x3",
        "grid": GRID,
        "outer_split": "SPXY 75/25; only calibration observations enter the sensitivity analysis",
        "preprocessing": PREPROCESSING,
        "n_synthetic": N_SYNTHETIC,
        "perturbation_mode": "local_std",
        "noise_scale": NOISE_SCALE,
        "cars_rule": "Fitted once per fold on measured fit samples; shared across all 27 combinations",
        "svr_rule": "Tuned once per fold on measured fit samples; shared across all 27 combinations",
        "ranking_rule": "Lower pooled five-fold calibration RMSE; no outer-holdout metric used",
        "algorithm_seed": ALGORITHM_SEED,
        "tasks": args.tasks,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_rows: list[dict] = []
    for task in args.tasks:
        rows, folds = run_task(task)
        write_csv_exclusive(args.output_dir / f"{task}_summary.csv", rows)
        write_csv_exclusive(args.output_dir / f"{task}_folds.csv", folds)
        all_rows.extend(rows)
    write_csv_exclusive(args.output_dir / "summary.csv", all_rows)
    print(f"completed {len(args.tasks)} tasks and {len(all_rows)} task-grid combinations", flush=True)


if __name__ == "__main__":
    main()
