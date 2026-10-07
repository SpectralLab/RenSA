from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from holdout_unified_train_only_protocol import fit_preprocessor
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv


LOGGER = logging.getLogger("cross_split_robustness")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    data_file: str
    target_column: str
    spectral_prefix: str


@dataclass(frozen=True)
class AugConfig:
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float

    @property
    def key(self) -> str:
        return f"n{self.n_synthetic}_{self.perturbation_mode}_noise{self.noise_scale:g}"


TASKS = {
    spec.key: spec
    for spec in (
        TaskSpec("coal_Q", "Coal GCV", "coal_Q.csv", "y", "x"),
        TaskSpec("coal_ash", "Coal Ash", "coal_ash.csv", "y", "x"),
        TaskSpec("soil_SOC", "Soil SOC", "soil_SOC.csv", "y", "x"),
        TaskSpec("diesel_CN", "Diesel CN", "diesel_CN.csv", "CN", ""),
        TaskSpec("diesel_FREEZE", "Diesel FREEZE", "diesel_FREEZE.csv", "FREEZE", ""),
    )
}


RESULT_FIELDS = [
    "task",
    "task_label",
    "split_method",
    "outer_seed",
    "requested_strata",
    "used_strata",
    "n_calibration",
    "n_holdout",
    "algorithm_seed",
    "inner_cv",
    "selected_preprocess",
    "selected_n_synthetic",
    "selected_perturbation_mode",
    "selected_noise_scale",
    "selected_inner_cv_rmsep",
    "selected_inner_cv_r2",
    "baseline_r2",
    "rensa_r2",
    "delta_r2",
    "baseline_rmsep",
    "rensa_rmsep",
    "delta_rmsep",
    "baseline_best_params",
    "rensa_best_params",
    "cars_selected_features",
    "cars_feature_hash",
    "paired_feature_set",
    "synthetic_requested",
    "synthetic_accepted",
    "acceptance_rate",
    "sparsity_median_5nn",
    "sparsity_p90_5nn",
    "sparsity_max_5nn",
    "calibration_indices",
    "holdout_indices",
    "elapsed_sec",
]

Y_STAT_FIELDS = [
    "task",
    "task_label",
    "split_method",
    "outer_seed",
    "subset",
    "n",
    "min",
    "q1",
    "median",
    "mean",
    "q3",
    "max",
    "sd",
]

SPARSITY_FIELDS = [
    "task",
    "task_label",
    "split_method",
    "outer_seed",
    "n_calibration",
    "sparsity_median_5nn",
    "sparsity_p90_5nn",
    "sparsity_max_5nn",
]

PREDICTION_FIELDS = [
    "task",
    "task_label",
    "split_method",
    "outer_seed",
    "sample_index",
    "y_true",
    "baseline_prediction",
    "rensa_prediction",
    "baseline_residual",
    "rensa_residual",
]

SELECTION_FIELDS = [
    "task",
    "split_method",
    "outer_seed",
    "rank",
    "selected",
    "preprocess",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "inner_cv_rmsep",
    "inner_cv_r2",
]

FAILURE_FIELDS = ["task", "split_method", "outer_seed", "error_type", "error_message", "elapsed_sec"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Paired RenSA cross-split robustness experiment.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "cross_split_robustness")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--split-method", choices=["mc", "ks", "both", "spxy-stats"], default="both")
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--preprocess-candidates", nargs="+", default=["raw", "snv", "sg", "snv-sg", "msc", "msc-sg"])
    parser.add_argument("--n-synthetic", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--perturbation-mode", nargs="+", choices=["none", "local_std"], default=["none", "local_std"])
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0, 0.008])
    parser.add_argument("--neighbor-space", choices=["response"], default="response")
    parser.add_argument("--response-bin-strategy", choices=["quantile"], default="quantile")
    parser.add_argument("--response-bins", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha-min", type=float, default=0.15)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--algorithm-seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_file or args.output_dir / "run.log")
    configs = augmentation_configs(args)
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }
    write_manifest(args, configs, param_grid)

    if args.split_method == "spxy-stats":
        write_spxy_statistics(args)
        return

    methods = ["mc", "ks"] if args.split_method == "both" else [args.split_method]
    failures = 0
    for task_key in args.tasks:
        spec = TASKS[task_key]
        _, X, y = load_spectrum_csv(
            args.data_dir / spec.data_file,
            target_column=spec.target_column,
            spectral_prefix=spec.spectral_prefix,
        )
        for method in methods:
            seeds = args.outer_seeds if method == "mc" else [-1]
            for outer_seed in seeds:
                if args.resume and is_completed(args.output_dir, method, task_key, outer_seed):
                    LOGGER.info("skip completed task=%s method=%s seed=%s", task_key, method, outer_seed)
                    continue
                started = time.perf_counter()
                try:
                    run_one_split(spec, X, y, method, outer_seed, configs, param_grid, args)
                except Exception as exc:  # failures are recorded and surfaced after all requested runs
                    failures += 1
                    append_rows(
                        args.output_dir / "failures.csv",
                        [{
                            "task": task_key,
                            "split_method": method_name(method),
                            "outer_seed": outer_seed if method == "mc" else "",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                        }],
                        FAILURE_FIELDS,
                    )
                    LOGGER.exception("failed task=%s method=%s seed=%s", task_key, method, outer_seed)

    refresh_derived_outputs(args.output_dir, args.bootstrap, args.algorithm_seed)
    if failures:
        raise SystemExit(f"{failures} requested split(s) failed; see failures.csv and run.log")


def run_one_split(
    spec: TaskSpec,
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    outer_seed: int,
    configs: list[AugConfig],
    param_grid: dict,
    args: argparse.Namespace,
) -> None:
    started = time.perf_counter()
    if method == "mc":
        train_idx, test_idx, used_strata = stratified_regression_split(
            y, args.test_size, outer_seed, args.strata
        )
    else:
        train_idx, test_idx = representative_split(
            X, y, test_size=args.test_size, method="ks", spxy_y_weight=1.0
        )
        used_strata = ""

    split_label = method_name(method)
    X_cal, y_cal = X[train_idx], y[train_idx]
    X_hold, y_hold = X[test_idx], y[test_idx]
    LOGGER.info(
        "start task=%s method=%s seed=%s calibration=%d holdout=%d candidates=%d",
        spec.key,
        split_label,
        outer_seed if method == "mc" else "deterministic",
        len(train_idx),
        len(test_idx),
        len(args.preprocess_candidates) * len(configs),
    )

    selected, selection_rows = select_rensa_candidate(
        X_cal, y_cal, configs, param_grid, args, spec.key, split_label, outer_seed
    )
    final = paired_final_fit(X_cal, y_cal, X_hold, y_hold, selected, param_grid, args)
    sparsity = response_sparsity(y_cal, neighbors=5)
    elapsed = round(time.perf_counter() - started, 3)

    result_row = {
        "task": spec.key,
        "task_label": spec.label,
        "split_method": split_label,
        "outer_seed": outer_seed if method == "mc" else "",
        "requested_strata": args.strata if method == "mc" else "",
        "used_strata": used_strata,
        "n_calibration": len(train_idx),
        "n_holdout": len(test_idx),
        "algorithm_seed": args.algorithm_seed,
        "inner_cv": args.inner_cv,
        "selected_preprocess": selected["preprocess"],
        "selected_n_synthetic": selected["config"].n_synthetic,
        "selected_perturbation_mode": selected["config"].perturbation_mode,
        "selected_noise_scale": selected["config"].noise_scale,
        "selected_inner_cv_rmsep": selected["inner_cv_rmsep"],
        "selected_inner_cv_r2": selected["inner_cv_r2"],
        "baseline_r2": final["baseline_r2"],
        "rensa_r2": final["rensa_r2"],
        "delta_r2": final["rensa_r2"] - final["baseline_r2"],
        "baseline_rmsep": final["baseline_rmsep"],
        "rensa_rmsep": final["rensa_rmsep"],
        "delta_rmsep": final["rensa_rmsep"] - final["baseline_rmsep"],
        "baseline_best_params": json.dumps(final["baseline_best_params"], sort_keys=True),
        "rensa_best_params": json.dumps(final["rensa_best_params"], sort_keys=True),
        "cars_selected_features": final["cars_selected_features"],
        "cars_feature_hash": final["cars_feature_hash"],
        "paired_feature_set": 1,
        "synthetic_requested": selected["config"].n_synthetic,
        "synthetic_accepted": final["synthetic_accepted"],
        "acceptance_rate": final["acceptance_rate"],
        **sparsity,
        "calibration_indices": json.dumps(train_idx.tolist()),
        "holdout_indices": json.dumps(test_idx.tolist()),
        "elapsed_sec": elapsed,
    }

    y_rows = y_statistics_rows(spec, split_label, outer_seed if method == "mc" else "", y_cal, y_hold)
    sparsity_row = {
        "task": spec.key,
        "task_label": spec.label,
        "split_method": split_label,
        "outer_seed": outer_seed if method == "mc" else "",
        "n_calibration": len(train_idx),
        **sparsity,
    }
    prediction_rows = []
    for idx, truth, base_pred, rensa_pred in zip(
        test_idx, y_hold, final["baseline_pred"], final["rensa_pred"]
    ):
        prediction_rows.append({
            "task": spec.key,
            "task_label": spec.label,
            "split_method": split_label,
            "outer_seed": outer_seed if method == "mc" else "",
            "sample_index": int(idx),
            "y_true": float(truth),
            "baseline_prediction": float(base_pred),
            "rensa_prediction": float(rensa_pred),
            "baseline_residual": float(truth - base_pred),
            "rensa_residual": float(truth - rensa_pred),
        })

    prefix = "monte_carlo" if method == "mc" else "ks"
    append_rows(args.output_dir / f"{prefix}_split_results.csv" if method == "mc" else args.output_dir / "ks_results.csv", [result_row], RESULT_FIELDS)
    append_rows(args.output_dir / f"{prefix}_y_statistics.csv", y_rows, Y_STAT_FIELDS)
    append_rows(args.output_dir / f"{prefix}_sparsity.csv", [sparsity_row], SPARSITY_FIELDS)
    append_rows(args.output_dir / f"{prefix}_predictions.csv", prediction_rows, PREDICTION_FIELDS)
    append_rows(args.output_dir / "candidate_selection_scores.csv", selection_rows, SELECTION_FIELDS)
    LOGGER.info(
        "done task=%s method=%s seed=%s delta_r2=%+.6f delta_rmsep=%+.6f elapsed=%.1fs",
        spec.key,
        split_label,
        outer_seed if method == "mc" else "deterministic",
        result_row["delta_r2"],
        result_row["delta_rmsep"],
        elapsed,
    )


def select_rensa_candidate(
    X: np.ndarray,
    y: np.ndarray,
    configs: list[AugConfig],
    param_grid: dict,
    args: argparse.Namespace,
    task: str,
    split_method: str,
    outer_seed: int,
) -> tuple[dict, list[dict]]:
    splitter = KFold(n_splits=min(args.inner_cv, X.shape[0]), shuffle=True, random_state=args.algorithm_seed)
    folds = list(splitter.split(X))
    candidates = [(preprocess, config) for preprocess in args.preprocess_candidates for config in configs]
    predictions = {candidate: np.empty_like(y, dtype=float) for candidate in candidates}

    for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
        for preprocess in args.preprocess_candidates:
            preprocessor = fit_preprocessor(X[fit_idx], preprocess)
            X_fit_pre = preprocessor.transform(X[fit_idx])
            X_valid_pre = preprocessor.transform(X[valid_idx])
            cars = make_cars(args)
            Z_fit_measured = cars.fit_transform(X_fit_pre, y[fit_idx])
            Z_valid = cars.transform(X_valid_pre)
            for config in configs:
                Z_fit, y_fit, _, _ = apply_rensa(Z_fit_measured, y[fit_idx], config, args)
                search = make_svr_search(param_grid, args, len(y_fit))
                search.fit(Z_fit, y_fit)
                predictions[(preprocess, config)][valid_idx] = search.predict(Z_valid).reshape(-1)
        LOGGER.info("inner candidate fold %d/%d complete", fold_number, len(folds))

    scored = []
    for preprocess, config in candidates:
        pred = predictions[(preprocess, config)]
        scored.append({
            "preprocess": preprocess,
            "config": config,
            "inner_cv_rmsep": rmse(y, pred),
            "inner_cv_r2": float(r2_score(y, pred)),
        })
    scored.sort(key=lambda item: (item["inner_cv_rmsep"], item["preprocess"], item["config"].key))
    rows = []
    for rank, item in enumerate(scored, start=1):
        config = item["config"]
        rows.append({
            "task": task,
            "split_method": split_method,
            "outer_seed": outer_seed if split_method == "stratified_monte_carlo" else "",
            "rank": rank,
            "selected": int(rank == 1),
            "preprocess": item["preprocess"],
            "n_synthetic": config.n_synthetic,
            "perturbation_mode": config.perturbation_mode,
            "noise_scale": config.noise_scale,
            "inner_cv_rmsep": item["inner_cv_rmsep"],
            "inner_cv_r2": item["inner_cv_r2"],
        })
    return scored[0], rows


def paired_final_fit(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    selected: dict,
    param_grid: dict,
    args: argparse.Namespace,
) -> dict:
    preprocessor = fit_preprocessor(X_cal, selected["preprocess"])
    X_cal_pre = preprocessor.transform(X_cal)
    X_hold_pre = preprocessor.transform(X_hold)
    cars = make_cars(args)
    Z_cal = cars.fit_transform(X_cal_pre, y_cal)
    Z_hold = cars.transform(X_hold_pre)
    selected_indices = np.asarray(cars.selected_indices_, dtype=np.int64)
    feature_hash = hashlib.sha256(selected_indices.tobytes()).hexdigest()[:16]

    baseline_search = make_svr_search(param_grid, args, len(y_cal))
    baseline_search.fit(Z_cal, y_cal)
    baseline_pred = baseline_search.predict(Z_hold).reshape(-1)

    Z_rensa, y_rensa, accepted, acceptance_rate = apply_rensa(
        Z_cal, y_cal, selected["config"], args
    )
    rensa_search = make_svr_search(param_grid, args, len(y_rensa))
    rensa_search.fit(Z_rensa, y_rensa)
    rensa_pred = rensa_search.predict(Z_hold).reshape(-1)

    return {
        "baseline_pred": baseline_pred,
        "rensa_pred": rensa_pred,
        "baseline_rmsep": rmse(y_hold, baseline_pred),
        "rensa_rmsep": rmse(y_hold, rensa_pred),
        "baseline_r2": float(r2_score(y_hold, baseline_pred)),
        "rensa_r2": float(r2_score(y_hold, rensa_pred)),
        "baseline_best_params": baseline_search.best_params_,
        "rensa_best_params": rensa_search.best_params_,
        "cars_selected_features": int(selected_indices.size),
        "cars_feature_hash": feature_hash,
        "synthetic_accepted": int(accepted),
        "acceptance_rate": float(acceptance_rate),
    }


def apply_rensa(
    X: np.ndarray, y: np.ndarray, config: AugConfig, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, int, float]:
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=config.n_synthetic,
        response_bins=args.response_bins,
        response_bin_strategy=args.response_bin_strategy,
        neighbors=args.neighbors,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scale=config.noise_scale,
        random_state=args.algorithm_seed,
        neighbor_space=args.neighbor_space,
        perturbation_mode=config.perturbation_mode,
    )
    result = augmenter.fit_resample(X, y)
    return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])


def make_cars(args: argparse.Namespace) -> CARSFeatureSelector:
    return CARSFeatureSelector(
        n_sampling=args.cars_sampling,
        min_features=args.cars_min_features,
        max_components=args.cars_components,
        cv=min(args.inner_cv, 5),
        random_state=args.algorithm_seed,
    )


def make_svr_search(param_grid: dict, args: argparse.Namespace, n_samples: int) -> GridSearchCV:
    splitter = KFold(
        n_splits=min(args.inner_cv, n_samples),
        shuffle=True,
        random_state=args.algorithm_seed,
    )
    return GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        n_jobs=16,
        refit=True,
    )


def augmentation_configs(args: argparse.Namespace) -> list[AugConfig]:
    configs = []
    seen = set()
    for n_synthetic, perturbation_mode, noise_scale in product(
        args.n_synthetic, args.perturbation_mode, args.noise_scale
    ):
        if perturbation_mode == "none" or noise_scale <= 0:
            perturbation_mode = "none"
            noise_scale = 0.0
        config = AugConfig(int(n_synthetic), str(perturbation_mode), float(noise_scale))
        if config not in seen:
            seen.add(config)
            configs.append(config)
    if not configs:
        raise ValueError("At least one augmented RenSA configuration is required.")
    return configs


def stratified_regression_split(
    y: np.ndarray, test_size: float, random_state: int, requested_bins: int = 5
) -> tuple[np.ndarray, np.ndarray, int]:
    y = np.asarray(y, dtype=float).reshape(-1)
    indices = np.arange(y.size)
    n_test = int(np.ceil(y.size * test_size))
    n_train = y.size - n_test
    for bins in range(min(int(requested_bins), y.size), 1, -1):
        labels = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"), dtype=int)
        unique, counts = np.unique(labels, return_counts=True)
        used_bins = int(unique.size)
        if used_bins < 2 or counts.min() < 2 or n_test < used_bins or n_train < used_bins:
            continue
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=labels,
        )
        if used_bins != requested_bins:
            LOGGER.warning(
                "stratification bins reduced requested=%d used=%d seed=%d",
                requested_bins,
                used_bins,
                random_state,
            )
        return np.sort(train_idx), np.sort(test_idx), used_bins
    raise ValueError("Unable to create a valid stratified regression split with at least two bins.")


def response_sparsity(y: np.ndarray, neighbors: int = 5) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1)
    scale = float(np.std(y, ddof=0))
    if scale <= 0:
        raise ValueError("Response sparsity is undefined for a constant calibration response.")
    z = ((y - np.mean(y)) / scale).reshape(-1, 1)
    n_neighbors = min(int(neighbors) + 1, z.shape[0])
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(z).kneighbors(z)
    fifth_distance = distances[:, -1]
    return {
        "sparsity_median_5nn": float(np.median(fifth_distance)),
        "sparsity_p90_5nn": float(np.quantile(fifth_distance, 0.90)),
        "sparsity_max_5nn": float(np.max(fifth_distance)),
    }


def y_statistics_rows(
    spec: TaskSpec,
    split_method: str,
    outer_seed: int | str,
    y_cal: np.ndarray,
    y_hold: np.ndarray,
) -> list[dict]:
    rows = []
    for subset, values in (("calibration", y_cal), ("holdout", y_hold)):
        values = np.asarray(values, dtype=float)
        rows.append({
            "task": spec.key,
            "task_label": spec.label,
            "split_method": split_method,
            "outer_seed": outer_seed,
            "subset": subset,
            "n": values.size,
            "min": float(np.min(values)),
            "q1": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q3": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
            "sd": float(np.std(values, ddof=1)),
        })
    return rows


def write_spxy_statistics(args: argparse.Namespace) -> None:
    rows = []
    for task_key in args.tasks:
        spec = TASKS[task_key]
        _, X, y = load_spectrum_csv(
            args.data_dir / spec.data_file,
            target_column=spec.target_column,
            spectral_prefix=spec.spectral_prefix,
        )
        for test_size in (0.20, 0.25, 0.30):
            train_idx, test_idx = representative_split(
                X, y, test_size=test_size, method="spxy", spxy_y_weight=1.0
            )
            split_label = f"spxy_{int(round(test_size * 100))}pct"
            rows.extend(y_statistics_rows(spec, split_label, "", y[train_idx], y[test_idx]))
    write_rows(args.output_dir / "spxy_y_statistics.csv", rows, Y_STAT_FIELDS)
    LOGGER.info("wrote SPXY response statistics for %d tasks", len(args.tasks))


def refresh_derived_outputs(output_dir: Path, n_bootstrap: int, random_state: int) -> None:
    mc_path = output_dir / "monte_carlo_split_results.csv"
    if not mc_path.exists() or mc_path.stat().st_size == 0:
        return
    frame = pd.read_csv(mc_path)
    summary_rows = []
    correlation_rows = []
    rng = np.random.default_rng(random_state)
    for task, group in frame.groupby("task", sort=False):
        for metric, improved in (("delta_r2", "gt0"), ("delta_rmsep", "lt0")):
            values = group[metric].to_numpy(dtype=float)
            boot_means = np.mean(
                rng.choice(values, size=(int(n_bootstrap), values.size), replace=True), axis=1
            ) if int(n_bootstrap) > 0 else np.asarray([], dtype=float)
            summary_rows.append({
                "task": task,
                "metric": metric,
                "n_splits": values.size,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                "empirical_p2_5": float(np.quantile(values, 0.025)),
                "empirical_p97_5": float(np.quantile(values, 0.975)),
                "fraction_improved": float(np.mean(values > 0)) if improved == "gt0" else float(np.mean(values < 0)),
                "bootstrap_mean_ci_low": float(np.quantile(boot_means, 0.025)) if boot_means.size else np.nan,
                "bootstrap_mean_ci_high": float(np.quantile(boot_means, 0.975)) if boot_means.size else np.nan,
                "bootstrap_seed": random_state,
                "bootstrap_replicates": int(n_bootstrap),
            })
        for sparsity_metric in ("sparsity_median_5nn", "sparsity_p90_5nn"):
            for gain_metric in ("delta_r2", "delta_rmsep"):
                x = group[sparsity_metric].to_numpy(dtype=float)
                gain = group[gain_metric].to_numpy(dtype=float)
                if x.size >= 3 and np.unique(x).size > 1 and np.unique(gain).size > 1:
                    rho, p_value = spearmanr(x, gain)
                else:
                    rho, p_value = np.nan, np.nan
                correlation_rows.append({
                    "task": task,
                    "sparsity_metric": sparsity_metric,
                    "gain_metric": gain_metric,
                    "rho": float(rho),
                    "p_value": float(p_value),
                    "n_splits": int(x.size),
                })
    write_rows(output_dir / "monte_carlo_summary.csv", summary_rows, list(summary_rows[0]))
    write_rows(
        output_dir / "sparsity_gain_correlations.csv",
        correlation_rows,
        list(correlation_rows[0]),
    )


def write_manifest(args: argparse.Namespace, configs: list[AugConfig], param_grid: dict) -> None:
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "tasks": args.tasks,
        "split_method": args.split_method,
        "outer_seeds": args.outer_seeds,
        "test_size": args.test_size,
        "requested_strata": args.strata,
        "algorithm_seed": args.algorithm_seed,
        "inner_cv": args.inner_cv,
        "preprocess_candidates": args.preprocess_candidates,
        "augmentation_configs": [asdict(config) for config in configs],
        "fixed_consistency_thresholds": {
            "max_spectral_angle": 0.18,
            "min_derivative_corr": 0.70,
            "envelope_margin": 0.08,
        },
        "cars": {
            "n_sampling": args.cars_sampling,
            "min_features": args.cars_min_features,
            "max_components": args.cars_components,
            "random_state": args.algorithm_seed,
        },
        "svr_grid": param_grid,
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "bootstrap_seed": args.algorithm_seed,
        "bootstrap_replicates": args.bootstrap,
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def append_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_completed(output_dir: Path, method: str, task: str, outer_seed: int) -> bool:
    path = output_dir / ("monte_carlo_split_results.csv" if method == "mc" else "ks_results.csv")
    if not path.exists() or path.stat().st_size == 0:
        return False
    frame = pd.read_csv(path, dtype={"outer_seed": str})
    expected_seed = str(outer_seed) if method == "mc" else "nan"
    return bool(((frame["task"] == task) & (frame["outer_seed"].astype(str) == expected_seed)).any())


def method_name(method: str) -> str:
    return "stratified_monte_carlo" if method == "mc" else "kennard_stone"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_gamma(value: str | float) -> str | float:
    return value if str(value) == "scale" else float(value)


if __name__ == "__main__":
    main()
