"""R3.3a: standard open-source SMOGN/SMOTER under the RenSA nested protocol.

Each baseline receives the same model-selection budget as RenSA: six
augmentation configurations crossed with six preprocessing methods (36
pipeline candidates per outer training set). Candidate selection uses the same
five measured-only outer-inner folds and the same 36-point RBF-SVR grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
RESULTS_DIR = EXPERIMENT_ROOT / "results_standard_smogn_smoter"
LOGS_DIR = EXPERIMENT_ROOT / "logs_standard_smogn_smoter"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_renvsa as base  # noqa: E402
import smogn_smoter_numeric as numeric_standard  # noqa: E402

try:  # noqa: E402
    import ImbalancedLearningRegression.over_sampling_smote as iblr_over_sampling
    import smogn.over_sampling as smogn_over_sampling
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "R3.3a requires smogn==0.1.2 and ImbalancedLearningRegression==0.0.2."
    ) from exc


ALGORITHM_SEED = base.ALGORITHM_SEED
INNER_CV = base.INNER_CV
PREPROCESSING = base.PREPROCESSING
METHODS = ("smoter", "smogn")
PACKAGE_VERSIONS = {
    "smogn": "0.1.2",
    "ImbalancedLearningRegression": "0.0.2",
    "audit_kernel": numeric_standard.IMPLEMENTATION_VERSION,
}
RELEVANCE_CONTROL_QUANTILES = (0.0, 0.10, 0.50, 0.90, 1.0)
SVR_GRID_SIZE = len(base.SVR_C) * len(base.SVR_GAMMA) * len(base.SVR_EPSILON)
SVR_JOBS = 16


@dataclass(frozen=True)
class BaselineConfig:
    config_id: str
    relevance_threshold: float
    k: int
    sampling_strategy: str
    perturbation: float


# Balanced space-filling grid: every tuned factor is represented while the
# total candidate count matches RenSA's six effective configurations.
SEARCH_CONFIGS = (
    BaselineConfig("B01", 0.40, 3, "balance", 0.01),
    BaselineConfig("B02", 0.40, 5, "extreme", 0.02),
    BaselineConfig("B03", 0.50, 3, "extreme", 0.05),
    BaselineConfig("B04", 0.50, 5, "balance", 0.01),
    BaselineConfig("B05", 0.60, 3, "balance", 0.02),
    BaselineConfig("B06", 0.60, 5, "extreme", 0.05),
)
PIPELINE_CANDIDATES_PER_METHOD = len(PREPROCESSING) * len(SEARCH_CONFIGS)


@dataclass(frozen=True)
class Candidate:
    method: str
    preprocessing: str
    config: BaselineConfig


@dataclass(frozen=True)
class ResampledData:
    X: np.ndarray
    y: np.ndarray
    n_input: int
    n_output: int
    n_synthetic_nonmatching: int
    n_exact_rows_output: int
    n_unique_original_retained: int
    n_interpolated: int
    n_gaussian: int
    n_replicated: int


@dataclass(frozen=True)
class FoldArtifact:
    valid_indices: np.ndarray
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    n_input: int
    n_output: int
    n_synthetic_nonmatching: int
    n_exact_rows_output: int
    n_unique_original_retained: int
    n_interpolated: int
    n_gaussian: int
    n_replicated: int


SPLIT_RESULT_FIELDS = [
    "method", "implementation", "package_version", "task", "split_method",
    "outer_seed", "actual_strata", "algorithm_seed", "n_calibration",
    "n_holdout", "n_full_wavelengths", "n_cars_features",
    "selected_cars_indices", "preprocessing", "selected_config_id",
    "relevance_method", "relevance_control_quantiles", "relevance_threshold",
    "k", "sampling_strategy", "perturbation", "n_train_input",
    "n_train_resampled", "net_sample_change", "n_synthetic_nonmatching",
    "n_exact_rows_output", "n_unique_original_retained", "n_interpolated",
    "n_gaussian", "n_replicated", "best_C",
    "best_gamma", "best_epsilon", "pipeline_candidates_evaluated",
    "svr_grid_combinations", "inner_cv_RMSEP", "inner_cv_R2", "R2",
    "RMSEP", "runtime_seconds",
]

PREDICTION_FIELDS = [
    "method", "task", "split_method", "outer_seed", "sample_index",
    "y_true", "y_pred", "residual",
]

CANDIDATE_SCORE_FIELDS = [
    "method", "task", "split_method", "outer_seed", "preprocessing",
    "config_id", "relevance_threshold", "k", "sampling_strategy",
    "perturbation", "inner_cv_RMSEP", "inner_cv_R2", "mean_n_resampled",
    "min_n_resampled", "max_n_resampled", "mean_n_synthetic_nonmatching",
    "mean_n_unique_original_retained", "mean_n_interpolated",
    "mean_n_gaussian", "mean_n_replicated", "is_selected",
]

SUMMARY_FIELDS = [
    "method", "task", "spxy_R2", "spxy_RMSEP", "ks_R2", "ks_RMSEP",
    "mc_n", "mc_mean_R2", "mc_SD_R2", "mc_median_R2", "mc_mean_RMSEP",
    "mc_SD_RMSEP", "mc_median_RMSEP", "mc_min_RMSEP", "mc_max_RMSEP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standard SMOGN and SMOTER with a RenSA-matched nested-CV budget."
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tasks", nargs="+", choices=tuple(base.DATASETS), default=list(base.DATASETS))
    parser.add_argument(
        "--split-methods", nargs="+", choices=("spxy", "ks", "mc"),
        default=["spxy", "ks", "mc"],
    )
    parser.add_argument("--mc-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--svr-jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    global SVR_JOBS
    args = parse_args()
    SVR_JOBS = max(1, int(args.svr_jobs))
    verify_environment()
    if args.self_test:
        run_self_test()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()
    write_manifest()
    if args.aggregate_only:
        rebuild_aggregate_outputs()
        logging.info("aggregate-only completed checkpoints=%d", len(load_checkpoints()))
        return

    invalid_mc_seeds = sorted(set(args.mc_seeds) - {0, 1, 2, 3, 4})
    if invalid_mc_seeds:
        raise ValueError(f"MC outer seeds must be selected from 0..4, got {invalid_mc_seeds}.")
    completed = completed_keys()
    new_runs = 0
    for task, split_method, outer_seed in requested_runs(args):
        missing_methods = [
            method for method in args.methods
            if experiment_key(method, task, split_method, outer_seed) not in completed
        ]
        if not missing_methods:
            if args.resume:
                logging.info("skip completed task=%s split=%s seed=%s", task, split_method, base.seed_text(outer_seed))
                continue
            raise FileExistsError(
                f"Requested results already exist for {(task, split_method, outer_seed)}; use --resume."
            )
        if not args.resume and len(missing_methods) != len(args.methods):
            raise FileExistsError(
                f"A partial result exists for {(task, split_method, outer_seed)}; use --resume."
            )
        run_one(task, split_method, outer_seed, tuple(missing_methods))
        new_runs += 1
        completed.update(
            experiment_key(method, task, split_method, outer_seed) for method in missing_methods
        )
        if args.max_runs and new_runs >= args.max_runs:
            break
    if not args.no_aggregate:
        rebuild_aggregate_outputs()
    logging.info("finished new_outer_runs=%d completed_method_checkpoints=%d", new_runs, len(completed_keys()))


def requested_runs(args: argparse.Namespace) -> Iterable[tuple[str, str, int | None]]:
    for task in args.tasks:
        for split_method in args.split_methods:
            if split_method == "mc":
                for seed in args.mc_seeds:
                    yield task, split_method, int(seed)
            else:
                yield task, split_method, None


def run_one(task: str, split_method: str, outer_seed: int | None, methods: tuple[str, ...]) -> None:
    started = time.perf_counter()
    _, X, y = base.load_task(task)
    n_full_wavelengths = int(X.shape[1])
    calibration_idx, holdout_idx, actual_strata = base.outer_split(X, y, split_method, outer_seed)
    base.assert_disjoint_complete_indices(calibration_idx, holdout_idx, X.shape[0])
    X_calibration = X[calibration_idx].copy()
    y_calibration = y[calibration_idx].copy()
    logging.info(
        "start task=%s split=%s seed=%s methods=%s calibration=%d holdout=%d full_dimension=%d",
        task, split_method, base.seed_text(outer_seed), ",".join(methods),
        calibration_idx.size, holdout_idx.size, n_full_wavelengths,
    )
    folds = list(
        KFold(
            n_splits=min(INNER_CV, X_calibration.shape[0]), shuffle=True,
            random_state=ALGORITHM_SEED,
        ).split(X_calibration)
    )
    base.validate_inner_folds(folds, X_calibration.shape[0])
    cache = build_inner_fold_cache(
        X_calibration, y_calibration, folds, n_full_wavelengths, methods
    )
    selected = {
        method: select_method_by_inner_cv(
            method, task, split_method, outer_seed, y_calibration, folds, cache
        )
        for method in methods
    }

    for method in methods:
        candidate, inner_metrics, candidate_scores = selected[method]
        holdout = base.HoldoutGuard(X, y, holdout_idx)
        final = fit_final_and_predict(
            X_calibration, y_calibration, holdout, candidate, n_full_wavelengths
        )
        assert holdout.revealed
        _, y_holdout = final.pop("holdout_values")
        prediction = np.asarray(final.pop("prediction"), dtype=float)
        elapsed = time.perf_counter() - started
        package_name, implementation = implementation_for(method)
        resampled = final["resampled"]
        row = {
            "method": method,
            "implementation": implementation,
            "package_version": PACKAGE_VERSIONS[package_name],
            "task": task,
            "split_method": split_method,
            "outer_seed": base.seed_value(outer_seed),
            "actual_strata": actual_strata if actual_strata is not None else "",
            "algorithm_seed": ALGORITHM_SEED,
            "n_calibration": int(calibration_idx.size),
            "n_holdout": int(holdout_idx.size),
            "n_full_wavelengths": n_full_wavelengths,
            "n_cars_features": final["n_cars_features"],
            "selected_cars_indices": json.dumps(final["selected_cars_indices"], separators=(",", ":")),
            "preprocessing": candidate.preprocessing,
            "selected_config_id": candidate.config.config_id,
            "relevance_method": "manual_train_fold_quantiles",
            "relevance_control_quantiles": json.dumps(RELEVANCE_CONTROL_QUANTILES),
            "relevance_threshold": candidate.config.relevance_threshold,
            "k": candidate.config.k,
            "sampling_strategy": candidate.config.sampling_strategy,
            "perturbation": candidate.config.perturbation if method == "smogn" else "",
            "n_train_input": resampled.n_input,
            "n_train_resampled": resampled.n_output,
            "net_sample_change": resampled.n_output - resampled.n_input,
            "n_synthetic_nonmatching": resampled.n_synthetic_nonmatching,
            "n_exact_rows_output": resampled.n_exact_rows_output,
            "n_unique_original_retained": resampled.n_unique_original_retained,
            "n_interpolated": resampled.n_interpolated,
            "n_gaussian": resampled.n_gaussian,
            "n_replicated": resampled.n_replicated,
            "best_C": final["best_C"],
            "best_gamma": final["best_gamma"],
            "best_epsilon": final["best_epsilon"],
            "pipeline_candidates_evaluated": PIPELINE_CANDIDATES_PER_METHOD,
            "svr_grid_combinations": SVR_GRID_SIZE,
            "inner_cv_RMSEP": inner_metrics["RMSEP"],
            "inner_cv_R2": inner_metrics["R2"],
            "R2": float(r2_score(y_holdout, prediction)),
            "RMSEP": float(np.sqrt(mean_squared_error(y_holdout, prediction))),
            "runtime_seconds": float(elapsed),
        }
        predictions = [
            {
                "method": method,
                "task": task,
                "split_method": split_method,
                "outer_seed": base.seed_value(outer_seed),
                "sample_index": int(sample_idx),
                "y_true": float(y_true),
                "y_pred": float(y_pred),
                "residual": float(y_true - y_pred),
            }
            for sample_idx, y_true, y_pred in zip(holdout_idx, y_holdout, prediction)
        ]
        save_checkpoint(
            method, task, split_method, outer_seed, row, predictions, candidate_scores
        )
        logging.info(
            "done method=%s task=%s split=%s seed=%s config=%s/%s n=%d->%d R2=%.6f RMSEP=%.6f elapsed=%.2fs",
            method, task, split_method, base.seed_text(outer_seed),
            candidate.preprocessing, candidate.config.config_id,
            row["n_train_input"], row["n_train_resampled"], row["R2"],
            row["RMSEP"], elapsed,
        )


def build_inner_fold_cache(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_full_wavelengths: int,
    methods: tuple[str, ...],
) -> dict[tuple[str, int, str, str], FoldArtifact]:
    cache: dict[tuple[str, int, str, str], FoldArtifact] = {}
    for fold_idx, (fit_idx, valid_idx) in enumerate(folds):
        X_fit_raw, y_fit = X[fit_idx], y[fit_idx]
        X_valid_raw, y_valid = X[valid_idx], y[valid_idx]
        for preprocessing in PREPROCESSING:
            preprocessor = base.fit_preprocessor_measured_only(
                X_fit_raw, preprocessing, expected_measured_count=fit_idx.size
            )
            X_fit_full = preprocessor.transform(X_fit_raw)
            X_valid_full = preprocessor.transform(X_valid_raw)
            assert X_fit_full.shape[1] == n_full_wavelengths
            selector = base.fit_cars_measured_only(
                X_fit_full, y_fit, expected_measured_count=fit_idx.size,
                forbidden_synthetic_count=1,
            )
            selected_indices = np.asarray(selector.selected_indices_, dtype=int)
            Z_valid = X_valid_full[:, selected_indices]
            assert Z_valid.shape[0] == valid_idx.size
            for method in methods:
                for config in SEARCH_CONFIGS:
                    resampled = resample_standard(method, X_fit_full, y_fit, config)
                    Z_fit = resampled.X[:, selected_indices]
                    assert Z_fit.shape[0] == resampled.y.size
                    cache[(method, fold_idx, preprocessing, config.config_id)] = FoldArtifact(
                        valid_indices=valid_idx.copy(), X_fit=Z_fit,
                        y_fit=resampled.y, X_valid=Z_valid, y_valid=y_valid.copy(),
                        n_input=resampled.n_input, n_output=resampled.n_output,
                        n_synthetic_nonmatching=resampled.n_synthetic_nonmatching,
                        n_exact_rows_output=resampled.n_exact_rows_output,
                        n_unique_original_retained=resampled.n_unique_original_retained,
                        n_interpolated=resampled.n_interpolated,
                        n_gaussian=resampled.n_gaussian,
                        n_replicated=resampled.n_replicated,
                    )
        logging.info("built standard-baseline cache for inner fold %d/%d", fold_idx + 1, len(folds))
    return cache


def select_method_by_inner_cv(
    method: str,
    task: str,
    split_method: str,
    outer_seed: int | None,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cache: dict[tuple[str, int, str, str], FoldArtifact],
) -> tuple[Candidate, dict[str, float], list[dict[str, Any]]]:
    scored: list[tuple[Candidate, dict[str, float], dict[str, Any]]] = []
    for preprocessing in PREPROCESSING:
        for config in SEARCH_CONFIGS:
            artifacts = [
                cache[(method, fold_idx, preprocessing, config.config_id)]
                for fold_idx in range(len(folds))
            ]
            prediction = np.empty_like(y, dtype=float)
            for artifact in artifacts:
                search = make_svr_search(artifact.X_fit.shape[0])
                search.fit(artifact.X_fit, artifact.y_fit)
                prediction[artifact.valid_indices] = search.predict(artifact.X_valid).reshape(-1)
            metrics = {
                "RMSEP": float(np.sqrt(mean_squared_error(y, prediction))),
                "R2": float(r2_score(y, prediction)),
            }
            n_output = np.asarray([artifact.n_output for artifact in artifacts], dtype=float)
            n_synthetic = np.asarray(
                [artifact.n_synthetic_nonmatching for artifact in artifacts], dtype=float
            )
            n_retained = np.asarray(
                [artifact.n_unique_original_retained for artifact in artifacts], dtype=float
            )
            n_interpolated = np.asarray(
                [artifact.n_interpolated for artifact in artifacts], dtype=float
            )
            n_gaussian = np.asarray(
                [artifact.n_gaussian for artifact in artifacts], dtype=float
            )
            n_replicated = np.asarray(
                [artifact.n_replicated for artifact in artifacts], dtype=float
            )
            score_row = {
                "method": method, "task": task, "split_method": split_method,
                "outer_seed": base.seed_value(outer_seed),
                "preprocessing": preprocessing, "config_id": config.config_id,
                "relevance_threshold": config.relevance_threshold, "k": config.k,
                "sampling_strategy": config.sampling_strategy,
                "perturbation": config.perturbation if method == "smogn" else "",
                "inner_cv_RMSEP": metrics["RMSEP"], "inner_cv_R2": metrics["R2"],
                "mean_n_resampled": float(np.mean(n_output)),
                "min_n_resampled": int(np.min(n_output)),
                "max_n_resampled": int(np.max(n_output)),
                "mean_n_synthetic_nonmatching": float(np.mean(n_synthetic)),
                "mean_n_unique_original_retained": float(np.mean(n_retained)),
                "mean_n_interpolated": float(np.mean(n_interpolated)),
                "mean_n_gaussian": float(np.mean(n_gaussian)),
                "mean_n_replicated": float(np.mean(n_replicated)),
                "is_selected": 0,
            }
            scored.append((Candidate(method, preprocessing, config), metrics, score_row))
            logging.info(
                "inner scored method=%s preprocessing=%s config=%s threshold=%.2f k=%d sampling=%s pert=%.3f",
                method, preprocessing, config.config_id, config.relevance_threshold,
                config.k, config.sampling_strategy, config.perturbation,
            )
    assert len(scored) == PIPELINE_CANDIDATES_PER_METHOD
    scored.sort(key=lambda item: (item[1]["RMSEP"], item[0].preprocessing, item[0].config.config_id))
    best_candidate, best_metrics, _ = scored[0]
    rows = []
    for candidate, _, row in scored:
        row["is_selected"] = int(candidate == best_candidate)
        rows.append(row)
    return best_candidate, best_metrics, rows


def fit_final_and_predict(
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    holdout: base.HoldoutGuard,
    candidate: Candidate,
    n_full_wavelengths: int,
) -> dict[str, Any]:
    preprocessor = base.fit_preprocessor_measured_only(
        X_calibration, candidate.preprocessing,
        expected_measured_count=X_calibration.shape[0],
    )
    X_measured_full = preprocessor.transform(X_calibration)
    assert X_measured_full.shape[1] == n_full_wavelengths
    resampled = resample_standard(
        candidate.method, X_measured_full, y_calibration, candidate.config
    )
    selector = base.fit_cars_measured_only(
        X_measured_full, y_calibration,
        expected_measured_count=X_calibration.shape[0],
        forbidden_synthetic_count=max(1, abs(resampled.n_output - resampled.n_input)),
    )
    selected_indices = np.asarray(selector.selected_indices_, dtype=int)
    X_holdout_raw, y_holdout = holdout.reveal("final_prediction")
    X_holdout_full = preprocessor.transform(X_holdout_raw)
    Z_fit = resampled.X[:, selected_indices]
    Z_holdout = X_holdout_full[:, selected_indices]
    search = make_svr_search(Z_fit.shape[0])
    search.fit(Z_fit, resampled.y)
    prediction = search.predict(Z_holdout).reshape(-1)
    params = search.best_params_
    return {
        "prediction": prediction,
        "holdout_values": (X_holdout_raw, y_holdout),
        "n_cars_features": int(selected_indices.size),
        "selected_cars_indices": selected_indices.tolist(),
        "resampled": resampled,
        "best_C": float(params["svr__C"]),
        "best_gamma": float(params["svr__gamma"]),
        "best_epsilon": float(params["svr__epsilon"]),
    }


def resample_standard(
    method: str, X: np.ndarray, y: np.ndarray, config: BaselineConfig
) -> ResampledData:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y have incompatible shapes.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Standard resamplers require finite training data.")
    output = numeric_standard.resample_numeric(
        method,
        X,
        y,
        relevance_control_points=relevance_control_points(y),
        relevance_threshold=config.relevance_threshold,
        k=config.k,
        sampling_strategy=config.sampling_strategy,
        perturbation=config.perturbation,
        random_state=ALGORITHM_SEED,
    )
    X_output = output.X
    y_output = output.y
    if X_output.shape[1] != X.shape[1] or X_output.shape[0] != y_output.size:
        raise RuntimeError(f"{method} returned an invalid resampled shape.")
    if not np.isfinite(X_output).all() or not np.isfinite(y_output).all():
        raise RuntimeError(f"{method} returned non-finite values.")
    synthetic, exact, retained = audit_resampled_rows(X, y, X_output, y_output)
    return ResampledData(
        X=X_output, y=y_output, n_input=int(y.size), n_output=int(y_output.size),
        n_synthetic_nonmatching=synthetic, n_exact_rows_output=exact,
        n_unique_original_retained=retained,
        n_interpolated=output.n_interpolated,
        n_gaussian=output.n_gaussian,
        n_replicated=output.n_replicated,
    )


def relevance_control_points(y: np.ndarray) -> list[list[float]]:
    values = np.quantile(np.asarray(y, dtype=float), RELEVANCE_CONTROL_QUANTILES)
    if not np.all(np.diff(values) > 0):
        raise ValueError("Training-fold response quantiles are not strictly increasing.")
    relevance = (1.0, 1.0, 0.0, 1.0, 1.0)
    return [
        [float(value), float(phi_value), 0.0]
        for value, phi_value in zip(values, relevance)
    ]


def audit_resampled_rows(
    X_input: np.ndarray,
    y_input: np.ndarray,
    X_output: np.ndarray,
    y_output: np.ndarray,
) -> tuple[int, int, int]:
    input_rows = np.column_stack([X_input, y_input]).astype(np.float64, copy=False)
    output_rows = np.column_stack([X_output, y_output]).astype(np.float64, copy=False)

    def row_key(row: np.ndarray) -> bytes:
        return np.ascontiguousarray(row).tobytes()

    input_keys = {row_key(row) for row in input_rows}
    output_keys = [row_key(row) for row in output_rows]
    exact_mask = np.asarray([key in input_keys for key in output_keys], dtype=bool)
    unique_retained = len(input_keys.intersection(output_keys))
    return int((~exact_mask).sum()), int(exact_mask.sum()), int(unique_retained)


def make_svr_search(n_samples: int) -> GridSearchCV:
    splitter = KFold(
        n_splits=min(INNER_CV, int(n_samples)), shuffle=True,
        random_state=ALGORITHM_SEED,
    )
    return GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid={
            "svr__C": list(base.SVR_C), "svr__gamma": list(base.SVR_GAMMA),
            "svr__epsilon": list(base.SVR_EPSILON),
        },
        scoring="neg_root_mean_squared_error", cv=splitter, n_jobs=SVR_JOBS,
        refit=True,
    )


def implementation_for(method: str) -> tuple[str, str]:
    if method == "smoter":
        return "audit_kernel", "smogn_smoter_numeric.smoter"
    if method == "smogn":
        return "audit_kernel", "smogn_smoter_numeric.smogn"
    raise ValueError(method)


def verify_environment() -> None:
    for package, expected in PACKAGE_VERSIONS.items():
        if package == "audit_kernel":
            continue
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {actual}.")
    if len(SEARCH_CONFIGS) != 6 or PIPELINE_CANDIDATES_PER_METHOD != 36:
        raise RuntimeError("The R3.3a budget must remain six configs x six preprocessors.")
    if SVR_GRID_SIZE != 36:
        raise RuntimeError("The R3.3a SVR grid must remain identical to RenSA.")


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 12))
    y = np.linspace(-2.0, 2.0, 30) + rng.normal(scale=0.03, size=30)
    for method in METHODS:
        result = resample_standard(method, X, y, SEARCH_CONFIGS[0])
        assert result.X.shape[1] == X.shape[1]
        assert result.n_output == result.y.size and result.n_output > 0
    print("R3.3a standard-baseline self-test passed.")


def save_checkpoint(
    method: str,
    task: str,
    split_method: str,
    outer_seed: int | None,
    split_result: dict[str, Any],
    predictions: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 2,
        "experiment_key": list(experiment_key(method, task, split_method, outer_seed)),
        "split_result": split_result,
        "predictions": predictions,
        "candidate_scores": candidate_scores,
    }
    destination = checkpoint_path(method, task, split_method, outer_seed)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def rebuild_aggregate_outputs() -> None:
    checkpoints = load_checkpoints()
    split_rows = [item["split_result"] for item in checkpoints]
    prediction_rows = [row for item in checkpoints for row in item["predictions"]]
    candidate_rows = [row for item in checkpoints for row in item["candidate_scores"]]
    atomic_write_csv(RESULTS_DIR / "split_results.csv", SPLIT_RESULT_FIELDS, split_rows)
    atomic_write_csv(RESULTS_DIR / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
    atomic_write_csv(RESULTS_DIR / "candidate_scores.csv", CANDIDATE_SCORE_FIELDS, candidate_rows)
    atomic_write_csv(RESULTS_DIR / "summary.csv", SUMMARY_FIELDS, build_summary(split_rows))
    write_manifest()


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for task in base.DATASETS:
            task_rows = [
                row for row in rows if row["method"] == method and row["task"] == task
            ]
            if not task_rows:
                continue
            spxy = next((row for row in task_rows if row["split_method"] == "spxy"), None)
            ks = next((row for row in task_rows if row["split_method"] == "ks"), None)
            mc = [row for row in task_rows if row["split_method"] == "mc"]
            mc_r2 = np.asarray([float(row["R2"]) for row in mc], dtype=float)
            mc_rmsep = np.asarray([float(row["RMSEP"]) for row in mc], dtype=float)
            output.append({
                "method": method, "task": task,
                "spxy_R2": spxy["R2"] if spxy else "",
                "spxy_RMSEP": spxy["RMSEP"] if spxy else "",
                "ks_R2": ks["R2"] if ks else "",
                "ks_RMSEP": ks["RMSEP"] if ks else "",
                "mc_n": int(mc_r2.size),
                "mc_mean_R2": base.safe_stat(mc_r2, np.mean),
                "mc_SD_R2": base.safe_sd(mc_r2),
                "mc_median_R2": base.safe_stat(mc_r2, np.median),
                "mc_mean_RMSEP": base.safe_stat(mc_rmsep, np.mean),
                "mc_SD_RMSEP": base.safe_sd(mc_rmsep),
                "mc_median_RMSEP": base.safe_stat(mc_rmsep, np.median),
                "mc_min_RMSEP": base.safe_stat(mc_rmsep, np.min),
                "mc_max_RMSEP": base.safe_stat(mc_rmsep, np.max),
            })
    return output


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_checkpoints() -> list[dict[str, Any]]:
    items = []
    if not CHECKPOINT_DIR.exists():
        return items
    for path in CHECKPOINT_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            items.append(json.load(handle))
    method_order = {method: index for index, method in enumerate(METHODS)}
    task_order = {task: index for index, task in enumerate(base.DATASETS)}
    split_order = {"spxy": 0, "ks": 1, "mc": 2}
    items.sort(key=lambda item: (
        method_order[item["split_result"]["method"]],
        task_order[item["split_result"]["task"]],
        split_order[item["split_result"]["split_method"]],
        -1 if item["split_result"]["outer_seed"] == ""
        else int(item["split_result"]["outer_seed"]),
    ))
    return items


def completed_keys() -> set[tuple[str, str, str, str]]:
    return {
        tuple(str(value) for value in item["experiment_key"])
        for item in load_checkpoints()
    }


def experiment_key(
    method: str, task: str, split_method: str, outer_seed: int | None
) -> tuple[str, str, str, str]:
    return method, task, split_method, "NA" if outer_seed is None else str(int(outer_seed))


def checkpoint_path(
    method: str, task: str, split_method: str, outer_seed: int | None
) -> Path:
    seed = "deterministic" if outer_seed is None else f"seed{int(outer_seed)}"
    return CHECKPOINT_DIR / f"{method}__{task}__{split_method}__{seed}.json"


def write_manifest() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "review_item": "R3.3a",
        "methods": {
            "smoter": {
                "implementation": "smogn_smoter_numeric.smoter",
                "version": numeric_standard.IMPLEMENTATION_VERSION,
                "algorithm_reference": numeric_standard.UBL_SMOTER_SOURCE,
                "paper_doi": numeric_standard.SMOTER_PAPER_DOI,
            },
            "smogn": {
                "implementation": "smogn_smoter_numeric.smogn",
                "version": numeric_standard.IMPLEMENTATION_VERSION,
                "algorithm_reference": numeric_standard.UBL_SMOGN_SOURCE,
                "paper_url": numeric_standard.SMOGN_PAPER_URL,
            },
        },
        "implementation_audit": {
            "scope": "finite continuous spectra",
            "local_source": "smogn_smoter_numeric.py",
            "local_source_sha256": source_sha256(Path(numeric_standard.__file__)),
            "conformance": [
                "predictor-only Euclidean kNN and distance matrix",
                "SMOTER inverse-distance response interpolation",
                "SMOGN half-median safe-distance rule",
                "UBL relevance bumps and balance/extreme sampling factors",
                "fully assigned finite synthetic rows",
                "bump-local random under-sampling",
            ],
            "python_reference_sources": {
                "smogn": {
                    "version": PACKAGE_VERSIONS["smogn"],
                    "source": "https://github.com/nickkunz/smogn",
                    "over_sampling_sha256": source_sha256(Path(smogn_over_sampling.__file__)),
                },
                "ImbalancedLearningRegression": {
                    "version": PACKAGE_VERSIONS["ImbalancedLearningRegression"],
                    "source": "https://github.com/paobranco/ImbalancedLearningRegression",
                    "over_sampling_sha256": source_sha256(Path(iblr_over_sampling.__file__)),
                },
            },
        },
        "relevance": {
            "method": "manual",
            "control_quantiles": list(RELEVANCE_CONTROL_QUANTILES),
            "control_relevance": [1.0, 1.0, 0.0, 1.0, 1.0],
            "fit_boundary": "recomputed from each measured-only training fold",
        },
        "candidate_budget": {
            "configs_per_method": len(SEARCH_CONFIGS),
            "preprocessing_methods": list(PREPROCESSING),
            "pipeline_candidates_per_method": PIPELINE_CANDIDATES_PER_METHOD,
            "outer_inner_folds": INNER_CV,
            "svr_grid_combinations": SVR_GRID_SIZE,
            "svr_inner_folds": INNER_CV,
            "matched_rensa_effective_configs": len(base.effective_rensa_configs()),
        },
        "configs": [config.__dict__ for config in SEARCH_CONFIGS],
        "protocol": {
            "augmentation_space": "full continuous preprocessed spectrum",
            "cars_fit": "measured training samples only",
            "validation_and_holdout": "measured samples only",
            "algorithm_seed": ALGORITHM_SEED,
            "random_generator": "numpy.random.default_rng",
            "distance_target_exclusion": "target never enters kNN or safe-distance calculations",
        },
    }
    destination = RESULTS_DIR / "experiment_manifest.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        try:
            if destination.read_text(encoding="utf-8") == serialized:
                return
        except (OSError, UnicodeDecodeError):
            pass
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    try:
        os.replace(temporary, destination)
    except PermissionError:
        # Concurrent task processes may race only on this identical manifest.
        if destination.exists() and destination.read_text(encoding="utf-8") == serialized:
            temporary.unlink(missing_ok=True)
            return
        raise


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / f"run_{os.getpid()}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


if __name__ == "__main__":
    main()
