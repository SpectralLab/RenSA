from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from respond_spectra.preprocessing import msc, savgol, snv


SUMMARY_FIELDS = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "protocol",
    "split_method",
    "test_size",
    "spxy_y_weight",
    "train_size",
    "test_count",
    "inner_cv",
    "selection_metric",
    "preprocess_candidates",
    "candidate_count",
    "selected_rank",
    "selected_preprocess",
    "selected_is_augmented",
    "selected_n_synthetic",
    "selected_perturbation_mode",
    "selected_noise_scale",
    "selected_inner_cv_rmse",
    "selected_inner_cv_r2",
    "baseline_preprocess",
    "baseline_inner_cv_rmse",
    "baseline_inner_cv_r2",
    "baseline_test_rmse",
    "baseline_test_r2",
    "selected_test_rmse",
    "selected_test_r2",
    "delta_test_rmse",
    "delta_test_r2",
    "bootstrap_delta_rmse_ci_low",
    "bootstrap_delta_rmse_ci_high",
    "bootstrap_delta_r2_ci_low",
    "bootstrap_delta_r2_ci_high",
    "bootstrap_p_rmse_improved",
    "bootstrap_p_r2_improved",
    "baseline_best_params",
    "selected_best_params",
    "baseline_selected_features",
    "selected_selected_features",
    "selected_synthetic_accepted",
    "selected_acceptance_rate",
    "cars_sampling",
    "cars_min_features",
    "cars_components",
    "svr_grid_size",
    "neighbor_space",
    "response_bin_strategy",
    "response_bins",
    "neighbors",
    "alpha_min",
    "alpha_max",
    "train_indices",
    "test_indices",
]


SCORE_FIELDS = [
    "run_id",
    "dataset",
    "protocol",
    "split_method",
    "test_size",
    "spxy_y_weight",
    "rank",
    "is_selected",
    "preprocess",
    "is_augmented",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "inner_cv_rmse",
    "inner_cv_r2",
    "fold_rmse",
    "fold_r2",
    "fold_synthetic_accepted",
    "fold_acceptance_rate",
]


PREDICTION_FIELDS = [
    "run_id",
    "dataset",
    "protocol",
    "split_method",
    "test_size",
    "spxy_y_weight",
    "sample_index",
    "y_true",
    "baseline_preprocess",
    "baseline_pred",
    "selected_preprocess",
    "selected_is_augmented",
    "selected_n_synthetic",
    "selected_perturbation_mode",
    "selected_noise_scale",
    "selected_pred",
    "baseline_residual",
    "selected_residual",
]


@dataclass(frozen=True)
class AugConfig:
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float

    @property
    def is_augmented(self) -> bool:
        return self.n_synthetic > 0

    def label(self) -> str:
        if not self.is_augmented:
            return "baseline"
        return f"n{self.n_synthetic}_{self.perturbation_mode}_noise{self.noise_scale:g}"


@dataclass(frozen=True)
class Candidate:
    preprocess: str
    config: AugConfig


@dataclass
class PreprocessModel:
    method: str
    msc_reference: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        window_length = min(21, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
        if self.method == "raw":
            return X
        if self.method == "snv":
            return snv(X)
        if self.method == "sg":
            return savgol(X, window_length=window_length, polyorder=2)
        if self.method == "snv-sg":
            return savgol(snv(X), window_length=window_length, polyorder=2)
        if self.method == "msc":
            return msc(X, reference=self.msc_reference)
        if self.method == "msc-sg":
            return savgol(msc(X, reference=self.msc_reference), window_length=window_length, polyorder=2)
        raise ValueError(f"Unknown preprocessing method: {self.method}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Unified strict train-only protocol for CARS-RBF-SVR: outer KS/SPXY "
            "holdout, inner-CV preprocessing and augmentation selection, one final test evaluation."
        )
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--loader", choices=["xprefix", "diesel"], default="xprefix")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_unified_trainonly_protocol.csv")
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=root / "results" / "holdout_unified_trainonly_protocol_scores.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=root / "results" / "holdout_unified_trainonly_protocol_predictions.csv",
    )
    parser.add_argument(
        "--preprocess-candidates",
        nargs="+",
        default=["raw", "snv", "sg", "snv-sg", "msc", "msc-sg"],
        choices=["raw", "snv", "sg", "snv-sg", "msc", "msc-sg"],
    )
    parser.add_argument("--protocol", choices=["free", "constrained"], default="free")
    parser.add_argument(
        "--protocol-label",
        default="",
        help="Optional label written to outputs, e.g. free-bounded-svr, while preserving protocol logic.",
    )
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, nargs="+", default=[0.25])
    parser.add_argument("--spxy-y-weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--perturbation-mode", nargs="+", default=["none", "local_std"], choices=["none", "local_std"])
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0, 0.008])
    parser.add_argument("--fixed-n-synthetic", type=int, default=80)
    parser.add_argument("--fixed-perturbation-mode", default="local_std", choices=["none", "local_std"])
    parser.add_argument("--fixed-noise-scale", type=float, default=0.008)
    parser.add_argument("--dedupe-equivalent-configs", action="store_true")
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--response-bin-strategy", default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--response-bins", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha-min", type=float, default=0.15)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)

    _, X, y = load_dataset(args)
    dataset = args.dataset_name or args.data.stem
    args.protocol_name = args.protocol_label or args.protocol
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }
    configs = list(candidate_configs(args))
    candidates = [Candidate(preprocess, config) for preprocess in args.preprocess_candidates for config in configs]
    completed = completed_keys(args.output) if args.resume else set()

    summary_header = not args.output.exists() or args.output.stat().st_size == 0
    scores_header = not args.scores_output.exists() or args.scores_output.stat().st_size == 0
    predictions_header = not args.predictions_output.exists() or args.predictions_output.stat().st_size == 0
    n_new = 0
    with (
        args.output.open("a", newline="") as summary_handle,
        args.scores_output.open("a", newline="") as scores_handle,
        args.predictions_output.open("a", newline="") as predictions_handle,
    ):
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
        scores_writer = csv.DictWriter(scores_handle, fieldnames=SCORE_FIELDS)
        predictions_writer = csv.DictWriter(predictions_handle, fieldnames=PREDICTION_FIELDS)
        if summary_header:
            summary_writer.writeheader()
        if scores_header:
            scores_writer.writeheader()
        if predictions_header:
            predictions_writer.writeheader()

        for test_size, spxy_y_weight in product(args.test_size, args.spxy_y_weight):
            key = (dataset, args.protocol_name, args.split_method, float(test_size), float(spxy_y_weight))
            if key in completed:
                print(f"skip completed split={key}", flush=True)
                continue
            if args.max_runs and n_new >= args.max_runs:
                break
            n_new += 1
            started = time.perf_counter()
            run_id = int(time.time())
            train_idx, test_idx = representative_split(
                X,
                y,
                test_size=float(test_size),
                method=args.split_method,
                spxy_y_weight=float(spxy_y_weight),
            )
            print(
                "run {}: dataset={} protocol={} split={} test={} spxy_y_weight={} candidates={}".format(
                    n_new,
                    dataset,
                    args.protocol_name,
                    args.split_method,
                    test_size,
                    spxy_y_weight,
                    len(candidates),
                ),
                flush=True,
            )
            row, score_rows, prediction_rows = evaluate_split(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                args=args,
                dataset=dataset,
                param_grid=param_grid,
                candidates=candidates,
                run_id=run_id,
                test_size=float(test_size),
                spxy_y_weight=float(spxy_y_weight),
            )
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            summary_writer.writerow(row)
            scores_writer.writerows(score_rows)
            predictions_writer.writerows(prediction_rows)
            summary_handle.flush()
            scores_handle.flush()
            predictions_handle.flush()
            print(
                "  baseline={} selected={} {} inner_rmse={:.4f}; test_r2={:.4f}->{:.4f} delta={:+.4f}".format(
                    row["baseline_preprocess"],
                    row["selected_preprocess"],
                    selected_config_from_row(row).label(),
                    row["selected_inner_cv_rmse"],
                    row["baseline_test_r2"],
                    row["selected_test_r2"],
                    row["delta_test_r2"],
                ),
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.scores_output}", flush=True)
    print(f"wrote {args.predictions_output}", flush=True)


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    param_grid: dict,
    candidates: list[Candidate],
    run_id: int,
    test_size: float,
    spxy_y_weight: float,
) -> tuple[dict, list[dict], list[dict]]:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    scored = score_candidates_by_inner_cv(X_train, y_train, args, param_grid, candidates)
    scored.sort(key=lambda item: item["inner_cv_rmse"])
    selected = scored[0]
    baseline_cv = min(
        (item for item in scored if not item["candidate"].config.is_augmented),
        key=lambda item: item["inner_cv_rmse"],
    )

    baseline_final = fit_final_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        args=args,
        param_grid=param_grid,
        candidate=baseline_cv["candidate"],
    )
    selected_final = fit_final_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        args=args,
        param_grid=param_grid,
        candidate=selected["candidate"],
    )
    stats = bootstrap_deltas(
        y_test,
        baseline_final["pred"],
        selected_final["pred"],
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )

    score_rows = []
    for rank, item in enumerate(scored, start=1):
        candidate = item["candidate"]
        config = candidate.config
        score_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "protocol": args.protocol_name,
                "split_method": args.split_method,
                "test_size": test_size,
                "spxy_y_weight": spxy_y_weight,
                "rank": rank,
                "is_selected": int(rank == 1),
                "preprocess": candidate.preprocess,
                "is_augmented": int(config.is_augmented),
                "n_synthetic": config.n_synthetic,
                "perturbation_mode": config.perturbation_mode,
                "noise_scale": config.noise_scale,
                "inner_cv_rmse": item["inner_cv_rmse"],
                "inner_cv_r2": item["inner_cv_r2"],
                "fold_rmse": json.dumps(item["fold_rmse"]),
                "fold_r2": json.dumps(item["fold_r2"]),
                "fold_synthetic_accepted": json.dumps(item["fold_synthetic_accepted"]),
                "fold_acceptance_rate": json.dumps(item["fold_acceptance_rate"]),
            }
        )

    prediction_rows = []
    selected_candidate = selected["candidate"]
    baseline_candidate = baseline_cv["candidate"]
    config = selected_candidate.config
    for sample_index, y_true, baseline_pred, selected_pred in zip(
        test_idx,
        y_test,
        baseline_final["pred"],
        selected_final["pred"],
    ):
        prediction_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "protocol": args.protocol_name,
                "split_method": args.split_method,
                "test_size": test_size,
                "spxy_y_weight": spxy_y_weight,
                "sample_index": int(sample_index),
                "y_true": float(y_true),
                "baseline_preprocess": baseline_candidate.preprocess,
                "baseline_pred": float(baseline_pred),
                "selected_preprocess": selected_candidate.preprocess,
                "selected_is_augmented": int(config.is_augmented),
                "selected_n_synthetic": config.n_synthetic,
                "selected_perturbation_mode": config.perturbation_mode,
                "selected_noise_scale": config.noise_scale,
                "selected_pred": float(selected_pred),
                "baseline_residual": float(y_true - baseline_pred),
                "selected_residual": float(y_true - selected_pred),
            }
        )

    row = {
        "run_id": run_id,
        "elapsed_sec": "",
        "dataset": dataset,
        "protocol": args.protocol_name,
        "split_method": args.split_method,
        "test_size": test_size,
        "spxy_y_weight": spxy_y_weight,
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "inner_cv": args.inner_cv,
        "selection_metric": "inner_cv_rmse",
        "preprocess_candidates": json.dumps(args.preprocess_candidates),
        "candidate_count": len(candidates),
        "selected_rank": 1,
        "selected_preprocess": selected_candidate.preprocess,
        "selected_is_augmented": int(config.is_augmented),
        "selected_n_synthetic": config.n_synthetic,
        "selected_perturbation_mode": config.perturbation_mode,
        "selected_noise_scale": config.noise_scale,
        "selected_inner_cv_rmse": selected["inner_cv_rmse"],
        "selected_inner_cv_r2": selected["inner_cv_r2"],
        "baseline_preprocess": baseline_candidate.preprocess,
        "baseline_inner_cv_rmse": baseline_cv["inner_cv_rmse"],
        "baseline_inner_cv_r2": baseline_cv["inner_cv_r2"],
        "baseline_test_rmse": baseline_final["rmse"],
        "baseline_test_r2": baseline_final["r2"],
        "selected_test_rmse": selected_final["rmse"],
        "selected_test_r2": selected_final["r2"],
        "delta_test_rmse": selected_final["rmse"] - baseline_final["rmse"],
        "delta_test_r2": selected_final["r2"] - baseline_final["r2"],
        **stats,
        "baseline_best_params": json.dumps(baseline_final["best_params"]),
        "selected_best_params": json.dumps(selected_final["best_params"]),
        "baseline_selected_features": baseline_final["selected_features"],
        "selected_selected_features": selected_final["selected_features"],
        "selected_synthetic_accepted": selected_final["synthetic_accepted"],
        "selected_acceptance_rate": selected_final["acceptance_rate"],
        "cars_sampling": args.cars_sampling,
        "cars_min_features": args.cars_min_features,
        "cars_components": args.cars_components,
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "neighbor_space": args.neighbor_space,
        "response_bin_strategy": args.response_bin_strategy,
        "response_bins": args.response_bins,
        "neighbors": args.neighbors,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "train_indices": json.dumps(train_idx.tolist()),
        "test_indices": json.dumps(test_idx.tolist()),
    }
    return row, score_rows, prediction_rows


def score_candidates_by_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    candidates: list[Candidate],
) -> list[dict]:
    splitter = KFold(n_splits=min(args.inner_cv, X.shape[0]), shuffle=True, random_state=args.random_state)
    folds = list(splitter.split(X))
    representative_by_key = {}
    for candidate in candidates:
        representative_by_key.setdefault(effective_candidate_key(candidate), candidate)
    reps = list(representative_by_key.values())

    predictions = {candidate: np.empty_like(y, dtype=float) for candidate in reps}
    fold_stats = {
        candidate: {
            "fold_rmse": [],
            "fold_r2": [],
            "fold_synthetic_accepted": [],
            "fold_acceptance_rate": [],
        }
        for candidate in reps
    }

    for fold_idx, (fit_idx, valid_idx) in enumerate(folds):
        for preprocess in sorted({candidate.preprocess for candidate in reps}):
            preprocessor = fit_preprocessor(X[fit_idx], preprocess)
            X_fit_pre = preprocessor.transform(X[fit_idx])
            X_valid_pre = preprocessor.transform(X[valid_idx])
            transformer = make_cars(args, fold_idx)
            Z_fit_original = transformer.fit_transform(X_fit_pre, y[fit_idx])
            Z_valid = transformer.transform(X_valid_pre)
            y_fit_original = y[fit_idx]
            y_valid = y[valid_idx]

            for candidate in [item for item in reps if item.preprocess == preprocess]:
                Z_fit, y_fit, accepted, rate = maybe_augment(
                    Z_fit_original,
                    y_fit_original,
                    args,
                    candidate.config,
                    fold_idx,
                )
                search = make_svr_search(param_grid, args, Z_fit.shape[0], fold_idx)
                search.fit(Z_fit, y_fit)
                pred = search.predict(Z_valid).reshape(-1)
                predictions[candidate][valid_idx] = pred
                fold_stats[candidate]["fold_rmse"].append(float(mean_squared_error(y_valid, pred, squared=False)))
                fold_stats[candidate]["fold_r2"].append(float(r2_score(y_valid, pred)))
                fold_stats[candidate]["fold_synthetic_accepted"].append(int(accepted))
                fold_stats[candidate]["fold_acceptance_rate"].append(float(rate))
        print(
            "    inner fold {}/{} scored {} effective candidates".format(
                fold_idx + 1,
                len(folds),
                len(reps),
            ),
            flush=True,
        )

    scored = []
    for candidate in candidates:
        rep = representative_by_key[effective_candidate_key(candidate)]
        pred = predictions[rep]
        scored.append(
            {
                "candidate": candidate,
                "inner_cv_rmse": float(mean_squared_error(y, pred, squared=False)),
                "inner_cv_r2": float(r2_score(y, pred)),
                **fold_stats[rep],
            }
        )
    return scored


def fit_final_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    candidate: Candidate,
) -> dict:
    preprocessor = fit_preprocessor(X_train, candidate.preprocess)
    X_train_pre = preprocessor.transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    transformer = make_cars(args, fold_idx=0)
    Z_train_original = transformer.fit_transform(X_train_pre, y_train)
    Z_test = transformer.transform(X_test_pre)
    Z_fit, y_fit, accepted, rate = maybe_augment(Z_train_original, y_train, args, candidate.config, fold_idx=0)
    search = make_svr_search(param_grid, args, Z_fit.shape[0], fold_idx=0)
    search.fit(Z_fit, y_fit)
    pred = search.predict(Z_test).reshape(-1)
    return {
        "pred": pred,
        "rmse": float(mean_squared_error(y_test, pred, squared=False)),
        "r2": float(r2_score(y_test, pred)),
        "best_params": search.best_params_,
        "selected_features": int(Z_train_original.shape[1]),
        "synthetic_accepted": int(accepted),
        "acceptance_rate": float(rate),
    }


def fit_preprocessor(X_train: np.ndarray, method: str) -> PreprocessModel:
    reference = X_train.mean(axis=0) if method in {"msc", "msc-sg"} else None
    return PreprocessModel(method=method, msc_reference=reference)


def maybe_augment(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    config: AugConfig,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if not config.is_augmented:
        return X, y, 0, 0.0
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=config.n_synthetic,
        response_bins=args.response_bins,
        response_bin_strategy=args.response_bin_strategy,
        neighbors=args.neighbors,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scale=config.noise_scale,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=config.perturbation_mode,
    )
    result = _clone_augmenter_for_fold(augmenter, fold_idx).fit_resample(X, y)
    return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])


def make_cars(args: argparse.Namespace, fold_idx: int) -> CARSFeatureSelector:
    return CARSFeatureSelector(
        n_sampling=args.cars_sampling,
        min_features=args.cars_min_features,
        max_components=args.cars_components,
        cv=min(args.inner_cv, 5),
        random_state=int(args.random_state) + int(fold_idx),
    )


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


def bootstrap_deltas(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    selected_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    empty = {
        "bootstrap_delta_rmse_ci_low": "",
        "bootstrap_delta_rmse_ci_high": "",
        "bootstrap_delta_r2_ci_low": "",
        "bootstrap_delta_r2_ci_high": "",
        "bootstrap_p_rmse_improved": "",
        "bootstrap_p_r2_improved": "",
    }
    if int(n_bootstrap) <= 0:
        return empty
    rng = np.random.default_rng(random_state)
    n = y_true.shape[0]
    delta_rmse = []
    delta_r2 = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        if np.allclose(y_true[idx].min(), y_true[idx].max()):
            continue
        base_rmse = mean_squared_error(y_true[idx], baseline_pred[idx], squared=False)
        sel_rmse = mean_squared_error(y_true[idx], selected_pred[idx], squared=False)
        base_r2 = r2_score(y_true[idx], baseline_pred[idx])
        sel_r2 = r2_score(y_true[idx], selected_pred[idx])
        delta_rmse.append(float(sel_rmse - base_rmse))
        delta_r2.append(float(sel_r2 - base_r2))
    if not delta_rmse:
        return empty
    delta_rmse_arr = np.asarray(delta_rmse, dtype=float)
    delta_r2_arr = np.asarray(delta_r2, dtype=float)
    return {
        "bootstrap_delta_rmse_ci_low": float(np.quantile(delta_rmse_arr, 0.025)),
        "bootstrap_delta_rmse_ci_high": float(np.quantile(delta_rmse_arr, 0.975)),
        "bootstrap_delta_r2_ci_low": float(np.quantile(delta_r2_arr, 0.025)),
        "bootstrap_delta_r2_ci_high": float(np.quantile(delta_r2_arr, 0.975)),
        "bootstrap_p_rmse_improved": float(np.mean(delta_rmse_arr < 0.0)),
        "bootstrap_p_r2_improved": float(np.mean(delta_r2_arr > 0.0)),
    }


def candidate_configs(args: argparse.Namespace):
    yield AugConfig(0, "none", 0.0)
    if args.protocol == "constrained":
        yield AugConfig(
            int(args.fixed_n_synthetic),
            str(args.fixed_perturbation_mode),
            float(args.fixed_noise_scale),
        )
        return
    seen = set()
    for n_synthetic, perturbation_mode, noise_scale in product(
        args.n_synthetic,
        args.perturbation_mode,
        args.noise_scale,
    ):
        config = AugConfig(int(n_synthetic), str(perturbation_mode), float(noise_scale))
        if args.dedupe_equivalent_configs:
            key = effective_config_key(config)
            if key in seen:
                continue
            seen.add(key)
        yield config


def effective_candidate_key(candidate: Candidate) -> tuple:
    return (candidate.preprocess, *effective_config_key(candidate.config))


def effective_config_key(config: AugConfig) -> tuple:
    if not config.is_augmented:
        return (0, "none", 0.0)
    if config.perturbation_mode == "none" or config.noise_scale <= 0:
        return (config.n_synthetic, "none", 0.0)
    return (config.n_synthetic, config.perturbation_mode, config.noise_scale)


def selected_config_from_row(row: dict) -> AugConfig:
    return AugConfig(
        int(row["selected_n_synthetic"]),
        str(row["selected_perturbation_mode"]),
        float(row["selected_noise_scale"]),
    )


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["dataset"], row["protocol"], row["split_method"], float(row["test_size"]), float(row["spxy_y_weight"]))
            for row in csv.DictReader(handle)
        }


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.loader == "xprefix":
        return load_spectrum_csv(
            args.data,
            target_column=args.target_column,
            spectral_prefix=args.spectral_prefix,
        )
    frame = pd.read_csv(args.data)
    y = frame[args.target_column].to_numpy(dtype=float).reshape(-1)
    spectral_columns = [
        column
        for column in frame.columns
        if column not in {"Label", args.target_column} and _is_float_like(column)
    ]
    X = frame.loc[:, spectral_columns].to_numpy(dtype=float)
    wavelengths = np.asarray([float(column) for column in spectral_columns], dtype=float)
    return wavelengths, X, y


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def _is_float_like(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
