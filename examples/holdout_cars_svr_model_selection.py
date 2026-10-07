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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_bp_ks_spxy import representative_split
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from run_coal_q_augmentation_demo import preprocess_spectra


SUMMARY_FIELDS = [
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
    "augmentation_guard_rel_rmse",
    "candidate_count",
    "baseline_candidate_count",
    "selected_is_augmented",
    "selected_n_synthetic",
    "selected_perturbation_mode",
    "selected_noise_scale",
    "selected_cars_sampling",
    "selected_cars_min_features",
    "selected_cars_components",
    "selected_inner_cv_rmse",
    "selected_inner_cv_r2",
    "best_baseline_inner_cv_rmse",
    "best_baseline_inner_cv_r2",
    "best_augmented_inner_cv_rmse",
    "best_augmented_inner_cv_r2",
    "default_baseline_test_rmse",
    "default_baseline_test_r2",
    "selected_baseline_test_rmse",
    "selected_baseline_test_r2",
    "selected_test_rmse",
    "selected_test_r2",
    "delta_vs_default_baseline_rmse",
    "delta_vs_default_baseline_r2",
    "delta_vs_selected_baseline_rmse",
    "delta_vs_selected_baseline_r2",
    "bootstrap_delta_vs_default_r2_ci_low",
    "bootstrap_delta_vs_default_r2_ci_high",
    "bootstrap_delta_vs_selected_baseline_r2_ci_low",
    "bootstrap_delta_vs_selected_baseline_r2_ci_high",
    "selected_best_params",
    "selected_features",
    "selected_synthetic_accepted",
    "selected_acceptance_rate",
    "svr_grid_size",
    "train_indices",
    "test_indices",
]


SCORE_FIELDS = [
    "run_id",
    "dataset",
    "preprocess",
    "test_size",
    "spxy_y_weight",
    "rank",
    "is_selected",
    "is_augmented",
    "n_synthetic",
    "perturbation_mode",
    "noise_scale",
    "cars_sampling",
    "cars_min_features",
    "cars_components",
    "inner_cv_rmse",
    "inner_cv_r2",
]


@dataclass(frozen=True)
class AugConfig:
    n_synthetic: int
    perturbation_mode: str
    noise_scale: float

    @property
    def is_augmented(self) -> bool:
        return self.n_synthetic > 0


@dataclass(frozen=True)
class ModelConfig:
    cars_sampling: int
    cars_min_features: int
    cars_components: int
    aug: AugConfig


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Strict train-only selection of CARS complexity and guarded augmentation."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "soil_SOC.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_cars_svr_modelselect.csv")
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=root / "results" / "holdout_cars_svr_modelselect_scores.csv",
    )
    parser.add_argument(
        "--preprocess",
        default="snv-sg",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, nargs="+", default=[0.25])
    parser.add_argument("--spxy-y-weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--augmentation-guard-rel-rmse", type=float, default=0.05)
    parser.add_argument("--n-synthetic", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--perturbation-mode", nargs="+", default=["none", "local_std"], choices=["none", "local_std"])
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0, 0.008])
    parser.add_argument("--cars-sampling", type=int, nargs="+", default=[40])
    parser.add_argument("--cars-min-features", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--cars-components", type=int, nargs="+", default=[8])
    parser.add_argument("--default-cars-sampling", type=int, default=40)
    parser.add_argument("--default-cars-min-features", type=int, default=40)
    parser.add_argument("--default-cars-components", type=int, default=8)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--response-bin-strategy", default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--response-bins", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha-min", type=float, default=0.15)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--svr-C", type=float, nargs="+", default=[1000.0, 3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(
        args.data,
        target_column=args.target_column,
        spectral_prefix=args.spectral_prefix,
    )
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem
    param_grid = {
        "svr__C": [float(value) for value in args.svr_C],
        "svr__gamma": [_parse_gamma(value) for value in args.svr_gamma],
        "svr__epsilon": [float(value) for value in args.svr_epsilon],
    }
    configs = list(model_configs(args))
    completed = completed_keys(args.output) if args.resume else set()
    write_summary_header = not args.output.exists() or args.output.stat().st_size == 0
    write_scores_header = not args.scores_output.exists() or args.scores_output.stat().st_size == 0

    with args.output.open("a", newline="") as summary_handle, args.scores_output.open("a", newline="") as scores_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
        scores_writer = csv.DictWriter(scores_handle, fieldnames=SCORE_FIELDS)
        if write_summary_header:
            summary_writer.writeheader()
        if write_scores_header:
            scores_writer.writeheader()
        for test_size, spxy_y_weight in product(args.test_size, args.spxy_y_weight):
            key = (args.split_method, float(test_size), float(spxy_y_weight))
            if key in completed:
                print(f"skip completed split={key}", flush=True)
                continue
            run_id = int(time.time())
            started = time.perf_counter()
            train_idx, test_idx = representative_split(
                X,
                y,
                test_size=float(test_size),
                method=args.split_method,
                spxy_y_weight=float(spxy_y_weight),
            )
            print(
                f"run split={args.split_method} test={test_size} weight={spxy_y_weight} candidates={len(configs)}",
                flush=True,
            )
            row, score_rows = evaluate_split(
                X,
                y,
                train_idx,
                test_idx,
                args,
                dataset,
                param_grid,
                configs,
                run_id,
                float(test_size),
                float(spxy_y_weight),
            )
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            summary_writer.writerow(row)
            scores_writer.writerows(score_rows)
            summary_handle.flush()
            scores_handle.flush()
            print(
                "  selected CARS=({}, {}, {}) aug={} n={} R2 default={:.4f} selected={:.4f}".format(
                    row["selected_cars_sampling"],
                    row["selected_cars_min_features"],
                    row["selected_cars_components"],
                    bool(row["selected_is_augmented"]),
                    row["selected_n_synthetic"],
                    row["default_baseline_test_r2"],
                    row["selected_test_r2"],
                ),
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.scores_output}", flush=True)


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    param_grid: dict,
    configs: list[ModelConfig],
    run_id: int,
    test_size: float,
    spxy_y_weight: float,
) -> tuple[dict, list[dict]]:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    scored = score_configs(X_train, y_train, args, param_grid, configs)
    scored.sort(key=lambda item: item["inner_cv_rmse"])

    baselines = [item for item in scored if not item["config"].aug.is_augmented]
    augmented = [item for item in scored if item["config"].aug.is_augmented]
    best_baseline = min(baselines, key=lambda item: item["inner_cv_rmse"])
    best_augmented = min(augmented, key=lambda item: item["inner_cv_rmse"])
    guard = float(args.augmentation_guard_rel_rmse)
    if best_augmented["inner_cv_rmse"] <= best_baseline["inner_cv_rmse"] * (1.0 - guard):
        selected = best_augmented
    else:
        selected = best_baseline

    default_baseline_config = ModelConfig(
        args.default_cars_sampling,
        args.default_cars_min_features,
        args.default_cars_components,
        AugConfig(0, "none", 0.0),
    )
    default_baseline = fit_final(X_train, y_train, X_test, y_test, args, param_grid, default_baseline_config)
    selected_baseline = fit_final(X_train, y_train, X_test, y_test, args, param_grid, best_baseline["config"])
    selected_final = fit_final(X_train, y_train, X_test, y_test, args, param_grid, selected["config"])
    stats_default = bootstrap_delta_r2(y_test, default_baseline["pred"], selected_final["pred"], args.bootstrap, args.random_state)
    stats_selected_baseline = bootstrap_delta_r2(
        y_test,
        selected_baseline["pred"],
        selected_final["pred"],
        args.bootstrap,
        args.random_state,
    )

    score_rows = []
    for rank, item in enumerate(scored, start=1):
        config = item["config"]
        score_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "preprocess": args.preprocess,
                "test_size": test_size,
                "spxy_y_weight": spxy_y_weight,
                "rank": rank,
                "is_selected": int(item is selected),
                "is_augmented": int(config.aug.is_augmented),
                "n_synthetic": config.aug.n_synthetic,
                "perturbation_mode": config.aug.perturbation_mode,
                "noise_scale": config.aug.noise_scale,
                "cars_sampling": config.cars_sampling,
                "cars_min_features": config.cars_min_features,
                "cars_components": config.cars_components,
                "inner_cv_rmse": item["inner_cv_rmse"],
                "inner_cv_r2": item["inner_cv_r2"],
            }
        )

    config = selected["config"]
    row = {
        "run_id": run_id,
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "split_method": args.split_method,
        "test_size": test_size,
        "spxy_y_weight": spxy_y_weight,
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "inner_cv": args.inner_cv,
        "augmentation_guard_rel_rmse": guard,
        "candidate_count": len(configs),
        "baseline_candidate_count": len(baselines),
        "selected_is_augmented": int(config.aug.is_augmented),
        "selected_n_synthetic": config.aug.n_synthetic,
        "selected_perturbation_mode": config.aug.perturbation_mode,
        "selected_noise_scale": config.aug.noise_scale,
        "selected_cars_sampling": config.cars_sampling,
        "selected_cars_min_features": config.cars_min_features,
        "selected_cars_components": config.cars_components,
        "selected_inner_cv_rmse": selected["inner_cv_rmse"],
        "selected_inner_cv_r2": selected["inner_cv_r2"],
        "best_baseline_inner_cv_rmse": best_baseline["inner_cv_rmse"],
        "best_baseline_inner_cv_r2": best_baseline["inner_cv_r2"],
        "best_augmented_inner_cv_rmse": best_augmented["inner_cv_rmse"],
        "best_augmented_inner_cv_r2": best_augmented["inner_cv_r2"],
        "default_baseline_test_rmse": default_baseline["rmse"],
        "default_baseline_test_r2": default_baseline["r2"],
        "selected_baseline_test_rmse": selected_baseline["rmse"],
        "selected_baseline_test_r2": selected_baseline["r2"],
        "selected_test_rmse": selected_final["rmse"],
        "selected_test_r2": selected_final["r2"],
        "delta_vs_default_baseline_rmse": selected_final["rmse"] - default_baseline["rmse"],
        "delta_vs_default_baseline_r2": selected_final["r2"] - default_baseline["r2"],
        "delta_vs_selected_baseline_rmse": selected_final["rmse"] - selected_baseline["rmse"],
        "delta_vs_selected_baseline_r2": selected_final["r2"] - selected_baseline["r2"],
        "bootstrap_delta_vs_default_r2_ci_low": stats_default["low"],
        "bootstrap_delta_vs_default_r2_ci_high": stats_default["high"],
        "bootstrap_delta_vs_selected_baseline_r2_ci_low": stats_selected_baseline["low"],
        "bootstrap_delta_vs_selected_baseline_r2_ci_high": stats_selected_baseline["high"],
        "selected_best_params": json.dumps(selected_final["best_params"]),
        "selected_features": selected_final["selected_features"],
        "selected_synthetic_accepted": selected_final["synthetic_accepted"],
        "selected_acceptance_rate": selected_final["acceptance_rate"],
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "train_indices": json.dumps(train_idx.tolist()),
        "test_indices": json.dumps(test_idx.tolist()),
    }
    return row, score_rows


def score_configs(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    configs: list[ModelConfig],
) -> list[dict]:
    splitter = KFold(n_splits=min(args.inner_cv, X.shape[0]), shuffle=True, random_state=args.random_state)
    folds = list(splitter.split(X))
    predictions = {config: np.empty_like(y, dtype=float) for config in configs}

    for fold_idx, (fit_idx, valid_idx) in enumerate(folds):
        y_fit_original = y[fit_idx]
        y_valid = y[valid_idx]
        transformed = {}
        for cars_key in sorted({(c.cars_sampling, c.cars_min_features, c.cars_components) for c in configs}):
            transformer = make_cars(*cars_key, args=args, fold_idx=fold_idx)
            Z_fit = transformer.fit_transform(X[fit_idx], y_fit_original)
            Z_valid = transformer.transform(X[valid_idx])
            transformed[cars_key] = (Z_fit, Z_valid)

        for config in configs:
            Z_fit_original, Z_valid = transformed[
                (config.cars_sampling, config.cars_min_features, config.cars_components)
            ]
            Z_fit, y_fit = maybe_augment(Z_fit_original, y_fit_original, args, config.aug, fold_idx)
            search = make_svr_search(param_grid, args, Z_fit.shape[0], fold_idx)
            search.fit(Z_fit, y_fit)
            predictions[config][valid_idx] = search.predict(Z_valid).reshape(-1)
        print(f"    inner fold {fold_idx + 1}/{len(folds)} scored", flush=True)

    scored = []
    for config in configs:
        pred = predictions[config]
        scored.append(
            {
                "config": config,
                "inner_cv_rmse": float(mean_squared_error(y, pred, squared=False)),
                "inner_cv_r2": float(r2_score(y, pred)),
            }
        )
    return scored


def fit_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    config: ModelConfig,
) -> dict:
    transformer = make_cars(
        config.cars_sampling,
        config.cars_min_features,
        config.cars_components,
        args=args,
        fold_idx=0,
    )
    Z_train_original = transformer.fit_transform(X_train, y_train)
    Z_test = transformer.transform(X_test)
    Z_fit, y_fit = maybe_augment(Z_train_original, y_train, args, config.aug, fold_idx=0)
    search = make_svr_search(param_grid, args, Z_fit.shape[0], fold_idx=0)
    search.fit(Z_fit, y_fit)
    pred = search.predict(Z_test).reshape(-1)
    return {
        "pred": pred,
        "rmse": float(mean_squared_error(y_test, pred, squared=False)),
        "r2": float(r2_score(y_test, pred)),
        "best_params": search.best_params_,
        "selected_features": int(Z_train_original.shape[1]),
        "synthetic_accepted": max(0, int(Z_fit.shape[0] - Z_train_original.shape[0])),
        "acceptance_rate": float((Z_fit.shape[0] - Z_train_original.shape[0]) / max(config.aug.n_synthetic, 1))
        if config.aug.is_augmented
        else 0.0,
    }


def maybe_augment(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    config: AugConfig,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.is_augmented:
        return X, y
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
    return result.X, result.y


def make_cars(
    cars_sampling: int,
    cars_min_features: int,
    cars_components: int,
    args: argparse.Namespace,
    fold_idx: int,
) -> CARSFeatureSelector:
    return CARSFeatureSelector(
        n_sampling=int(cars_sampling),
        min_features=int(cars_min_features),
        max_components=int(cars_components),
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


def bootstrap_delta_r2(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    selected_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    if int(n_bootstrap) <= 0:
        return {"low": "", "high": ""}
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
        if np.allclose(y_true[idx].min(), y_true[idx].max()):
            continue
        deltas.append(float(r2_score(y_true[idx], selected_pred[idx]) - r2_score(y_true[idx], baseline_pred[idx])))
    arr = np.asarray(deltas, dtype=float)
    return {"low": float(np.quantile(arr, 0.025)), "high": float(np.quantile(arr, 0.975))}


def model_configs(args: argparse.Namespace):
    aug_configs = [AugConfig(0, "none", 0.0)]
    seen = set()
    for n_synthetic, perturbation_mode, noise_scale in product(
        args.n_synthetic,
        args.perturbation_mode,
        args.noise_scale,
    ):
        aug = AugConfig(int(n_synthetic), str(perturbation_mode), float(noise_scale))
        key = effective_aug_key(aug)
        if key in seen:
            continue
        seen.add(key)
        aug_configs.append(aug)

    for cars_sampling, cars_min_features, cars_components, aug in product(
        args.cars_sampling,
        args.cars_min_features,
        args.cars_components,
        aug_configs,
    ):
        yield ModelConfig(int(cars_sampling), int(cars_min_features), int(cars_components), aug)


def effective_aug_key(config: AugConfig) -> tuple:
    if not config.is_augmented:
        return (0, "none", 0.0)
    if config.perturbation_mode == "none" or config.noise_scale <= 0:
        return (config.n_synthetic, "none", 0.0)
    return (config.n_synthetic, config.perturbation_mode, config.noise_scale)


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["split_method"], float(row["test_size"]), float(row["spxy_y_weight"]))
            for row in csv.DictReader(handle)
        }


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


if __name__ == "__main__":
    main()
