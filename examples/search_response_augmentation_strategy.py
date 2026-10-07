from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold, _validated_cv_inputs
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "model",
    "preprocess",
    "cv",
    "cv_seeds",
    "inner_cv",
    "cars_sampling",
    "cars_min_features",
    "cars_components",
    "svr_grid_size",
    "n_synthetic",
    "neighbor_space",
    "spectrum_weight",
    "neighbors",
    "response_bins",
    "response_bin_strategy",
    "alpha_min",
    "alpha_max",
    "perturbation_mode",
    "noise_scale",
    "min_derivative_corr",
    "max_spectral_angle",
    "envelope_margin",
    "response_consistency",
    "baseline_mean_rmse",
    "baseline_std_rmse",
    "baseline_mean_r2",
    "baseline_std_r2",
    "baseline_worst_seed_r2",
    "baseline_worst_fold_r2",
    "augmented_mean_rmse",
    "augmented_std_rmse",
    "augmented_mean_r2",
    "augmented_std_r2",
    "augmented_worst_seed_r2",
    "augmented_worst_fold_r2",
    "delta_mean_rmse",
    "delta_mean_r2",
    "delta_std_r2",
    "delta_worst_seed_r2",
    "delta_worst_fold_r2",
    "r2_improvement_rate",
    "rmse_improvement_rate",
    "baseline_r2_by_seed",
    "augmented_r2_by_seed",
    "baseline_rmse_by_seed",
    "augmented_rmse_by_seed",
    "delta_r2_by_seed",
    "delta_rmse_by_seed",
    "augmented_acceptance_rate_mean",
    "augmented_acceptance_rate_by_seed",
    "augmented_synthetic_accepted_by_seed",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Search response-driven spectral augmentation strategies with stability metrics."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "response_augmentation_strategy.csv",
    )
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--cv-seeds", type=int, nargs="+", default=[11, 29, 42])
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-min-features", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--n-synthetic", type=int, nargs="+", default=[0, 40, 60, 80, 100])
    parser.add_argument("--neighbor-space", nargs="+", default=["response"], choices=["joint", "spectrum", "response"])
    parser.add_argument("--spectrum-weight", type=float, nargs="+", default=[0.35])
    parser.add_argument("--neighbors", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--response-bins", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument(
        "--response-bin-strategy",
        nargs="+",
        default=["quantile"],
        choices=["quantile", "uniform"],
    )
    parser.add_argument("--alpha-min", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--alpha-max", type=float, nargs="+", default=[0.80, 0.90])
    parser.add_argument("--perturbation-mode", nargs="+", default=["local_std"], choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0, 0.004, 0.008])
    parser.add_argument("--min-derivative-corr", type=float, default=0.70)
    parser.add_argument("--max-spectral-angle", type=float, default=0.18)
    parser.add_argument("--envelope-margin", type=float, default=0.08)
    parser.add_argument("--response-consistency", action="store_true")
    parser.add_argument("--svr-C", type=float, nargs="+", default=[3000.0, 5000.0, 10000.0])
    parser.add_argument("--svr-gamma", nargs="+", default=["0.0015", "0.002", "0.003", "0.005"])
    parser.add_argument("--svr-epsilon", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem

    param_grid = {
        "svr__C": [float(v) for v in args.svr_C],
        "svr__gamma": [_parse_gamma(v) for v in args.svr_gamma],
        "svr__epsilon": [float(v) for v in args.svr_epsilon],
    }
    configs = list(iter_augmentation_configs(args))
    completed = _completed_keys(args.output) if args.resume else set()
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    baseline_cache: dict[int, dict] = {}
    n_new = 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for config in configs:
            key = config_key(config)
            if key in completed:
                print(f"skip completed {key}", flush=True)
                continue
            if args.max_runs and n_new >= args.max_runs:
                break

            n_new += 1
            started = time.perf_counter()
            print(f"run {n_new}: {key}", flush=True)
            row = evaluate_config(
                X=X,
                y=y,
                args=args,
                dataset=dataset,
                param_grid=param_grid,
                config=config,
                baseline_cache=baseline_cache,
            )
            row["run_id"] = int(time.time())
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            writer.writerow(row)
            handle.flush()
            print(
                "  baseline_r2={:.4f} augmented_r2={:.4f} delta={:+.4f} "
                "aug_std_r2={:.4f}".format(
                    row["baseline_mean_r2"],
                    row["augmented_mean_r2"],
                    row["delta_mean_r2"],
                    row["augmented_std_r2"],
                ),
                flush=True,
            )

    print(f"wrote {args.output}", flush=True)


def iter_augmentation_configs(args: argparse.Namespace):
    for (
        n_synthetic,
        neighbor_space,
        spectrum_weight,
        neighbors,
        response_bins,
        response_bin_strategy,
        alpha_min,
        alpha_max,
        perturbation_mode,
        noise_scale,
    ) in product(
        args.n_synthetic,
        args.neighbor_space,
        args.spectrum_weight,
        args.neighbors,
        args.response_bins,
        args.response_bin_strategy,
        args.alpha_min,
        args.alpha_max,
        args.perturbation_mode,
        args.noise_scale,
    ):
        if alpha_min >= alpha_max:
            continue
        if n_synthetic == 0 and (
            neighbor_space != args.neighbor_space[0]
            or spectrum_weight != args.spectrum_weight[0]
            or neighbors != args.neighbors[0]
            or response_bins != args.response_bins[0]
            or response_bin_strategy != args.response_bin_strategy[0]
            or alpha_min != args.alpha_min[0]
            or alpha_max != args.alpha_max[0]
            or perturbation_mode != args.perturbation_mode[0]
            or noise_scale != args.noise_scale[0]
        ):
            continue
        yield {
            "n_synthetic": int(n_synthetic),
            "neighbor_space": str(neighbor_space),
            "spectrum_weight": float(spectrum_weight),
            "neighbors": int(neighbors),
            "response_bins": int(response_bins),
            "response_bin_strategy": str(response_bin_strategy),
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "perturbation_mode": str(perturbation_mode),
            "noise_scale": float(noise_scale),
        }


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    param_grid: dict,
    config: dict,
    baseline_cache: dict[int, dict],
) -> dict:
    baseline_runs = []
    augmented_runs = []
    for cv_seed in args.cv_seeds:
        if cv_seed not in baseline_cache:
            baseline_cache[cv_seed] = evaluate_tuned_cars_svr_cv(
                X=X,
                y=y,
                args=args,
                param_grid=param_grid,
                augmenter=None,
                random_state=int(cv_seed),
            )
        baseline = baseline_cache[cv_seed]
        augmenter = None
        if config["n_synthetic"] > 0:
            augmenter = ResponseDrivenAugmenter(
                n_synthetic=config["n_synthetic"],
                response_bins=config["response_bins"],
                response_bin_strategy=config["response_bin_strategy"],
                neighbors=config["neighbors"],
                alpha_min=config["alpha_min"],
                alpha_max=config["alpha_max"],
                noise_scale=config["noise_scale"],
                min_derivative_corr=args.min_derivative_corr,
                max_spectral_angle=args.max_spectral_angle,
                envelope_margin=args.envelope_margin,
                random_state=int(cv_seed),
                neighbor_space=config["neighbor_space"],
                spectrum_weight=config["spectrum_weight"],
                perturbation_mode=config["perturbation_mode"],
                response_consistency=args.response_consistency,
            )
        augmented = evaluate_tuned_cars_svr_cv(
            X=X,
            y=y,
            args=args,
            param_grid=param_grid,
            augmenter=augmenter,
            random_state=int(cv_seed),
        )
        baseline_runs.append(baseline)
        augmented_runs.append(augmented)

    summary = summarize_runs(baseline_runs, augmented_runs)
    acceptance_by_seed = [
        float(np.mean([fold["acceptance_rate"] for fold in run["folds"]]))
        if run["folds"]
        else 0.0
        for run in augmented_runs
    ]
    accepted_by_seed = [
        int(round(float(np.mean([fold["synthetic_accepted"] for fold in run["folds"]]))))
        if run["folds"]
        else 0
        for run in augmented_runs
    ]

    return {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "model": "cars-rbf-svr",
        "preprocess": args.preprocess,
        "cv": args.cv,
        "cv_seeds": json.dumps([int(seed) for seed in args.cv_seeds]),
        "inner_cv": args.inner_cv,
        "cars_sampling": args.cars_sampling,
        "cars_min_features": args.cars_min_features,
        "cars_components": args.cars_components,
        "svr_grid_size": len(ParameterGrid(param_grid)),
        "n_synthetic": config["n_synthetic"],
        "neighbor_space": config["neighbor_space"],
        "spectrum_weight": config["spectrum_weight"],
        "neighbors": config["neighbors"],
        "response_bins": config["response_bins"],
        "response_bin_strategy": config["response_bin_strategy"],
        "alpha_min": config["alpha_min"],
        "alpha_max": config["alpha_max"],
        "perturbation_mode": config["perturbation_mode"],
        "noise_scale": config["noise_scale"],
        "min_derivative_corr": args.min_derivative_corr,
        "max_spectral_angle": args.max_spectral_angle,
        "envelope_margin": args.envelope_margin,
        "response_consistency": bool(args.response_consistency),
        **summary,
        "baseline_r2_by_seed": json.dumps([run["r2"] for run in baseline_runs]),
        "augmented_r2_by_seed": json.dumps([run["r2"] for run in augmented_runs]),
        "baseline_rmse_by_seed": json.dumps([run["rmse"] for run in baseline_runs]),
        "augmented_rmse_by_seed": json.dumps([run["rmse"] for run in augmented_runs]),
        "delta_r2_by_seed": json.dumps(
            [aug["r2"] - base["r2"] for base, aug in zip(baseline_runs, augmented_runs)]
        ),
        "delta_rmse_by_seed": json.dumps(
            [aug["rmse"] - base["rmse"] for base, aug in zip(baseline_runs, augmented_runs)]
        ),
        "augmented_acceptance_rate_mean": float(np.mean(acceptance_by_seed)),
        "augmented_acceptance_rate_by_seed": json.dumps(acceptance_by_seed),
        "augmented_synthetic_accepted_by_seed": json.dumps(accepted_by_seed),
    }


def evaluate_tuned_cars_svr_cv(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    param_grid: dict,
    augmenter: ResponseDrivenAugmenter | None,
    random_state: int,
) -> dict:
    X, y, splitter = _validated_cv_inputs(X, y, args.cv, random_state)
    inner = KFold(n_splits=args.inner_cv, shuffle=True, random_state=random_state)
    pred = np.empty_like(y, dtype=float)
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        transformer = CARSFeatureSelector(
            n_sampling=args.cars_sampling,
            min_features=args.cars_min_features,
            max_components=args.cars_components,
            cv=min(args.cv, 5),
            random_state=int(random_state) + fold_idx,
        )
        X_train = transformer.fit_transform(X[train_idx], y[train_idx])
        X_test = transformer.transform(X[test_idx])
        y_train = y[train_idx]

        metadata = {
            "fold": fold_idx,
            "selected_features": int(X_train.shape[1]),
            "synthetic_accepted": 0,
            "acceptance_rate": 0.0,
        }
        if augmenter is not None:
            fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
            result = fold_augmenter.fit_resample(X_train, y_train)
            X_fit, y_fit = result.X, result.y
            metadata["synthetic_accepted"] = result.n_synthetic
            metadata["acceptance_rate"] = float(result.metadata["acceptance_rate"])
        else:
            X_fit, y_fit = X_train, y_train

        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        )
        search.fit(X_fit, y_fit)
        fold_pred = search.predict(X_test).reshape(-1)
        pred[test_idx] = fold_pred
        metadata["rmse"] = float(mean_squared_error(y[test_idx], fold_pred, squared=False))
        metadata["r2"] = float(r2_score(y[test_idx], fold_pred))
        metadata["best_params"] = search.best_params_
        folds.append(metadata)

    return {
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "r2": float(r2_score(y, pred)),
        "folds": folds,
    }


def summarize_runs(baseline_runs: list[dict], augmented_runs: list[dict]) -> dict:
    base_rmse = np.asarray([run["rmse"] for run in baseline_runs], dtype=float)
    aug_rmse = np.asarray([run["rmse"] for run in augmented_runs], dtype=float)
    base_r2 = np.asarray([run["r2"] for run in baseline_runs], dtype=float)
    aug_r2 = np.asarray([run["r2"] for run in augmented_runs], dtype=float)
    base_fold_r2 = np.asarray([fold["r2"] for run in baseline_runs for fold in run["folds"]], dtype=float)
    aug_fold_r2 = np.asarray([fold["r2"] for run in augmented_runs for fold in run["folds"]], dtype=float)

    return {
        "baseline_mean_rmse": float(base_rmse.mean()),
        "baseline_std_rmse": float(base_rmse.std(ddof=1)) if base_rmse.size > 1 else 0.0,
        "baseline_mean_r2": float(base_r2.mean()),
        "baseline_std_r2": float(base_r2.std(ddof=1)) if base_r2.size > 1 else 0.0,
        "baseline_worst_seed_r2": float(base_r2.min()),
        "baseline_worst_fold_r2": float(base_fold_r2.min()),
        "augmented_mean_rmse": float(aug_rmse.mean()),
        "augmented_std_rmse": float(aug_rmse.std(ddof=1)) if aug_rmse.size > 1 else 0.0,
        "augmented_mean_r2": float(aug_r2.mean()),
        "augmented_std_r2": float(aug_r2.std(ddof=1)) if aug_r2.size > 1 else 0.0,
        "augmented_worst_seed_r2": float(aug_r2.min()),
        "augmented_worst_fold_r2": float(aug_fold_r2.min()),
        "delta_mean_rmse": float(aug_rmse.mean() - base_rmse.mean()),
        "delta_mean_r2": float(aug_r2.mean() - base_r2.mean()),
        "delta_std_r2": float(
            (aug_r2.std(ddof=1) if aug_r2.size > 1 else 0.0)
            - (base_r2.std(ddof=1) if base_r2.size > 1 else 0.0)
        ),
        "delta_worst_seed_r2": float(aug_r2.min() - base_r2.min()),
        "delta_worst_fold_r2": float(aug_fold_r2.min() - base_fold_r2.min()),
        "r2_improvement_rate": float(np.mean(aug_r2 > base_r2)),
        "rmse_improvement_rate": float(np.mean(aug_rmse < base_rmse)),
    }


def _parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def config_key(config: dict) -> tuple:
    return (
        int(config["n_synthetic"]),
        str(config["neighbor_space"]),
        float(config["spectrum_weight"]),
        int(config["neighbors"]),
        int(config["response_bins"]),
        str(config["response_bin_strategy"]),
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        str(config["perturbation_mode"]),
        float(config["noise_scale"]),
    )


def _completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            config_key(
                {
                    "n_synthetic": row["n_synthetic"],
                    "neighbor_space": row["neighbor_space"],
                    "spectrum_weight": row["spectrum_weight"],
                    "neighbors": row["neighbors"],
                    "response_bins": row["response_bins"],
                    "response_bin_strategy": row.get("response_bin_strategy", "quantile"),
                    "alpha_min": row["alpha_min"],
                    "alpha_max": row["alpha_max"],
                    "perturbation_mode": row["perturbation_mode"],
                    "noise_scale": row["noise_scale"],
                }
            )
            for row in csv.DictReader(handle)
        }


if __name__ == "__main__":
    main()
