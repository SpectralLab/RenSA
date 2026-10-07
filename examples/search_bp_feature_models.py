from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold, _validated_cv_inputs
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "model",
    "cv",
    "n_synthetic",
    "neighbor_space",
    "perturbation_mode",
    "feature_param",
    "hidden_layers",
    "alpha",
    "learning_rate_init",
    "baseline_rmse",
    "baseline_r2",
    "augmented_rmse",
    "augmented_r2",
    "delta_rmse",
    "delta_r2",
    "selected_features_by_fold",
    "augmented_synthetic_accepted_by_fold",
    "augmented_acceptance_rate_mean",
    "augmented_acceptance_rate_by_fold",
]


class PLSFeatureTransformer(BaseEstimator, TransformerMixin):
    """Use PLS X-scores as supervised low-dimensional BP inputs."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_components = min(int(self.n_components), X.shape[1], X.shape[0] - 1)
        if n_components < 1:
            raise ValueError("PLS needs at least one component.")
        self.model_ = PLSRegression(n_components=n_components)
        self.model_.fit(X, y)
        self.n_components_ = n_components
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, dtype=float))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare PCA-BP, PLS-BP, and CARS-BP with fold-train augmentation."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "bp_feature_models.csv")
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument("--models", nargs="+", default=["pca-bp", "pls-bp", "cars-bp"])
    parser.add_argument("--components", type=int, nargs="+", default=[5, 8, 10, 12, 15])
    parser.add_argument("--cars-sampling", type=int, default=40)
    parser.add_argument("--cars-components", type=int, default=8)
    parser.add_argument("--cars-min-features", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, default=80)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--hidden-layers", nargs="+", default=["16", "32", "32,16"])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--learning-rate-init", type=float, nargs="+", default=[0.001])
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)

    combos = _model_combos(args)
    completed = _completed_keys(args.output) if args.resume else set()
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    dataset = args.dataset_name or args.data.stem

    n_new = 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for model_name, feature_param, hidden_layers, alpha, lr in combos:
            key = (args.preprocess, model_name, feature_param, hidden_layers, alpha, lr)
            if key in completed:
                print(f"skip completed {key}", flush=True)
                continue
            if args.max_runs and n_new >= args.max_runs:
                break
            n_new += 1
            print(
                f"run {n_new}: {model_name} feature={feature_param} hidden={hidden_layers} alpha={alpha}",
                flush=True,
            )
            started = time.perf_counter()
            row = evaluate_combo(
                X=X,
                y=y,
                args=args,
                dataset=dataset,
                model_name=model_name,
                feature_param=feature_param,
                hidden_layers=hidden_layers,
                alpha=alpha,
                learning_rate_init=lr,
            )
            row["run_id"] = int(time.time())
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            writer.writerow(row)
            handle.flush()
            print(
                "  baseline_r2={:.4f} augmented_r2={:.4f} delta_r2={:+.4f}".format(
                    row["baseline_r2"], row["augmented_r2"], row["delta_r2"]
                ),
                flush=True,
            )

    print(f"wrote {args.output}", flush=True)


def evaluate_combo(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    feature_param: int,
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
) -> dict:
    transformer = make_transformer(model_name, feature_param, args)
    estimator = make_bp_regressor(
        hidden_layers=hidden_layers,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=6,
        neighbors=5,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=args.perturbation_mode,
    )

    baseline = evaluate_bp_cv(
        X, y, transformer, estimator, augmenter=None, cv=args.cv, random_state=args.random_state
    )
    augmented = evaluate_bp_cv(
        X, y, transformer, estimator, augmenter=augmenter, cv=args.cv, random_state=args.random_state
    )
    acceptance_rates = [fold["acceptance_rate"] for fold in augmented["folds"]]

    return {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "model": model_name,
        "cv": args.cv,
        "n_synthetic": args.n_synthetic,
        "neighbor_space": args.neighbor_space,
        "perturbation_mode": args.perturbation_mode,
        "feature_param": feature_param,
        "hidden_layers": json.dumps(list(hidden_layers)),
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "augmented_rmse": augmented["rmse"],
        "augmented_r2": augmented["r2"],
        "delta_rmse": augmented["rmse"] - baseline["rmse"],
        "delta_r2": augmented["r2"] - baseline["r2"],
        "selected_features_by_fold": json.dumps(augmented["selected_features_by_fold"]),
        "augmented_synthetic_accepted_by_fold": json.dumps(
            [fold["synthetic_accepted"] for fold in augmented["folds"]]
        ),
        "augmented_acceptance_rate_mean": float(np.mean(acceptance_rates)),
        "augmented_acceptance_rate_by_fold": json.dumps(acceptance_rates),
    }


def evaluate_bp_cv(
    X: np.ndarray,
    y: np.ndarray,
    transformer,
    estimator,
    augmenter: ResponseDrivenAugmenter | None,
    cv: int,
    random_state: int,
) -> dict:
    X, y, splitter = _validated_cv_inputs(X, y, cv, random_state)
    pred = np.empty_like(y, dtype=float)
    selected_features = []
    fold_metadata = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        fold_transformer = clone(transformer)
        X_train = fold_transformer.fit_transform(X[train_idx], y[train_idx])
        X_test = fold_transformer.transform(X[test_idx])
        selected_features.append(int(X_train.shape[1]))
        y_train = y[train_idx]

        if augmenter is not None:
            fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
            result = fold_augmenter.fit_resample(X_train, y_train)
            X_fit, y_fit = result.X, result.y
            fold_metadata.append(
                {
                    "fold": fold_idx,
                    "synthetic_accepted": result.n_synthetic,
                    "acceptance_rate": float(result.metadata["acceptance_rate"]),
                }
            )
        else:
            X_fit, y_fit = X_train, y_train

        model = clone(estimator)
        model.fit(X_fit, y_fit)
        pred[test_idx] = model.predict(X_test).reshape(-1)

    return {
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "r2": float(r2_score(y, pred)),
        "selected_features_by_fold": selected_features,
        "folds": fold_metadata,
    }


def make_transformer(model_name: str, feature_param: int, args: argparse.Namespace):
    if model_name == "pca-bp":
        return PCA(n_components=int(feature_param), random_state=args.random_state)
    if model_name == "pls-bp":
        return PLSFeatureTransformer(n_components=int(feature_param))
    if model_name == "cars-bp":
        return CARSFeatureSelector(
            n_sampling=args.cars_sampling,
            min_features=int(feature_param),
            max_components=args.cars_components,
            cv=min(args.cv, 5),
            random_state=args.random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_bp_regressor(
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    random_state: int,
):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate_init=float(learning_rate_init),
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=80,
        random_state=int(random_state),
    )
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), mlp),
        transformer=StandardScaler(),
    )


def _model_combos(args: argparse.Namespace):
    hidden_layers = [_parse_hidden_layers(value) for value in args.hidden_layers]
    combos = []
    for model_name in args.models:
        if model_name in {"pca-bp", "pls-bp"}:
            feature_params = args.components
        elif model_name == "cars-bp":
            feature_params = args.cars_min_features
        else:
            raise ValueError(f"Unknown model: {model_name}")
        combos.extend(product([model_name], feature_params, hidden_layers, args.alpha, args.learning_rate_init))
    return combos


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                row["preprocess"],
                row["model"],
                int(row["feature_param"]),
                tuple(json.loads(row["hidden_layers"])),
                float(row["alpha"]),
                float(row["learning_rate_init"]),
            )
            for row in reader
        }


if __name__ == "__main__":
    main()
