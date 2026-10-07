from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "train_size",
    "test_count",
    "model",
    "n_synthetic",
    "ensemble_strategy",
    "ensemble_top_k",
    "spxy_y_weight",
    "best_params",
    "inner_cv_rmse",
    "inner_cv_r2",
    "test_rmse",
    "test_r2",
    "baseline_test_rmse",
    "baseline_test_r2",
    "delta_test_rmse",
    "delta_test_r2",
    "bootstrap_delta_rmse_ci_low",
    "bootstrap_delta_rmse_ci_high",
    "bootstrap_delta_r2_ci_low",
    "bootstrap_delta_r2_ci_high",
    "bootstrap_p_rmse_improved",
    "bootstrap_p_r2_improved",
    "synthetic_accepted",
    "acceptance_rate",
    "train_indices",
    "test_indices",
]


PREDICTION_FIELDS = [
    "run_id",
    "dataset",
    "preprocess",
    "split_method",
    "test_size",
    "spxy_y_weight",
    "model",
    "n_synthetic",
    "sample_index",
    "y_true",
    "baseline_pred",
    "augmented_pred",
    "baseline_residual",
    "augmented_residual",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="KS/SPXY holdout evaluation with train-only inner CV and optional augmentation."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "holdout_bp_ks_spxy.csv")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=root / "results" / "holdout_bp_ks_spxy_predictions.csv",
    )
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument("--split-method", choices=["ks", "spxy"], default="spxy")
    parser.add_argument("--test-size", type=float, nargs="+", default=[0.25])
    parser.add_argument("--model", choices=["pca-bp", "ensemble-pls-bp", "ensemble-plsr"], default="pca-bp")
    parser.add_argument(
        "--ensemble-strategy",
        choices=["best", "top-k", "all"],
        default="best",
        help="For ensemble-pls-bp, average only the best hyperparameter set, top-k sets, or all grid sets.",
    )
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument(
        "--spxy-y-weight",
        type=float,
        nargs="+",
        default=[1.0],
        help="Weight applied to autoscaled y when building the SPXY distance space.",
    )
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, default=90)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--components", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    parser.add_argument("--hidden-layers", nargs="+", default=["16", "20"])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--learning-rate-init", type=float, nargs="+", default=[0.001])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 137])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(
        args.data,
        target_column=args.target_column,
        spectral_prefix=args.spectral_prefix,
    )
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    predictions_header = not args.predictions_output.exists() or args.predictions_output.stat().st_size == 0
    with args.output.open("a", newline="") as handle, args.predictions_output.open("a", newline="") as pred_handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        pred_writer = csv.DictWriter(pred_handle, fieldnames=PREDICTION_FIELDS)
        if write_header:
            writer.writeheader()
        if predictions_header:
            pred_writer.writeheader()
        for test_size, spxy_y_weight in product(args.test_size, args.spxy_y_weight):
            train_idx, test_idx = representative_split(
                X,
                y,
                test_size=float(test_size),
                method=args.split_method,
                spxy_y_weight=float(spxy_y_weight),
            )
            started = time.perf_counter()
            print(
                "split={} test_size={} spxy_y_weight={} train={} test={} model={}".format(
                    args.split_method,
                    test_size,
                    spxy_y_weight,
                    train_idx.size,
                    test_idx.size,
                    args.model,
                ),
                flush=True,
            )
            row, prediction_rows = evaluate_holdout(
                X,
                y,
                train_idx,
                test_idx,
                args,
                dataset,
                test_size=float(test_size),
                spxy_y_weight=float(spxy_y_weight),
            )
            row["run_id"] = int(time.time())
            row["elapsed_sec"] = round(time.perf_counter() - started, 3)
            for prediction_row in prediction_rows:
                prediction_row["run_id"] = row["run_id"]
            writer.writerow(row)
            pred_writer.writerows(prediction_rows)
            handle.flush()
            pred_handle.flush()
            print(
                "baseline_test_r2={:.4f} augmented_test_r2={:.4f} delta={:+.4f}".format(
                    row["baseline_test_r2"], row["test_r2"], row["delta_test_r2"]
                ),
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.predictions_output}", flush=True)


def evaluate_holdout(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    test_size: float,
    spxy_y_weight: float,
) -> tuple[dict, list[dict]]:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    param_grid = list(model_param_grid(args))

    baseline = inner_select_and_test(
        X_train,
        y_train,
        X_test,
        y_test,
        args,
        param_grid,
        n_synthetic=0,
    )
    augmented = inner_select_and_test(
        X_train,
        y_train,
        X_test,
        y_test,
        args,
        param_grid,
        n_synthetic=args.n_synthetic,
    )
    stats = bootstrap_deltas(
        y_test,
        baseline["pred"],
        augmented["pred"],
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )
    prediction_rows = []
    for sample_index, y_true, baseline_pred, augmented_pred in zip(
        test_idx,
        y_test,
        baseline["pred"],
        augmented["pred"],
    ):
        prediction_rows.append(
            {
                "run_id": "",
                "dataset": dataset,
                "preprocess": args.preprocess,
                "split_method": args.split_method,
                "test_size": test_size,
                "spxy_y_weight": spxy_y_weight,
                "model": args.model,
                "n_synthetic": args.n_synthetic,
                "sample_index": int(sample_index),
                "y_true": float(y_true),
                "baseline_pred": float(baseline_pred),
                "augmented_pred": float(augmented_pred),
                "baseline_residual": float(y_true - baseline_pred),
                "augmented_residual": float(y_true - augmented_pred),
            }
        )
    row = {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "split_method": args.split_method,
        "test_size": test_size,
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "model": args.model,
        "n_synthetic": args.n_synthetic,
        "ensemble_strategy": args.ensemble_strategy,
        "ensemble_top_k": args.ensemble_top_k,
        "spxy_y_weight": spxy_y_weight,
        "best_params": json.dumps(augmented["best_params"]),
        "inner_cv_rmse": augmented["inner_cv_rmse"],
        "inner_cv_r2": augmented["inner_cv_r2"],
        "test_rmse": augmented["test_rmse"],
        "test_r2": augmented["test_r2"],
        "baseline_test_rmse": baseline["test_rmse"],
        "baseline_test_r2": baseline["test_r2"],
        "delta_test_rmse": augmented["test_rmse"] - baseline["test_rmse"],
        "delta_test_r2": augmented["test_r2"] - baseline["test_r2"],
        **stats,
        "synthetic_accepted": augmented["synthetic_accepted"],
        "acceptance_rate": augmented["acceptance_rate"],
        "train_indices": json.dumps(train_idx.tolist()),
        "test_indices": json.dumps(test_idx.tolist()),
    }
    return row, prediction_rows


def inner_select_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    param_grid: list[dict],
    n_synthetic: int,
) -> dict:
    splitter = KFold(n_splits=min(args.inner_cv, X_train.shape[0]), shuffle=True, random_state=args.random_state)
    scored = []
    for params in param_grid:
        pred = np.empty_like(y_train, dtype=float)
        for fold_idx, (inner_train, inner_valid) in enumerate(splitter.split(X_train)):
            estimator = fit_model(
                X_train[inner_train],
                y_train[inner_train],
                args,
                params,
                n_synthetic=n_synthetic,
                fold_idx=fold_idx,
            )
            pred[inner_valid] = predict_model(estimator, X_train[inner_valid], args.model)
        rmse = float(mean_squared_error(y_train, pred, squared=False))
        r2 = float(r2_score(y_train, pred))
        scored.append({"params": params, "inner_cv_rmse": rmse, "inner_cv_r2": r2})

    scored.sort(key=lambda item: item["inner_cv_rmse"])
    best = scored[0]
    selected_params = select_ensemble_params(scored, args)
    final_params = (
        {"ensemble_members": [item["params"] for item in selected_params]}
        if len(selected_params) > 1
        else best["params"]
    )

    final = fit_model(
        X_train,
        y_train,
        args,
        final_params,
        n_synthetic=n_synthetic,
        fold_idx=0,
    )
    test_pred = predict_model(final, X_test, args.model)
    return {
        "best_params": final_params,
        "inner_cv_rmse": best["inner_cv_rmse"],
        "inner_cv_r2": best["inner_cv_r2"],
        "pred": test_pred,
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "synthetic_accepted": final.get("synthetic_accepted", 0),
        "acceptance_rate": final.get("acceptance_rate", 0.0),
    }


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    params: dict,
    n_synthetic: int,
    fold_idx: int,
) -> dict:
    if "ensemble_members" in params:
        estimators = [
            fit_model(
                X,
                y,
                args,
                member_params,
                n_synthetic=n_synthetic,
                fold_idx=fold_idx,
            )
            for member_params in params["ensemble_members"]
        ]
        accepted = [estimator.get("synthetic_accepted", 0) for estimator in estimators]
        rates = [estimator.get("acceptance_rate", 0.0) for estimator in estimators]
        return {
            "estimators": estimators,
            "synthetic_accepted": int(round(float(np.mean(accepted)))) if accepted else 0,
            "acceptance_rate": float(np.mean(rates)) if rates else 0.0,
        }

    if args.model == "pca-bp":
        transformer = PCA(n_components=int(params["components"]), random_state=args.random_state)
        Z = transformer.fit_transform(X)
        target = y
        base_model = None
    else:
        transformer = PLSRegression(n_components=int(params["components"]))
        Z = transformer.fit_transform(X, y)[0]
        target = y
        base_model = None

    synthetic_accepted = 0
    acceptance_rate = 0.0
    if n_synthetic > 0:
        augmenter = ResponseDrivenAugmenter(
            n_synthetic=n_synthetic,
            response_bins=6,
            neighbors=5,
            noise_scale=args.noise_scale,
            random_state=args.random_state,
            neighbor_space=args.neighbor_space,
            perturbation_mode=args.perturbation_mode,
        )
        result = _clone_augmenter_for_fold(augmenter, fold_idx).fit_resample(Z, y)
        Z_fit = result.X
        target_fit = result.y
        synthetic_accepted = result.n_synthetic
        acceptance_rate = float(result.metadata["acceptance_rate"])
    else:
        Z_fit = Z
        target_fit = target

    if args.model == "ensemble-plsr":
        model = LinearRegression()
        model.fit(Z_fit, target_fit)
        bp_models = [model]
    elif args.model == "pca-bp":
        model = make_bp(params, seed=args.random_state, args=args)
        model.fit(Z_fit, target_fit)
        bp_models = [model]
    else:
        bp_models = []
        for seed in args.seeds:
            model = make_bp(params, seed=seed, args=args)
            model.fit(Z_fit, target_fit)
            bp_models.append(model)

    return {
        "transformer": transformer,
        "base_model": base_model,
        "bp_models": bp_models,
        "synthetic_accepted": synthetic_accepted,
        "acceptance_rate": acceptance_rate,
    }


def predict_model(estimator: dict, X: np.ndarray, model_name: str) -> np.ndarray:
    if "estimators" in estimator:
        preds = [predict_model(member, X, model_name).reshape(-1) for member in estimator["estimators"]]
        return np.mean(np.vstack(preds), axis=0)
    Z = estimator["transformer"].transform(X)
    preds = [model.predict(Z).reshape(-1) for model in estimator["bp_models"]]
    return np.mean(np.vstack(preds), axis=0)


def make_bp(params: dict, seed: int, args: argparse.Namespace):
    hidden = tuple(int(item) for item in str(params["hidden_layers"]).split(",") if item)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="tanh",
        solver="lbfgs",
        alpha=float(params["alpha"]),
        max_iter=int(args.max_iter),
        random_state=int(seed),
    )
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), mlp),
        transformer=StandardScaler(),
    )


def model_param_grid(args: argparse.Namespace):
    if args.model == "ensemble-plsr":
        for components in args.components:
            yield {"components": int(components)}
        return
    for components, hidden, alpha in product(args.components, args.hidden_layers, args.alpha):
        yield {
            "components": int(components),
            "hidden_layers": hidden,
            "alpha": float(alpha),
        }


def select_ensemble_params(scored: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.model not in {"ensemble-pls-bp", "ensemble-plsr"} or args.ensemble_strategy == "best":
        return scored[:1]
    if args.ensemble_strategy == "all":
        return scored
    top_k = min(max(int(args.ensemble_top_k), 1), len(scored))
    return scored[:top_k]


def bootstrap_deltas(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    augmented_pred: np.ndarray,
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
        aug_rmse = mean_squared_error(y_true[idx], augmented_pred[idx], squared=False)
        base_r2 = r2_score(y_true[idx], baseline_pred[idx])
        aug_r2 = r2_score(y_true[idx], augmented_pred[idx])
        delta_rmse.append(float(aug_rmse - base_rmse))
        delta_r2.append(float(aug_r2 - base_r2))
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


def representative_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    method: str,
    spxy_y_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = _autoscale(X)
    if method == "spxy":
        y_scaled = _autoscale(y.reshape(-1, 1))
        space = np.column_stack([X_scaled, float(spxy_y_weight) * y_scaled])
    else:
        space = X_scaled
    n_train = int(round(X.shape[0] * (1.0 - test_size)))
    n_train = min(max(n_train, 2), X.shape[0] - 1)
    train_idx = kennard_stone_indices(space, n_train)
    mask = np.ones(X.shape[0], dtype=bool)
    mask[train_idx] = False
    test_idx = np.flatnonzero(mask)
    return np.asarray(train_idx, dtype=int), test_idx


def kennard_stone_indices(X: np.ndarray, n_select: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    distances = _squared_distances(X)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [int(first), int(second)]
    remaining = np.ones(X.shape[0], dtype=bool)
    remaining[selected] = False
    min_dist = np.minimum(distances[:, first], distances[:, second])
    while len(selected) < n_select:
        candidates = np.flatnonzero(remaining)
        next_idx = int(candidates[np.argmax(min_dist[candidates])])
        selected.append(next_idx)
        remaining[next_idx] = False
        min_dist = np.minimum(min_dist, distances[:, next_idx])
    return np.asarray(selected, dtype=int)


def _squared_distances(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)


def _autoscale(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, ddof=1, keepdims=True), 1e-12)


if __name__ == "__main__":
    main()
