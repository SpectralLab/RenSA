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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold, _validated_cv_inputs
from run_coal_q_augmentation_demo import preprocess_spectra


FIELDNAMES = [
    "run_id",
    "elapsed_sec",
    "dataset",
    "preprocess",
    "mode",
    "cv",
    "n_synthetic",
    "neighbor_space",
    "perturbation_mode",
    "noise_scale",
    "components",
    "hidden_layers",
    "alpha",
    "seeds",
    "ensemble_size",
    "baseline_rmse",
    "baseline_r2",
    "augmented_rmse",
    "augmented_r2",
    "delta_rmse",
    "delta_r2",
    "augmented_synthetic_accepted_by_fold",
    "augmented_acceptance_rate_mean",
    "augmented_acceptance_rate_by_fold",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Strict-CV ensemble PLS-BP and PLS-residual-BP experiments."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", type=Path, default=root / "results" / "ensemble_pls_bp.csv")
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["direct", "residual"],
        choices=["direct", "residual"],
        help="direct: BP predicts y from PLS scores; residual: PLS prediction + BP residual.",
    )
    parser.add_argument("--components", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--hidden-layers", nargs="+", default=["6", "8", "10", "12"])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 137])
    parser.add_argument("--solver", default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu", "logistic"])
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-synthetic", type=int, default=0)
    parser.add_argument("--neighbor-space", default="response", choices=["joint", "spectrum", "response"])
    parser.add_argument("--perturbation-mode", default="local_std", choices=["local_std", "difference", "none"])
    parser.add_argument("--noise-scale", type=float, default=0.008)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column)
    X = preprocess_spectra(X, args.preprocess)
    dataset = args.dataset_name or args.data.stem

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for mode in args.modes:
            started = time.perf_counter()
            print(f"run mode={mode} n_synthetic={args.n_synthetic}", flush=True)
            row = evaluate_mode(X, y, args, dataset, mode)
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


def evaluate_mode(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    dataset: str,
    mode: str,
) -> dict:
    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=6,
        neighbors=5,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        neighbor_space=args.neighbor_space,
        perturbation_mode=args.perturbation_mode,
    )
    baseline = evaluate_ensemble_cv(
        X,
        y,
        args,
        mode=mode,
        augmenter=None,
    )
    augmented = evaluate_ensemble_cv(
        X,
        y,
        args,
        mode=mode,
        augmenter=augmenter if args.n_synthetic > 0 else None,
    )
    acceptance_rates = [fold["acceptance_rate"] for fold in augmented["folds"]]
    return {
        "run_id": "",
        "elapsed_sec": "",
        "dataset": dataset,
        "preprocess": args.preprocess,
        "mode": mode,
        "cv": args.cv,
        "n_synthetic": args.n_synthetic,
        "neighbor_space": args.neighbor_space,
        "perturbation_mode": args.perturbation_mode,
        "noise_scale": args.noise_scale,
        "components": json.dumps(args.components),
        "hidden_layers": json.dumps(args.hidden_layers),
        "alpha": json.dumps(args.alpha),
        "seeds": json.dumps(args.seeds),
        "ensemble_size": len(args.components)
        * len(args.hidden_layers)
        * len(args.alpha)
        * len(args.seeds),
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "augmented_rmse": augmented["rmse"],
        "augmented_r2": augmented["r2"],
        "delta_rmse": augmented["rmse"] - baseline["rmse"],
        "delta_r2": augmented["r2"] - baseline["r2"],
        "augmented_synthetic_accepted_by_fold": json.dumps(
            [fold["synthetic_accepted"] for fold in augmented["folds"]]
        ),
        "augmented_acceptance_rate_mean": float(np.mean(acceptance_rates))
        if acceptance_rates
        else 0.0,
        "augmented_acceptance_rate_by_fold": json.dumps(acceptance_rates),
    }


def evaluate_ensemble_cv(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    mode: str,
    augmenter: ResponseDrivenAugmenter | None,
) -> dict:
    X, y, splitter = _validated_cv_inputs(X, y, args.cv, args.random_state)
    pred = np.empty_like(y, dtype=float)
    fold_metadata = []
    members = list(product(args.components, args.hidden_layers, args.alpha, args.seeds))

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        member_preds = []
        fold_accepts = []
        fold_rates = []
        for n_components, hidden, alpha, seed in members:
            pls = _fit_pls(X[train_idx], y[train_idx], int(n_components))
            T_train = pls.transform(X[train_idx])
            T_test = pls.transform(X[test_idx])
            y_train = y[train_idx]
            if mode == "residual":
                base_train = pls.predict(X[train_idx]).reshape(-1)
                base_test = pls.predict(X[test_idx]).reshape(-1)
                target = y_train - base_train
            else:
                base_test = 0.0
                target = y_train

            if augmenter is not None:
                fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
                result = fold_augmenter.fit_resample(T_train, y_train)
                T_fit = result.X
                if mode == "residual":
                    base_fit = pls.predict(_scores_to_original_space(pls, T_fit)).reshape(-1)
                    y_fit = result.y - base_fit
                else:
                    y_fit = result.y
                fold_accepts.append(result.n_synthetic)
                fold_rates.append(float(result.metadata["acceptance_rate"]))
            else:
                T_fit = T_train
                y_fit = target

            model = make_bp(hidden, alpha, seed, args)
            model.fit(T_fit, y_fit)
            member_preds.append((np.asarray(base_test) + model.predict(T_test)).reshape(-1))

        pred[test_idx] = np.mean(np.vstack(member_preds), axis=0)
        if augmenter is not None:
            fold_metadata.append(
                {
                    "fold": fold_idx,
                    "synthetic_accepted": int(round(np.mean(fold_accepts))),
                    "acceptance_rate": float(np.mean(fold_rates)),
                }
            )

    return {
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "r2": float(r2_score(y, pred)),
        "folds": fold_metadata,
    }


def make_bp(hidden: str, alpha: float, seed: int, args: argparse.Namespace):
    hidden_layers = tuple(int(item) for item in hidden.split(",") if item)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=args.activation,
        solver=args.solver,
        alpha=float(alpha),
        max_iter=int(args.max_iter),
        random_state=int(seed),
        learning_rate_init=0.001,
        early_stopping=False,
    )
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), mlp),
        transformer=StandardScaler(),
    )


def _fit_pls(X: np.ndarray, y: np.ndarray, n_components: int) -> PLSRegression:
    upper = min(n_components, X.shape[1], X.shape[0] - 1)
    model = PLSRegression(n_components=upper)
    model.fit(X, y)
    return model


def _scores_to_original_space(pls: PLSRegression, scores: np.ndarray) -> np.ndarray:
    # Approximate inverse transform for residual targets of synthetic score samples.
    return scores @ pls.x_loadings_.T * pls._x_std + pls._x_mean


if __name__ == "__main__":
    main()
