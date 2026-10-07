from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_unified_train_only_protocol import bootstrap_deltas
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold


CONTROL_FIELDS = [
    "dataset",
    "control",
    "description",
    "train_size",
    "test_count",
    "n_synthetic_requested",
    "n_synthetic_accepted",
    "acceptance_rate",
    "test_rmse",
    "test_r2",
    "delta_rmse_vs_baseline",
    "delta_r2_vs_baseline",
    "delta_r2_95ci",
    "best_params",
]

PREDICTION_FIELDS = [
    "dataset",
    "control",
    "sample_index",
    "y_true",
    "prediction",
    "residual",
    "abs_residual",
]

RESIDUAL_FIELDS = [
    "dataset",
    "quantile",
    "y_min",
    "y_max",
    "n",
    "baseline_mean_abs_residual",
    "response_neighbor_mean_abs_residual",
    "delta_abs_residual",
]

STABILITY_FIELDS = [
    "dataset",
    "run",
    "random_state",
    "n_features",
    "jaccard_vs_primary",
    "selected_indices",
    "selected_wavelengths_nm",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Coal_Q fixed-control ablations for the train-only protocol."
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "coal_Q.csv")
    parser.add_argument("--primary", type=Path, default=root / "results" / "holdout_unified_trainonly_primary.csv")
    parser.add_argument(
        "--primary-predictions",
        type=Path,
        default=root / "results" / "holdout_unified_trainonly_primary_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "publication_coal_diesel")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--dataset-name", default="coal_Q")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cars-stability-runs", type=int, default=20)
    parser.add_argument(
        "--wavelength-start-nm",
        type=float,
        default=1000.0,
        help="Wavelength assigned to the first spectral variable.",
    )
    parser.add_argument(
        "--wavelength-step-nm",
        type=float,
        default=1.0,
        help="Wavelength increment between adjacent spectral variables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, X, y = load_spectrum_csv(args.data, target_column=args.target_column, spectral_prefix=args.spectral_prefix)

    primary = read_primary_row(args.primary, args.dataset_name)
    train_idx = np.asarray(json.loads(primary["train_indices"]), dtype=int)
    test_idx = np.asarray(json.loads(primary["test_indices"]), dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cars = CARSFeatureSelector(
        n_sampling=int(primary["cars_sampling"]),
        min_features=int(primary["cars_min_features"]),
        max_components=int(primary["cars_components"]),
        cv=min(int(primary["inner_cv"]), 5),
        random_state=args.random_state,
    )
    Z_train = cars.fit_transform(X_train, y_train)
    Z_test = cars.transform(X_test)

    controls = evaluate_controls(Z_train, y_train, Z_test, y_test, test_idx, primary, args)
    residual_rows = residual_quantiles(args.dataset_name, controls, y_test)
    stability_rows, frequency = cars_stability(X_train, y_train, cars.selected_indices_, primary, args)

    write_rows(args.output_dir / "table_coal_q_control_ablation.csv", controls["rows"], CONTROL_FIELDS)
    write_rows(args.output_dir / "table_coal_q_control_predictions.csv", controls["predictions"], PREDICTION_FIELDS)
    write_rows(args.output_dir / "table_coal_q_residual_quantiles.csv", residual_rows, RESIDUAL_FIELDS)
    write_rows(args.output_dir / "table_coal_q_cars_feature_stability.csv", stability_rows, STABILITY_FIELDS)

    print(f"wrote coal_Q mechanism/control artifacts to {args.output_dir}", flush=True)


def evaluate_controls(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    test_idx: np.ndarray,
    primary: dict,
    args: argparse.Namespace,
) -> dict:
    param_grid = {
        "svr__C": [3000.0, 5000.0, 10000.0],
        "svr__gamma": [0.0015, 0.002, 0.003, 0.005],
        "svr__epsilon": [0.05, 0.1, 0.15],
    }
    configs = [
        ("no_augmentation", "Measured training spectra only."),
        ("response_neighbor", "Response-neighbor interpolation selected by inner CV in the main protocol."),
        ("spectrum_neighbor", "Spectrum-neighbor interpolation with the same synthetic count."),
        ("random_pair", "Random measured-pair interpolation without response-neighbor selection."),
        ("shuffled_response", "Response-neighbor interpolation after shuffling training responses."),
        ("noise_only", "Local noise around measured spectra with duplicated anchor responses."),
    ]
    rows = []
    predictions = []
    pred_by_control = {}
    baseline_pred = None
    baseline_rmse = None
    baseline_r2 = None

    for control, description in configs:
        Z_fit, y_fit, accepted, rate = resample_control(control, Z_train, y_train, primary, args.random_state)
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=min(int(primary["inner_cv"]), Z_fit.shape[0]), shuffle=True, random_state=args.random_state),
            n_jobs=1,
        )
        search.fit(Z_fit, y_fit)
        pred = search.predict(Z_test).reshape(-1)
        rmse = float(mean_squared_error(y_test, pred, squared=False))
        r2 = float(r2_score(y_test, pred))
        pred_by_control[control] = pred
        if control == "no_augmentation":
            baseline_pred = pred
            baseline_rmse = rmse
            baseline_r2 = r2
        stats = (
            bootstrap_deltas(y_test, baseline_pred, pred, args.bootstrap, args.random_state)
            if baseline_pred is not None
            else {}
        )
        rows.append(
            {
                "dataset": args.dataset_name,
                "control": control,
                "description": description,
                "train_size": int(y_train.size),
                "test_count": int(y_test.size),
                "n_synthetic_requested": int(primary["selected_n_synthetic"]) if control != "no_augmentation" else 0,
                "n_synthetic_accepted": accepted,
                "acceptance_rate": rate,
                "test_rmse": rmse,
                "test_r2": r2,
                "delta_rmse_vs_baseline": "" if baseline_rmse is None else rmse - baseline_rmse,
                "delta_r2_vs_baseline": "" if baseline_r2 is None else r2 - baseline_r2,
                "delta_r2_95ci": ci(stats.get("bootstrap_delta_r2_ci_low"), stats.get("bootstrap_delta_r2_ci_high")),
                "best_params": json.dumps(search.best_params_),
            }
        )
        for sample_index, y_true, y_pred in zip(test_idx, y_test, pred):
            predictions.append(
                {
                    "dataset": args.dataset_name,
                    "control": control,
                    "sample_index": int(sample_index),
                    "y_true": float(y_true),
                    "prediction": float(y_pred),
                    "residual": float(y_true - y_pred),
                    "abs_residual": float(abs(y_true - y_pred)),
                }
            )
        print(f"  {control}: test_r2={r2:.4f} accepted={accepted}", flush=True)
    return {"rows": rows, "predictions": predictions, "pred_by_control": pred_by_control}


def resample_control(
    control: str,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    primary: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n_synthetic = int(primary["selected_n_synthetic"])
    if control == "no_augmentation":
        return Z_train, y_train, 0, 0.0
    if control == "response_neighbor":
        result = make_response_neighbor(Z_train, y_train, primary, random_state)
        return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "spectrum_neighbor":
        result = make_augmenter(primary, random_state, neighbor_space="spectrum").fit_resample(Z_train, y_train)
        return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "random_pair":
        return random_pair_resample(Z_train, y_train, n_synthetic, random_state)
    if control == "shuffled_response":
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(y_train)
        result = make_augmenter(primary, random_state, neighbor_space="response").fit_resample(Z_train, shuffled)
        Z_out = np.vstack([Z_train, result.X[result.synthetic_mask]])
        y_out = np.concatenate([y_train, result.y[result.synthetic_mask]])
        return Z_out, y_out, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "noise_only":
        return noise_only_resample(Z_train, y_train, n_synthetic, random_state)
    raise ValueError(f"unknown control: {control}")


def make_response_neighbor(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    primary: dict,
    random_state: int,
):
    return make_augmenter(primary, random_state, neighbor_space="response").fit_resample(Z_train, y_train)


def make_augmenter(primary: dict, random_state: int, neighbor_space: str) -> ResponseDrivenAugmenter:
    return _clone_augmenter_for_fold(
        ResponseDrivenAugmenter(
            n_synthetic=int(primary["selected_n_synthetic"]),
            response_bins=int(primary["response_bins"]),
            response_bin_strategy=primary["response_bin_strategy"],
            neighbors=int(primary["neighbors"]),
            alpha_min=float(primary["alpha_min"]),
            alpha_max=float(primary["alpha_max"]),
            noise_scale=float(primary["selected_noise_scale"]),
            random_state=random_state,
            neighbor_space=neighbor_space,
            perturbation_mode=primary["selected_perturbation_mode"],
        ),
        0,
    )


def random_pair_resample(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    n_synthetic: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    first = rng.integers(0, Z_train.shape[0], size=n_synthetic)
    second = rng.integers(0, Z_train.shape[0] - 1, size=n_synthetic)
    second = np.where(second >= first, second + 1, second)
    alpha = rng.uniform(0.15, 0.85, size=n_synthetic)
    Z_syn = (1.0 - alpha[:, None]) * Z_train[first] + alpha[:, None] * Z_train[second]
    y_syn = (1.0 - alpha) * y_train[first] + alpha * y_train[second]
    return np.vstack([Z_train, Z_syn]), np.concatenate([y_train, y_syn]), int(n_synthetic), 1.0


def noise_only_resample(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    n_synthetic: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    anchors = rng.integers(0, Z_train.shape[0], size=n_synthetic)
    scale = Z_train.std(axis=0, ddof=1)
    Z_syn = Z_train[anchors] + rng.normal(0.0, 0.008, size=(n_synthetic, Z_train.shape[1])) * scale
    y_syn = y_train[anchors]
    return np.vstack([Z_train, Z_syn]), np.concatenate([y_train, y_syn]), int(n_synthetic), 1.0


def residual_quantiles(dataset: str, controls: dict, y_test: np.ndarray) -> list[dict]:
    baseline = controls["pred_by_control"]["no_augmentation"]
    response = controls["pred_by_control"]["response_neighbor"]
    edges = np.quantile(y_test, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for idx in range(4):
        if idx == 3:
            mask = (y_test >= edges[idx]) & (y_test <= edges[idx + 1])
        else:
            mask = (y_test >= edges[idx]) & (y_test < edges[idx + 1])
        base_abs = np.abs(y_test[mask] - baseline[mask])
        response_abs = np.abs(y_test[mask] - response[mask])
        rows.append(
            {
                "dataset": dataset,
                "quantile": f"Q{idx + 1}",
                "y_min": float(y_test[mask].min()),
                "y_max": float(y_test[mask].max()),
                "n": int(mask.sum()),
                "baseline_mean_abs_residual": float(base_abs.mean()),
                "response_neighbor_mean_abs_residual": float(response_abs.mean()),
                "delta_abs_residual": float(response_abs.mean() - base_abs.mean()),
            }
        )
    return rows


def cars_stability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    primary_indices: np.ndarray,
    primary: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], np.ndarray]:
    rows = []
    counts = np.zeros(X_train.shape[1], dtype=int)
    primary_set = set(int(i) for i in primary_indices)
    for run in range(int(args.cars_stability_runs)):
        seed = int(args.random_state + run)
        selector = CARSFeatureSelector(
            n_sampling=int(primary["cars_sampling"]),
            min_features=int(primary["cars_min_features"]),
            max_components=int(primary["cars_components"]),
            cv=min(int(primary["inner_cv"]), 5),
            random_state=seed,
        )
        selector.fit(X_train, y_train)
        selected = np.asarray(selector.selected_indices_, dtype=int)
        counts[selected] += 1
        selected_set = set(int(i) for i in selected)
        selected_wavelengths = wavelength_axis(X_train.shape[1], args)[selected]
        rows.append(
            {
                "dataset": args.dataset_name,
                "run": run + 1,
                "random_state": seed,
                "n_features": int(selected.size),
                "jaccard_vs_primary": len(primary_set & selected_set) / max(len(primary_set | selected_set), 1),
                "selected_indices": json.dumps(selected.tolist()),
                "selected_wavelengths_nm": json.dumps([round(float(value), 6) for value in selected_wavelengths]),
            }
        )
    return rows, counts / max(int(args.cars_stability_runs), 1)








def read_primary_row(path: Path, dataset: str) -> dict:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    if not rows:
        raise ValueError(f"no primary row found for {dataset} in {path}")
    return rows[0]




def wavelength_axis(n_features: int, args: argparse.Namespace) -> np.ndarray:
    return float(args.wavelength_start_nm) + float(args.wavelength_step_nm) * np.arange(n_features, dtype=float)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ci(low, high) -> str:
    if low in {"", None} or high in {"", None}:
        return ""
    return f"[{float(low):.4f}, {float(high):.4f}]"


if __name__ == "__main__":
    main()
