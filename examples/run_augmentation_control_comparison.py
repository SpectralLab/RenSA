from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_unified_train_only_protocol import bootstrap_deltas, fit_preprocessor
from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv
from respond_spectra.evaluation import _clone_augmenter_for_fold


CONTROL_FIELDS = [
    "dataset",
    "endpoint_role",
    "control",
    "control_family",
    "description",
    "preprocess",
    "train_size",
    "test_count",
    "n_synthetic_requested",
    "n_synthetic_accepted",
    "acceptance_rate",
    "test_rmse",
    "test_r2",
    "delta_rmse_vs_no_aug",
    "delta_r2_vs_no_aug",
    "delta_r2_95ci",
    "best_params",
]


@dataclass(frozen=True)
class Endpoint:
    name: str
    data: Path
    result: Path
    role: str
    target_column: str = "y"
    spectral_prefix: str = "x"
    preprocess: str | None = None
    svr_C: tuple[float, ...] = (3000.0, 5000.0, 10000.0)
    svr_gamma: tuple[float, ...] = (0.0015, 0.002, 0.003, 0.005)
    svr_epsilon: tuple[float, ...] = (0.05, 0.1, 0.15)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Control augmentation comparison against the proposed response-neighbor strategy."
    )
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "publication_coal_diesel")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spectral-prefix", default="x")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = [
        Endpoint(
            "coal_Q",
            Path("data/coal_Q.csv"),
            args.results_dir / "holdout_unified_trainonly_primary.csv",
            "primary proof-of-concept",
        ),
        Endpoint(
            "coal_ash",
            Path("data/coal_ash.csv"),
            args.results_dir / "holdout_unified_trainonly_coal_ash_raw.csv",
            "supportive predefined raw sensitivity",
            preprocess="raw",
        ),
        Endpoint(
            "diesel_CN",
            Path("data/diesel_CN.csv"),
            args.results_dir / "holdout_cars_svr_augselect_diesel_cn_snvsg_spxy25_free_gridlow.csv",
            "supportive predefined snv-sg sensitivity",
            target_column="CN",
            spectral_prefix="",
            preprocess="snv-sg",
            svr_C=(1000.0, 3000.0, 5000.0),
            svr_gamma=(0.001, 0.0015, 0.002, 0.003),
        ),
        Endpoint(
            "diesel_FREEZE",
            Path("data/diesel_FREEZE.csv"),
            args.results_dir / "holdout_cars_svr_augselect_diesel_freeze_snv_spxy25_free.csv",
            "boundary diesel endpoint under predefined snv",
            target_column="FREEZE",
            spectral_prefix="",
            preprocess="snv",
        ),
        Endpoint(
            "soil_SOC",
            Path("data/soil_SOC.csv"),
            args.results_dir / "holdout_cars_svr_augselect_soil_snvsg_spxy25_fixed80_singlegrid.csv",
            "external soil endpoint under locked snv-sg protocol",
            preprocess="snv-sg",
            svr_C=(1000.0,),
            svr_gamma=(0.003,),
            svr_epsilon=(0.1,),
        ),
    ]
    rows: list[dict] = []
    for endpoint in endpoints:
        rows.extend(evaluate_endpoint(endpoint, args))

    out = args.output_dir / "table_augmentation_control_comparison.csv"
    write_rows(out, rows, CONTROL_FIELDS)
    print(f"wrote {out}", flush=True)


def evaluate_endpoint(endpoint: Endpoint, args: argparse.Namespace) -> list[dict]:
    _, X, y = load_spectrum_csv(
        endpoint.data,
        target_column=endpoint.target_column,
        spectral_prefix=endpoint.spectral_prefix,
    )
    primary = read_result_row(endpoint.result, endpoint.name)
    train_idx = np.asarray(json.loads(primary["train_indices"]), dtype=int)
    test_idx = np.asarray(json.loads(primary["test_indices"]), dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    preprocess = endpoint.preprocess or primary.get("selected_preprocess") or primary.get("preprocess") or "raw"
    preprocessor = fit_preprocessor(X_train, preprocess)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    cars = CARSFeatureSelector(
        n_sampling=int(primary["cars_sampling"]),
        min_features=int(primary["cars_min_features"]),
        max_components=int(primary["cars_components"]),
        cv=min(int(primary["inner_cv"]), 5),
        random_state=args.random_state,
    )
    Z_train = cars.fit_transform(X_train, y_train)
    Z_test = cars.transform(X_test)

    rows = []
    baseline_pred = None
    baseline_rmse = None
    baseline_r2 = None
    for control, family, description in control_configs():
        Z_fit, y_fit, accepted, rate = resample_control(control, Z_train, y_train, primary, args.random_state)
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid=svr_param_grid(endpoint),
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=min(int(primary["inner_cv"]), Z_fit.shape[0]), shuffle=True, random_state=args.random_state),
            n_jobs=1,
        )
        search.fit(Z_fit, y_fit)
        pred = search.predict(Z_test).reshape(-1)
        rmse = float(mean_squared_error(y_test, pred, squared=False))
        r2 = float(r2_score(y_test, pred))
        if control == "no_augmentation":
            baseline_pred = pred
            baseline_rmse = rmse
            baseline_r2 = r2
        stats = bootstrap_deltas(y_test, baseline_pred, pred, args.bootstrap, args.random_state)
        rows.append(
            {
                "dataset": endpoint.name,
                "endpoint_role": endpoint.role,
                "control": control,
                "control_family": family,
                "description": description,
                "preprocess": preprocess,
                "train_size": int(y_train.size),
                "test_count": int(y_test.size),
                "n_synthetic_requested": 0 if control == "no_augmentation" else int(primary["selected_n_synthetic"]),
                "n_synthetic_accepted": accepted,
                "acceptance_rate": rate,
                "test_rmse": rmse,
                "test_r2": r2,
                "delta_rmse_vs_no_aug": "" if baseline_rmse is None else rmse - baseline_rmse,
                "delta_r2_vs_no_aug": "" if baseline_r2 is None else r2 - baseline_r2,
                "delta_r2_95ci": ci(stats.get("bootstrap_delta_r2_ci_low"), stats.get("bootstrap_delta_r2_ci_high")),
                "best_params": json.dumps(search.best_params_),
            }
        )
        print(f"  {endpoint.name} {control}: test_r2={r2:.4f} accepted={accepted}", flush=True)
    return rows


def control_configs() -> list[tuple[str, str, str]]:
    return [
        ("no_augmentation", "baseline", "Measured training spectra only."),
        ("response_neighbor", "proposed", "Proposed response-neighbor interpolation with the selected synthetic count."),
        ("random_mixup", "mixup", "Random measured-pair interpolation, matching the synthetic count."),
        ("spectrum_neighbor", "spectral VSG-like", "Spectrum-neighbor interpolation, matching the synthetic count."),
        ("noise_only", "spectral VSG-like", "Local spectral jitter around measured spectra with duplicated anchor responses."),
        ("shuffled_response", "negative control", "Response-neighbor interpolation after shuffling training responses."),
        ("smoter_like", "SmoteR-like", "Sparse-response-bin anchor sampling with spectrum-neighbor interpolation."),
        ("smogn_like", "SMOGN-like", "Sparse-response-bin anchor sampling with local Gaussian spectral noise."),
    ]


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
        result = make_augmenter(primary, random_state, neighbor_space="response").fit_resample(Z_train, y_train)
        return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "spectrum_neighbor":
        result = make_augmenter(primary, random_state, neighbor_space="spectrum").fit_resample(Z_train, y_train)
        return result.X, result.y, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "random_mixup":
        return random_pair_resample(Z_train, y_train, n_synthetic, random_state)
    if control == "noise_only":
        return noise_only_resample(Z_train, y_train, n_synthetic, selected_noise(primary), random_state)
    if control == "shuffled_response":
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(y_train)
        result = make_augmenter(primary, random_state, neighbor_space="response").fit_resample(Z_train, shuffled)
        Z_out = np.vstack([Z_train, result.X[result.synthetic_mask]])
        y_out = np.concatenate([y_train, result.y[result.synthetic_mask]])
        return Z_out, y_out, result.n_synthetic, float(result.metadata["acceptance_rate"])
    if control == "smoter_like":
        return smoter_like_resample(Z_train, y_train, n_synthetic, primary, random_state)
    if control == "smogn_like":
        return smogn_like_resample(Z_train, y_train, n_synthetic, selected_noise(primary), random_state)
    raise ValueError(f"unknown control: {control}")


def make_augmenter(primary: dict, random_state: int, neighbor_space: str) -> ResponseDrivenAugmenter:
    return _clone_augmenter_for_fold(
        ResponseDrivenAugmenter(
            n_synthetic=int(primary["selected_n_synthetic"]),
            response_bins=int(primary["response_bins"]),
            response_bin_strategy=primary["response_bin_strategy"],
            neighbors=int(primary["neighbors"]),
            alpha_min=float(primary["alpha_min"]),
            alpha_max=float(primary["alpha_max"]),
            noise_scale=selected_noise(primary),
            random_state=random_state,
            neighbor_space=neighbor_space,
            perturbation_mode=primary["selected_perturbation_mode"],
        ),
        0,
    )


def random_pair_resample(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    first = rng.integers(0, X.shape[0], size=n_synthetic)
    second = rng.integers(0, X.shape[0] - 1, size=n_synthetic)
    second = np.where(second >= first, second + 1, second)
    alpha = rng.uniform(0.15, 0.85, size=n_synthetic)
    X_syn = (1.0 - alpha[:, None]) * X[first] + alpha[:, None] * X[second]
    y_syn = (1.0 - alpha) * y[first] + alpha * y[second]
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn]), int(n_synthetic), 1.0


def noise_only_resample(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    noise_scale: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    anchors = rng.integers(0, X.shape[0], size=n_synthetic)
    scale = X.std(axis=0, ddof=1)
    effective_noise = max(float(noise_scale), 0.008)
    X_syn = X[anchors] + rng.normal(0.0, effective_noise, size=(n_synthetic, X.shape[1])) * scale
    y_syn = y[anchors]
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn]), int(n_synthetic), 1.0


def smoter_like_resample(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    primary: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    bins = response_bins(y, int(primary["response_bins"]), primary["response_bin_strategy"])
    weights = sparse_bin_weights(bins)
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    distances = np.linalg.norm(X_scaled[:, None, :] - X_scaled[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(int(primary["neighbors"]), X.shape[0] - 1)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    anchors = rng.choice(X.shape[0], size=n_synthetic, p=weights)
    partners = np.asarray([rng.choice(neighbors[i]) for i in anchors], dtype=int)
    alpha = rng.uniform(float(primary["alpha_min"]), float(primary["alpha_max"]), size=n_synthetic)
    X_syn = (1.0 - alpha[:, None]) * X[anchors] + alpha[:, None] * X[partners]
    y_syn = (1.0 - alpha) * y[anchors] + alpha * y[partners]
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn]), int(n_synthetic), 1.0


def smogn_like_resample(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    noise_scale: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    bins = response_bins(y, 6, "quantile")
    weights = sparse_bin_weights(bins)
    anchors = rng.choice(X.shape[0], size=n_synthetic, p=weights)
    scale = X.std(axis=0, ddof=1)
    effective_noise = max(float(noise_scale), 0.008)
    X_syn = X[anchors] + rng.normal(0.0, effective_noise, size=(n_synthetic, X.shape[1])) * scale
    y_syn = y[anchors]
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn]), int(n_synthetic), 1.0


def response_bins(y: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if np.allclose(y.min(), y.max()):
        return np.zeros_like(y, dtype=int)
    if strategy == "uniform":
        edges = np.linspace(y.min(), y.max(), int(n_bins) + 1)
    else:
        edges = np.quantile(y, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros_like(y, dtype=int)
    return np.clip(np.digitize(y, edges[1:-1], right=True), 0, edges.size - 2)


def sparse_bin_weights(bins: np.ndarray) -> np.ndarray:
    counts = np.bincount(bins)
    weights = 1.0 / counts[bins]
    return weights / weights.sum()


def selected_noise(primary: dict) -> float:
    return float(primary.get("selected_noise_scale", 0.0) or 0.0)


def svr_param_grid(endpoint: Endpoint) -> dict:
    return {
        "svr__C": list(endpoint.svr_C),
        "svr__gamma": list(endpoint.svr_gamma),
        "svr__epsilon": list(endpoint.svr_epsilon),
    }


def read_result_row(path: Path, dataset: str) -> dict:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset") == dataset]
    if not rows:
        raise ValueError(f"no row found for {dataset} in {path}")
    return rows[0]


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
