from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_unified_train_only_protocol import fit_preprocessor
from respond_spectra import CARSFeatureSelector, load_spectrum_csv


FIELDS = [
    "dataset",
    "role",
    "preprocess",
    "train_size",
    "test_count",
    "y_min",
    "y_max",
    "y_range",
    "y_std",
    "unique_y_count",
    "duplicate_y_count",
    "zero_adjacent_y_gap_fraction",
    "median_adjacent_y_gap",
    "median_adjacent_y_gap_over_range",
    "positive_median_adjacent_y_gap",
    "positive_median_adjacent_y_gap_over_range",
    "p90_adjacent_y_gap_over_range",
    "max_adjacent_y_gap_over_range",
    "mean_response_neighbor_spectral_percentile",
    "mean_response_spectrum_neighbor_jaccard",
    "spectral_knn_y_mae_over_std",
    "spectral_knn_y_r2",
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Neighborhood diagnostics for endpoint suitability.")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "publication_coal_diesel")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = [
        Endpoint("coal_Q", Path("data/coal_Q.csv"), args.results_dir / "holdout_unified_trainonly_primary.csv", "primary"),
        Endpoint(
            "coal_ash",
            Path("data/coal_ash.csv"),
            args.results_dir / "holdout_unified_trainonly_coal_ash_raw.csv",
            "supportive coal",
            preprocess="raw",
        ),
        Endpoint(
            "diesel_CN",
            Path("data/diesel_CN.csv"),
            args.results_dir / "holdout_cars_svr_augselect_diesel_cn_snvsg_spxy25_free_gridlow.csv",
            "supportive diesel",
            target_column="CN",
            spectral_prefix="",
            preprocess="snv-sg",
        ),
        Endpoint(
            "diesel_FREEZE",
            Path("data/diesel_FREEZE.csv"),
            args.results_dir / "holdout_cars_svr_augselect_diesel_freeze_snv_spxy25_free.csv",
            "boundary diesel",
            target_column="FREEZE",
            spectral_prefix="",
            preprocess="snv",
        ),
        Endpoint(
            "soil_SOC",
            Path("data/soil_SOC.csv"),
            args.results_dir / "holdout_cars_svr_augselect_soil_snvsg_spxy25_fixed80_singlegrid.csv",
            "exploratory soil",
            preprocess="snv-sg",
        ),
    ]
    rows = [diagnose(endpoint, args) for endpoint in endpoints]
    path = args.output_dir / "table_endpoint_neighborhood_diagnostics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def diagnose(endpoint: Endpoint, args: argparse.Namespace) -> dict:
    _, X, y = load_spectrum_csv(
        endpoint.data,
        target_column=endpoint.target_column,
        spectral_prefix=endpoint.spectral_prefix,
    )
    primary = read_result_row(endpoint.result, endpoint.name)
    train_idx = np.asarray(json.loads(primary["train_indices"]), dtype=int)
    test_idx = np.asarray(json.loads(primary["test_indices"]), dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    preprocess = endpoint.preprocess or primary.get("selected_preprocess") or primary.get("preprocess") or "raw"
    preprocessor = fit_preprocessor(X_train, preprocess)
    X_pre = preprocessor.transform(X_train)

    cars = CARSFeatureSelector(
        n_sampling=int(primary["cars_sampling"]),
        min_features=int(primary["cars_min_features"]),
        max_components=int(primary["cars_components"]),
        cv=min(int(primary["inner_cv"]), 5),
        random_state=args.random_state,
    )
    Z = cars.fit_transform(X_pre, y_train)
    Z = StandardScaler().fit_transform(Z)

    y_range = float(y_train.max() - y_train.min())
    y_std = float(y_train.std(ddof=1))
    gaps = np.diff(np.sort(y_train))
    unique_y = np.unique(y_train)
    duplicate_y_count = int(y_train.size - unique_y.size)
    zero_gap_fraction = float(np.mean(np.isclose(gaps, 0.0))) if gaps.size else 0.0
    positive_gaps = gaps[gaps > 0.0]
    if gaps.size == 0 or y_range <= 1e-12:
        gap_stats = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        positive_median = float(np.median(positive_gaps)) if positive_gaps.size else 0.0
        gap_stats = (
            float(np.median(gaps)),
            float(np.median(gaps) / y_range),
            positive_median,
            float(positive_median / y_range),
            float(np.quantile(gaps, 0.90) / y_range),
            float(np.max(gaps) / y_range),
        )

    k = min(int(args.neighbors), Z.shape[0] - 1)
    spectral_neighbors = neighbor_sets(Z, k)
    response_neighbors = neighbor_sets(y_train[:, None], k)
    distances = pairwise_distances(Z)
    response_percentile = mean_response_neighbor_spectral_percentile(distances, response_neighbors)
    jaccard = mean_jaccard(spectral_neighbors, response_neighbors)
    knn_pred = np.asarray([float(np.mean(y_train[list(neigh)])) for neigh in spectral_neighbors])
    mae_over_std = float(np.mean(np.abs(y_train - knn_pred)) / (y_std + 1e-12))
    spectral_knn_r2 = float(r2_score(y_train, knn_pred))

    return {
        "dataset": endpoint.name,
        "role": endpoint.role,
        "preprocess": preprocess,
        "train_size": int(train_idx.size),
        "test_count": int(test_idx.size),
        "y_min": float(y_train.min()),
        "y_max": float(y_train.max()),
        "y_range": y_range,
        "y_std": y_std,
        "unique_y_count": int(unique_y.size),
        "duplicate_y_count": duplicate_y_count,
        "zero_adjacent_y_gap_fraction": zero_gap_fraction,
        "median_adjacent_y_gap": gap_stats[0],
        "median_adjacent_y_gap_over_range": gap_stats[1],
        "positive_median_adjacent_y_gap": gap_stats[2],
        "positive_median_adjacent_y_gap_over_range": gap_stats[3],
        "p90_adjacent_y_gap_over_range": gap_stats[4],
        "max_adjacent_y_gap_over_range": gap_stats[5],
        "mean_response_neighbor_spectral_percentile": response_percentile,
        "mean_response_spectrum_neighbor_jaccard": jaccard,
        "spectral_knn_y_mae_over_std": mae_over_std,
        "spectral_knn_y_r2": spectral_knn_r2,
    }


def neighbor_sets(values: np.ndarray, k: int) -> list[set[int]]:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(values)
    indices = nbrs.kneighbors(values, return_distance=False)
    return [set(int(v) for v in row[1:]) for row in indices]


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=2)


def mean_response_neighbor_spectral_percentile(distances: np.ndarray, response_neighbors: list[set[int]]) -> float:
    percentiles = []
    n = distances.shape[0]
    denom = max(n - 2, 1)
    for idx, neigh in enumerate(response_neighbors):
        order = np.argsort(distances[idx])
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        for j in neigh:
            percentiles.append((int(ranks[j]) - 1) / denom)
    return float(np.mean(percentiles))


def mean_jaccard(a: list[set[int]], b: list[set[int]]) -> float:
    values = []
    for left, right in zip(a, b):
        values.append(len(left & right) / max(len(left | right), 1))
    return float(np.mean(values))


def read_result_row(path: Path, dataset: str) -> dict:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset") == dataset]
    if not rows:
        raise ValueError(f"no row found for {dataset} in {path}")
    return rows[0]


if __name__ == "__main__":
    main()
