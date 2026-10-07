"""Isolated RandomPair + same RenSA filter -> measured-only CARS -> RBF-SVR control experiment.

Only the second-parent pairing rule is changed from response-neighbor pairing to
global random pairing. Anchor weighting, interpolation, local perturbation,
consistency filtering, preprocessing, measured-only CARS, SVR tuning, data
splitting, seeds, evaluation, checkpointing, and outputs follow the original
full-spectrum RenSA experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
RESULTS_DIR = EXPERIMENT_ROOT / "results_randompair_same_filter"
LOGS_DIR = EXPERIMENT_ROOT / "logs_randompair_same_filter"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
sys.path.insert(0, str(PROJECT_ROOT))

from respond_spectra import CARSFeatureSelector, ResponseDrivenAugmenter, load_spectrum_csv  # noqa: E402
from respond_spectra.augment import AugmentationResult, _AugmentationDiagnostics  # noqa: E402
from respond_spectra.preprocessing import msc, savgol, snv  # noqa: E402


ALGORITHM_SEED = 42
TEST_SIZE = 0.25
INNER_CV = 5
PREPROCESSING = ("raw", "snv", "sg", "snv-sg", "msc", "msc-sg")
N_SYNTHETIC = (40, 60, 80)
PERTURBATIONS = ("none", "local_std")
NOISE_SCALES = (0.0, 0.008)
SVR_C = (3000.0, 5000.0, 10000.0)
SVR_GAMMA = (0.0015, 0.002, 0.003, 0.005)
SVR_EPSILON = (0.05, 0.10, 0.15)

MAX_SPECTRAL_ANGLE = 0.18
MIN_DERIVATIVE_CORR = 0.70
ENVELOPE_MARGIN = 0.08


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    target: str
    loader: str = "xprefix"


DATASETS = {
    "coal_Q": DatasetSpec("coal_Q.csv", "y"),
    "coal_ash": DatasetSpec("coal_ash.csv", "y"),
    "soil_SOC": DatasetSpec("soil_SOC.csv", "y"),
    "diesel_CN": DatasetSpec("diesel_CN.csv", "CN", "diesel"),
    "diesel_FREEZE": DatasetSpec("diesel_FREEZE.csv", "FREEZE", "diesel"),
}


@dataclass(frozen=True)
class RenSAConfig:
    n_synthetic: int
    perturbation: str
    noise_scale: float

    @property
    def effective_key(self) -> tuple[int, str, float]:
        if self.perturbation == "none" or self.noise_scale <= 0:
            return self.n_synthetic, "none", 0.0
        return self.n_synthetic, self.perturbation, self.noise_scale


@dataclass(frozen=True)
class Candidate:
    preprocessing: str
    rensa: RenSAConfig


@dataclass(frozen=True)
class Preprocessor:
    method: str
    msc_reference: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        window = min(21, X.shape[1] if X.shape[1] % 2 else X.shape[1] - 1)
        if self.method == "raw":
            return X.copy()
        if self.method == "snv":
            return snv(X)
        if self.method == "sg":
            return savgol(X, window_length=window, polyorder=2)
        if self.method == "snv-sg":
            return savgol(snv(X), window_length=window, polyorder=2)
        if self.method == "msc":
            return msc(X, reference=self.msc_reference)
        if self.method == "msc-sg":
            return savgol(msc(X, reference=self.msc_reference), window_length=window, polyorder=2)
        raise ValueError(f"Unknown preprocessing method: {self.method}")


@dataclass(frozen=True)
class FoldArtifact:
    valid_indices: np.ndarray
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    requested_synthetic: int
    accepted_synthetic: int
    acceptance_rate: float


@dataclass
class HoldoutGuard:
    """Prevent outer holdout values from being exposed before final prediction."""

    X_source: np.ndarray
    y_source: np.ndarray
    indices: np.ndarray
    revealed: bool = False

    def reveal(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        if stage != "final_prediction":
            raise RuntimeError("Outer holdout may only be accessed at final_prediction stage.")
        if self.revealed:
            raise RuntimeError("Outer holdout was already revealed.")
        self.revealed = True
        return self.X_source[self.indices].copy(), self.y_source[self.indices].copy()


class AuditedFullSpectrumAugmenter(ResponseDrivenAugmenter):
    """RenSA wrapper that fails if generation or filtering sees reduced spectra."""

    def __init__(self, *, expected_n_features: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.expected_n_features = int(expected_n_features)
        self.consistency_calls_ = 0

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2
        assert X.shape[1] == self.expected_n_features, (
            "RenSA input is not the complete preprocessed spectrum."
        )
        result = super().fit_resample(X, y)
        assert result.X.shape[1] == self.expected_n_features
        assert self.consistency_calls_ > 0, "Full-spectrum consistency filtering was not executed."
        return result

    def _consistency_report(
        self,
        x_new: np.ndarray,
        base_x: np.ndarray,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        base_y: float,
        X: np.ndarray,
        y: np.ndarray,
    ):
        vectors = (x_new, base_x, parent_a, parent_b)
        assert all(np.asarray(vector).shape == (self.expected_n_features,) for vector in vectors)
        assert np.asarray(X).shape[1] == self.expected_n_features, (
            "Consistency filtering input is not the complete continuous spectrum."
        )
        self.consistency_calls_ += 1
        return super()._consistency_report(x_new, base_x, parent_a, parent_b, base_y, X, y)


class AuditedRandomPairSameFilterAugmenter(AuditedFullSpectrumAugmenter):
    """Random-pair control with every non-pairing RenSA operation preserved.

    This class intentionally changes exactly one mechanism in
    :class:`ResponseDrivenAugmenter.fit_resample`: after selecting anchor ``i``
    with the original sparse-response anchor weights, parent ``j`` is sampled
    uniformly from all other measured samples instead of from ``i``'s response
    neighbors.

    Crucially, ``local_candidates`` are still computed by the inherited
    ``_local_neighbors`` method and are passed unchanged to
    ``_local_perturbation``. Therefore the local perturbation definition is
    identical to the RenSA control; only the interpolation partner is random.
    The inherited ``_consistency_report`` is also used unchanged, so the
    spectral-angle, derivative-correlation, and envelope filters and their
    thresholds remain exactly the same.
    """

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> AugmentationResult:
        X, y = self._validate_inputs(X, y)
        assert X.ndim == 2
        assert X.shape[1] == self.expected_n_features, (
            "RandomPair+Filter input is not the complete preprocessed spectrum."
        )

        if self.n_synthetic <= 0:
            return AugmentationResult(
                X=X.copy(),
                y=y.copy(),
                synthetic_mask=np.zeros(X.shape[0], dtype=bool),
                metadata={
                    "accepted": 0,
                    "attempted": 0,
                    "acceptance_rate": 0.0,
                    "requested": int(self.n_synthetic),
                    "neighbor_space": self.neighbor_space,
                    "perturbation_mode": self.perturbation_mode,
                    "pairing_rule": "random_global",
                },
            )

        rng = np.random.default_rng(self.random_state)

        # Identical to RenSA: anchors are preferentially sampled from sparse
        # response bins using the same binning strategy and anchor weights.
        bins = self._response_bins(y)
        anchor_weights = self._anchor_weights(bins)

        # Identical to RenSA and deliberately retained ONLY for local
        # perturbation. These response-neighbor candidates are not used for
        # selecting parent j in this control.
        local_neighbor_indices = self._local_neighbors(X, y)
        all_indices = np.arange(X.shape[0], dtype=int)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[float] = []
        diagnostics = _AugmentationDiagnostics()
        attempted = 0
        max_attempts = max(self.n_synthetic * self.max_attempt_multiplier, self.n_synthetic)

        while len(synthetic_X) < self.n_synthetic and attempted < max_attempts:
            attempted += 1

            # Same anchor selection as RenSA.
            i = int(rng.choice(X.shape[0], p=anchor_weights))

            # Keep the original response-neighbor set for local perturbation.
            local_candidates = local_neighbor_indices[i]
            if local_candidates.size == 0:
                diagnostics.rejections["no_neighbor"] += 1
                continue

            # THE ONLY MECHANISTIC CHANGE:
            # RenSA: j = rng.choice(local_candidates)
            # Control: j is uniformly sampled from every other measured sample.
            random_pair_candidates = all_indices[all_indices != i]
            if random_pair_candidates.size == 0:
                diagnostics.rejections["no_random_pair"] += 1
                continue
            j = int(rng.choice(random_pair_candidates))

            # Identical interpolation rule and alpha range.
            alpha = float(rng.uniform(self.alpha_min, self.alpha_max))
            base_x = (1.0 - alpha) * X[i] + alpha * X[j]
            base_y = (1.0 - alpha) * y[i] + alpha * y[j]

            # Identical local perturbation definition. Importantly, the local
            # response-neighbor set of anchor i is preserved here rather than
            # being replaced by the global random-pair candidate set.
            x_new = base_x + self._local_perturbation(X, local_candidates, rng)

            # Identical RenSA consistency filtering and thresholds.
            check = self._consistency_report(
                x_new,
                base_x,
                X[i],
                X[j],
                base_y,
                X,
                y,
            )
            diagnostics.observe_attempt(y[i], y[j], base_y, check)

            if check.passed:
                synthetic_X.append(x_new)
                synthetic_y.append(base_y)
                diagnostics.observe_accept(base_y, check)
            else:
                diagnostics.rejections[check.reason] += 1

        if synthetic_X:
            X_syn = np.vstack(synthetic_X)
            y_syn = np.asarray(synthetic_y, dtype=float)
            X_out = np.vstack([X, X_syn])
            y_out = np.concatenate([y, y_syn])
            synthetic_mask = np.concatenate(
                [np.zeros(X.shape[0], dtype=bool), np.ones(X_syn.shape[0], dtype=bool)]
            )
        else:
            X_out = X.copy()
            y_out = y.copy()
            synthetic_mask = np.zeros(X.shape[0], dtype=bool)

        accepted = int(len(synthetic_X))
        metadata = {
            "accepted": accepted,
            "attempted": attempted,
            "acceptance_rate": accepted / attempted if attempted else 0.0,
            "requested": int(self.n_synthetic),
            # Kept because it still defines the local perturbation neighborhood.
            "neighbor_space": self.neighbor_space,
            "perturbation_mode": self.perturbation_mode,
            "pairing_rule": "random_global",
            **diagnostics.as_metadata(),
        }
        result = AugmentationResult(
            X=X_out,
            y=y_out,
            synthetic_mask=synthetic_mask,
            metadata=metadata,
        )

        # Preserve the full-spectrum audit guarantees of the original wrapper.
        assert result.X.shape[1] == self.expected_n_features
        assert self.consistency_calls_ > 0, "Full-spectrum consistency filtering was not executed."
        return result


class AuditedMeasuredOnlyCARS(CARSFeatureSelector):
    """CARS wrapper that records and checks the measured-only fitting boundary."""

    def __init__(self, *, expected_measured_count: int, forbidden_synthetic_count: int):
        super().__init__(
            n_sampling=40,
            min_features=40,
            max_components=8,
            cv=INNER_CV,
            random_state=ALGORITHM_SEED,
        )
        self.expected_measured_count = int(expected_measured_count)
        self.forbidden_synthetic_count = int(forbidden_synthetic_count)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert X.shape[0] == self.expected_measured_count
        assert y.shape[0] == self.expected_measured_count
        if self.forbidden_synthetic_count > 0:
            assert X.shape[0] != self.expected_measured_count + self.forbidden_synthetic_count
        self.cars_fit_n_samples_ = int(X.shape[0])
        self.cars_fit_source_ = "measured_only"
        fitted = super().fit(X, y)
        assert self.cars_fit_n_samples_ == self.expected_measured_count
        assert self.cars_fit_source_ == "measured_only"
        return fitted


SPLIT_RESULT_FIELDS = [
    "task",
    "split_method",
    "outer_seed",
    "actual_strata",
    "algorithm_seed",
    "n_calibration",
    "n_holdout",
    "n_full_wavelengths",
    "n_cars_features",
    "selected_cars_indices",
    "preprocessing",
    "best_n_syn",
    "best_perturbation",
    "best_noise_scale",
    "requested_synthetic",
    "accepted_synthetic",
    "acceptance_rate",
    "best_C",
    "best_gamma",
    "best_epsilon",
    "inner_cv_RMSEP",
    "inner_cv_R2",
    "R2",
    "RMSEP",
    "runtime_seconds",
]

PREDICTION_FIELDS = [
    "task",
    "split_method",
    "outer_seed",
    "sample_index",
    "y_true",
    "y_pred",
    "residual",
]

Y_DISTRIBUTION_FIELDS = [
    "task",
    "split_method",
    "outer_seed",
    "subset",
    "n",
    "min",
    "Q1",
    "median",
    "mean",
    "Q3",
    "max",
    "SD",
]

SPARSITY_FIELDS = [
    "task",
    "split_method",
    "outer_seed",
    "sparsity_median_5nn",
    "sparsity_p90_5nn",
    "sparsity_max_5nn",
]

SUMMARY_FIELDS = [
    "task",
    "spxy_R2",
    "spxy_RMSEP",
    "ks_R2",
    "ks_RMSEP",
    "mc_n",
    "mc_mean_R2",
    "mc_SD_R2",
    "mc_median_R2",
    "mc_mean_RMSEP",
    "mc_SD_RMSEP",
    "mc_median_RMSEP",
    "mc_min_RMSEP",
    "mc_max_RMSEP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomPair + same RenSA filter -> measured-only CARS -> RBF-SVR control experiments."
    )
    parser.add_argument("--tasks", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--split-methods",
        nargs="+",
        choices=("spxy", "ks", "mc"),
        default=["spxy", "ks", "mc"],
    )
    parser.add_argument("--mc-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run only coal_Q + deterministic SPXY.")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional limit for controlled execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    if args.smoke_test:
        args.tasks = ["coal_Q"]
        args.split_methods = ["spxy"]
        args.mc_seeds = []
    invalid_mc_seeds = sorted(set(args.mc_seeds) - {0, 1, 2, 3, 4})
    if invalid_mc_seeds:
        raise ValueError(f"MC outer seeds must be selected from 0..4, got {invalid_mc_seeds}.")
    assert ALGORITHM_SEED == 42

    completed = completed_keys()
    new_runs = 0
    for task, split_method, outer_seed in requested_runs(args):
        key = experiment_key(task, split_method, outer_seed)
        if key in completed:
            if args.resume:
                logging.info("skip completed task=%s split=%s outer_seed=%s", task, split_method, seed_text(outer_seed))
                continue
            raise FileExistsError(
                f"Experiment {key} already exists. Use --resume to preserve and skip completed results."
            )
        if args.max_runs > 0 and new_runs >= args.max_runs:
            break
        try:
            run_one(task, split_method, outer_seed)
        except Exception:
            logging.exception(
                "failed task=%s split=%s outer_seed=%s",
                task,
                split_method,
                seed_text(outer_seed),
            )
            raise
        completed.add(key)
        new_runs += 1
        rebuild_aggregate_outputs()

    rebuild_aggregate_outputs()
    logging.info("finished new_runs=%d completed_checkpoints=%d", new_runs, len(completed_keys()))


def requested_runs(args: argparse.Namespace) -> Iterable[tuple[str, str, int | None]]:
    for task in args.tasks:
        for method in args.split_methods:
            if method == "mc":
                for seed in args.mc_seeds:
                    yield task, method, int(seed)
            else:
                yield task, method, None


def run_one(task: str, split_method: str, outer_seed: int | None) -> None:
    started = time.perf_counter()
    _, X, y = load_task(task)
    n_full_wavelengths = int(X.shape[1])
    calibration_idx, holdout_idx, actual_strata = outer_split(X, y, split_method, outer_seed)
    assert_disjoint_complete_indices(calibration_idx, holdout_idx, X.shape[0])
    if split_method == "mc":
        assert outer_seed in {0, 1, 2, 3, 4}
    else:
        assert outer_seed is None

    X_calibration = X[calibration_idx].copy()
    y_calibration = y[calibration_idx].copy()
    holdout = HoldoutGuard(X, y, holdout_idx)
    logging.info(
        "start task=%s split=%s outer_seed=%s calibration=%d holdout=%d full_dimension=%d actual_strata=%s",
        task,
        split_method,
        seed_text(outer_seed),
        calibration_idx.size,
        holdout_idx.size,
        n_full_wavelengths,
        actual_strata if actual_strata is not None else "NA",
    )

    best, inner_metrics = select_by_inner_cv(X_calibration, y_calibration, n_full_wavelengths)
    final = fit_final_and_predict(
        X_calibration,
        y_calibration,
        holdout,
        best,
        n_full_wavelengths,
    )
    assert holdout.revealed, "Outer holdout was not restricted to final prediction."
    X_holdout, y_holdout = final.pop("holdout_values")
    prediction = np.asarray(final.pop("prediction"), dtype=float)
    elapsed = time.perf_counter() - started

    row = {
        "task": task,
        "split_method": split_method,
        "outer_seed": seed_value(outer_seed),
        "actual_strata": actual_strata if actual_strata is not None else "",
        "algorithm_seed": ALGORITHM_SEED,
        "n_calibration": int(calibration_idx.size),
        "n_holdout": int(holdout_idx.size),
        "n_full_wavelengths": n_full_wavelengths,
        "n_cars_features": final["n_cars_features"],
        "selected_cars_indices": json.dumps(final["selected_cars_indices"], separators=(",", ":")),
        "preprocessing": best.preprocessing,
        "best_n_syn": best.rensa.n_synthetic,
        "best_perturbation": best.rensa.perturbation,
        "best_noise_scale": best.rensa.noise_scale,
        "requested_synthetic": final["requested_synthetic"],
        "accepted_synthetic": final["accepted_synthetic"],
        "acceptance_rate": final["acceptance_rate"],
        "best_C": final["best_C"],
        "best_gamma": final["best_gamma"],
        "best_epsilon": final["best_epsilon"],
        "inner_cv_RMSEP": inner_metrics["RMSEP"],
        "inner_cv_R2": inner_metrics["R2"],
        "R2": float(r2_score(y_holdout, prediction)),
        "RMSEP": float(np.sqrt(mean_squared_error(y_holdout, prediction))),
        "runtime_seconds": float(elapsed),
    }
    predictions = [
        {
            "task": task,
            "split_method": split_method,
            "outer_seed": seed_value(outer_seed),
            "sample_index": int(sample_idx),
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "residual": float(y_true - y_pred),
        }
        for sample_idx, y_true, y_pred in zip(holdout_idx, y_holdout, prediction)
    ]
    y_stats = [
        distribution_row(task, split_method, outer_seed, "calibration", y_calibration),
        distribution_row(task, split_method, outer_seed, "holdout", y_holdout),
    ]
    sparsity = response_sparsity_row(task, split_method, outer_seed, y_calibration)
    save_checkpoint(task, split_method, outer_seed, row, predictions, y_stats, sparsity)
    logging.info(
        "done task=%s split=%s outer_seed=%s calibration=%d holdout=%d full_dimension=%d "
        "cars_features=%d randompair_filter=n%d/%s/noise%g requested=%d accepted=%d R2=%.6f RMSEP=%.6f elapsed=%.2fs",
        task,
        split_method,
        seed_text(outer_seed),
        calibration_idx.size,
        holdout_idx.size,
        n_full_wavelengths,
        row["n_cars_features"],
        best.rensa.n_synthetic,
        best.rensa.perturbation,
        best.rensa.noise_scale,
        row["requested_synthetic"],
        row["accepted_synthetic"],
        row["R2"],
        row["RMSEP"],
        elapsed,
    )
    del X_holdout


def select_by_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_full_wavelengths: int,
) -> tuple[Candidate, dict[str, float]]:
    assert X.shape[1] == n_full_wavelengths
    splitter = KFold(n_splits=min(INNER_CV, X.shape[0]), shuffle=True, random_state=ALGORITHM_SEED)
    folds = list(splitter.split(X))
    validate_inner_folds(folds, X.shape[0])
    cache = build_inner_fold_cache(X, y, folds, n_full_wavelengths)

    scored: list[tuple[Candidate, dict[str, float]]] = []
    for preprocessing in PREPROCESSING:
        for rensa in effective_rensa_configs():
            artifacts = [cache[(fold_idx, preprocessing, rensa.effective_key)] for fold_idx in range(len(folds))]
            prediction = np.empty_like(y, dtype=float)
            for artifact in artifacts:
                search = make_svr_search(artifact.X_fit.shape[0])
                search.fit(artifact.X_fit, artifact.y_fit)
                fold_prediction = search.predict(artifact.X_valid).reshape(-1)
                assert fold_prediction.shape[0] == artifact.y_valid.shape[0]
                prediction[artifact.valid_indices] = fold_prediction
            metrics = {
                "RMSEP": float(np.sqrt(mean_squared_error(y, prediction))),
                "R2": float(r2_score(y, prediction)),
            }
            scored.append((Candidate(preprocessing, rensa), metrics))
            logging.info(
                "inner scored preprocessing=%s randompair_filter=n%d/%s/noise%g",
                preprocessing,
                rensa.n_synthetic,
                rensa.perturbation,
                rensa.noise_scale,
            )
    if not scored:
        raise RuntimeError("Inner-CV produced no valid pipeline candidate.")
    scored.sort(
        key=lambda item: (
            item[1]["RMSEP"],
            item[0].preprocessing,
            item[0].rensa.effective_key,
        )
    )
    return scored[0]


def build_inner_fold_cache(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_full_wavelengths: int,
) -> dict[tuple[int, str, tuple[int, str, float]], FoldArtifact]:
    cache: dict[tuple[int, str, tuple[int, str, float]], FoldArtifact] = {}
    for fold_idx, (fit_idx, valid_idx) in enumerate(folds):
        assert np.intersect1d(fit_idx, valid_idx).size == 0
        X_measured_fit_raw = X[fit_idx]
        y_measured_fit = y[fit_idx]
        X_measured_valid_raw = X[valid_idx]
        y_measured_valid = y[valid_idx]
        for preprocessing in PREPROCESSING:
            preprocessor = fit_preprocessor_measured_only(
                X_measured_fit_raw,
                preprocessing,
                expected_measured_count=fit_idx.size,
            )
            X_measured_fit_full = preprocessor.transform(X_measured_fit_raw)
            X_measured_valid_full = preprocessor.transform(X_measured_valid_raw)
            assert X_measured_fit_full.shape[1] == n_full_wavelengths
            assert X_measured_valid_full.shape == (valid_idx.size, n_full_wavelengths)

            # All RandomPair+Filter candidates are generated and filtered in full continuous
            # spectral space before CARS is fitted on measured samples only.
            augmented_by_config = {
                config.effective_key: augment_full_spectra(
                    X_measured_fit_full,
                    y_measured_fit,
                    config,
                    n_full_wavelengths,
                )
                for config in effective_rensa_configs()
            }
            max_synthetic = max(result.n_synthetic for result in augmented_by_config.values())
            selector = fit_cars_measured_only(
                X_measured_fit_full,
                y_measured_fit,
                expected_measured_count=fit_idx.size,
                forbidden_synthetic_count=max_synthetic,
            )
            selected_indices = np.asarray(selector.selected_indices_, dtype=int)

            for config in effective_rensa_configs():
                result = augmented_by_config[config.effective_key]
                X_synthetic_full = result.X[result.synthetic_mask]
                y_synthetic = result.y[result.synthetic_mask]
                Z_measured, Z_synthetic, Z_valid = apply_fixed_cars_mask(
                    selected_indices,
                    X_measured_fit_full,
                    X_synthetic_full,
                    X_measured_valid_full,
                    n_full_wavelengths,
                )
                X_fit = np.vstack([Z_measured, Z_synthetic])
                y_fit = np.concatenate([y_measured_fit, y_synthetic])
                assert X_fit.shape[0] == fit_idx.size + result.n_synthetic
                assert y_fit.shape[0] == X_fit.shape[0]
                assert Z_valid.shape[0] == valid_idx.size, "Inner validation must remain measured-only."
                cache[(fold_idx, preprocessing, config.effective_key)] = FoldArtifact(
                    valid_indices=valid_idx.copy(),
                    X_fit=X_fit,
                    y_fit=y_fit,
                    X_valid=Z_valid,
                    y_valid=y_measured_valid.copy(),
                    requested_synthetic=config.n_synthetic,
                    accepted_synthetic=result.n_synthetic,
                    acceptance_rate=float(result.metadata["acceptance_rate"]),
                )
        logging.info("built strict full-spectrum RandomPair+Filter cache for inner fold %d/%d", fold_idx + 1, len(folds))
    return cache


def fit_final_and_predict(
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    holdout: HoldoutGuard,
    candidate: Candidate,
    n_full_wavelengths: int,
) -> dict[str, Any]:
    preprocessor = fit_preprocessor_measured_only(
        X_calibration,
        candidate.preprocessing,
        expected_measured_count=X_calibration.shape[0],
    )
    X_measured_full = preprocessor.transform(X_calibration)
    assert X_measured_full.shape[1] == n_full_wavelengths

    # Required order: full-spectrum RandomPair+Filter first; measured-only CARS second.
    augmented = augment_full_spectra(
        X_measured_full,
        y_calibration,
        candidate.rensa,
        n_full_wavelengths,
    )
    selector = fit_cars_measured_only(
        X_measured_full,
        y_calibration,
        expected_measured_count=X_calibration.shape[0],
        forbidden_synthetic_count=augmented.n_synthetic,
    )
    selected_indices = np.asarray(selector.selected_indices_, dtype=int)

    # The holdout is first exposed here, after all model-selection decisions and
    # after the final preprocessor, RandomPair+Filter generation, and CARS mask are fixed.
    X_holdout_raw, y_holdout = holdout.reveal("final_prediction")
    X_holdout_full = preprocessor.transform(X_holdout_raw)
    X_synthetic_full = augmented.X[augmented.synthetic_mask]
    y_synthetic = augmented.y[augmented.synthetic_mask]
    Z_measured, Z_synthetic, Z_holdout = apply_fixed_cars_mask(
        selected_indices,
        X_measured_full,
        X_synthetic_full,
        X_holdout_full,
        n_full_wavelengths,
    )
    X_fit = np.vstack([Z_measured, Z_synthetic])
    y_fit = np.concatenate([y_calibration, y_synthetic])
    search = make_svr_search(X_fit.shape[0])
    search.fit(X_fit, y_fit)
    prediction = search.predict(Z_holdout).reshape(-1)
    assert prediction.shape[0] == y_holdout.shape[0]
    best_params = search.best_params_
    return {
        "prediction": prediction,
        "holdout_values": (X_holdout_raw, y_holdout),
        "n_cars_features": int(selected_indices.size),
        "selected_cars_indices": selected_indices.tolist(),
        "requested_synthetic": int(candidate.rensa.n_synthetic),
        "accepted_synthetic": int(augmented.n_synthetic),
        "acceptance_rate": float(augmented.metadata["acceptance_rate"]),
        "best_C": float(best_params["svr__C"]),
        "best_gamma": float(best_params["svr__gamma"]),
        "best_epsilon": float(best_params["svr__epsilon"]),
    }


def fit_preprocessor_measured_only(
    X_measured: np.ndarray,
    method: str,
    *,
    expected_measured_count: int,
) -> Preprocessor:
    X_measured = np.asarray(X_measured, dtype=float)
    assert X_measured.shape[0] == expected_measured_count
    if method not in PREPROCESSING:
        raise ValueError(f"Unsupported preprocessing method: {method}")
    reference = X_measured.mean(axis=0) if method in {"msc", "msc-sg"} else None
    return Preprocessor(method=method, msc_reference=reference)


def augment_full_spectra(
    X_measured_full: np.ndarray,
    y_measured: np.ndarray,
    config: RenSAConfig,
    n_full_wavelengths: int,
):
    assert X_measured_full.shape[1] == n_full_wavelengths
    assert X_measured_full.shape[0] == y_measured.shape[0]
    assert ALGORITHM_SEED == 42
    augmenter = AuditedRandomPairSameFilterAugmenter(
        expected_n_features=n_full_wavelengths,
        n_synthetic=config.n_synthetic,
        response_bins=6,
        response_bin_strategy="quantile",
        neighbors=5,
        alpha_min=0.15,
        alpha_max=0.85,
        noise_scale=config.noise_scale,
        max_spectral_angle=MAX_SPECTRAL_ANGLE,
        min_derivative_corr=MIN_DERIVATIVE_CORR,
        envelope_margin=ENVELOPE_MARGIN,
        random_state=ALGORITHM_SEED,
        neighbor_space="response",
        perturbation_mode=config.perturbation,
    )
    result = augmenter.fit_resample(X_measured_full, y_measured)
    measured_count = X_measured_full.shape[0]
    assert result.n_original == measured_count
    assert not result.synthetic_mask[:measured_count].any()
    assert result.synthetic_mask[measured_count:].all()
    assert np.allclose(result.X[:measured_count], X_measured_full)
    assert np.allclose(result.y[:measured_count], y_measured)
    return result


def fit_cars_measured_only(
    X_measured_full: np.ndarray,
    y_measured: np.ndarray,
    *,
    expected_measured_count: int,
    forbidden_synthetic_count: int,
) -> AuditedMeasuredOnlyCARS:
    selector = AuditedMeasuredOnlyCARS(
        expected_measured_count=expected_measured_count,
        forbidden_synthetic_count=forbidden_synthetic_count,
    )
    selector.fit(X_measured_full, y_measured)
    assert selector.cars_fit_n_samples_ == expected_measured_count
    assert selector.cars_fit_source_ == "measured_only"
    return selector


def apply_fixed_cars_mask(
    selected_indices: np.ndarray,
    X_measured_full: np.ndarray,
    X_synthetic_full: np.ndarray,
    X_evaluation_full: np.ndarray,
    n_full_wavelengths: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_indices = np.asarray(selected_indices, dtype=int)
    matrices = (X_measured_full, X_synthetic_full, X_evaluation_full)
    assert all(matrix.ndim == 2 and matrix.shape[1] == n_full_wavelengths for matrix in matrices)
    assert selected_indices.ndim == 1 and selected_indices.size > 0
    assert np.all(np.diff(selected_indices) > 0)
    assert selected_indices[0] >= 0 and selected_indices[-1] < n_full_wavelengths
    transformed = tuple(matrix[:, selected_indices] for matrix in matrices)
    assert all(matrix.shape[1] == selected_indices.size for matrix in transformed)
    assert np.array_equal(transformed[0], X_measured_full[:, selected_indices])
    assert np.array_equal(transformed[1], X_synthetic_full[:, selected_indices])
    assert np.array_equal(transformed[2], X_evaluation_full[:, selected_indices])
    return transformed


def make_svr_search(n_samples: int) -> GridSearchCV:
    splitter = KFold(
        n_splits=min(INNER_CV, int(n_samples)),
        shuffle=True,
        random_state=ALGORITHM_SEED,
    )
    return GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        param_grid={
            "svr__C": list(SVR_C),
            "svr__gamma": list(SVR_GAMMA),
            "svr__epsilon": list(SVR_EPSILON),
        },
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        n_jobs=16,
        refit=True,
    )


def declared_rensa_configs() -> tuple[RenSAConfig, ...]:
    return tuple(
        RenSAConfig(int(n), str(perturbation), float(noise))
        for n, perturbation, noise in product(N_SYNTHETIC, PERTURBATIONS, NOISE_SCALES)
    )


def effective_rensa_configs() -> tuple[RenSAConfig, ...]:
    """Deduplicate parameter combinations that are identical in the core RenSA implementation."""
    unique: dict[tuple[int, str, float], RenSAConfig] = {}
    for config in declared_rensa_configs():
        key = config.effective_key
        if key not in unique:
            unique[key] = RenSAConfig(*key)
    return tuple(unique.values())


def load_task(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[task]
    path = PROJECT_ROOT / "data" / spec.filename
    if spec.loader == "xprefix":
        wavelengths, X, y = load_spectrum_csv(path, target_column=spec.target, spectral_prefix="x")
    else:
        frame = pd.read_csv(path)
        if spec.target not in frame.columns:
            raise ValueError(f"Missing target {spec.target!r} in {path}.")
        spectral_columns = [
            column
            for column in frame.columns
            if column not in {"Label", spec.target} and is_float_like(column)
        ]
        wavelengths = np.asarray([float(column) for column in spectral_columns], dtype=float)
        X = frame.loc[:, spectral_columns].to_numpy(dtype=float)
        y = frame[spec.target].to_numpy(dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0] or wavelengths.size != X.shape[1]:
        raise ValueError(f"Invalid spectral dataset shape for {task}.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Dataset {task} contains non-finite values.")
    return wavelengths, X, y


def outer_split(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    outer_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    if method in {"spxy", "ks"}:
        assert outer_seed is None
        X_scaled = autoscale(X)
        space = np.column_stack([X_scaled, autoscale(y.reshape(-1, 1))]) if method == "spxy" else X_scaled
        n_calibration = int(round(X.shape[0] * (1.0 - TEST_SIZE)))
        n_calibration = min(max(n_calibration, 2), X.shape[0] - 1)
        calibration = kennard_stone_indices(space, n_calibration)
        mask = np.ones(X.shape[0], dtype=bool)
        mask[calibration] = False
        return calibration, np.flatnonzero(mask), None
    if method != "mc":
        raise ValueError(f"Unknown split method: {method}")
    if outer_seed not in {0, 1, 2, 3, 4}:
        raise ValueError("MC outer split seed must be one of 0, 1, 2, 3, 4.")
    indices = np.arange(X.shape[0], dtype=int)
    for requested_strata in range(5, 0, -1):
        strata = quantile_strata(y, requested_strata)
        actual = int(np.unique(strata).size)
        if actual != requested_strata and requested_strata > 1:
            continue
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=outer_seed)
            calibration, holdout = next(splitter.split(indices, strata))
        except ValueError:
            continue
        return np.sort(indices[calibration]), np.sort(indices[holdout]), actual
    raise RuntimeError("Unable to construct even a one-stratum Monte-Carlo split.")


def quantile_strata(y: np.ndarray, n_strata: int) -> np.ndarray:
    if n_strata <= 1:
        return np.zeros(y.shape[0], dtype=int)
    labels = pd.qcut(pd.Series(y), q=n_strata, labels=False, duplicates="drop")
    values = labels.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Quantile stratification produced missing labels.")
    _, normalized = np.unique(values.astype(int), return_inverse=True)
    return normalized.astype(int)


def kennard_stone_indices(X: np.ndarray, n_select: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    distances = squared_distances(X)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [int(first), int(second)]
    remaining = np.ones(X.shape[0], dtype=bool)
    remaining[selected] = False
    min_distance = np.minimum(distances[:, first], distances[:, second])
    while len(selected) < n_select:
        candidates = np.flatnonzero(remaining)
        next_index = int(candidates[np.argmax(min_distance[candidates])])
        selected.append(next_index)
        remaining[next_index] = False
        min_distance = np.minimum(min_distance, distances[:, next_index])
    return np.asarray(selected, dtype=int)


def squared_distances(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)


def autoscale(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, ddof=1, keepdims=True), 1e-12)


def validate_inner_folds(folds: list[tuple[np.ndarray, np.ndarray]], n_samples: int) -> None:
    validation_counts = np.zeros(n_samples, dtype=int)
    for fit_idx, valid_idx in folds:
        assert np.intersect1d(fit_idx, valid_idx).size == 0
        assert fit_idx.size + valid_idx.size == n_samples
        validation_counts[valid_idx] += 1
    assert np.all(validation_counts == 1), "Inner validation allocation is invalid."


def assert_disjoint_complete_indices(train: np.ndarray, test: np.ndarray, n_samples: int) -> None:
    assert np.intersect1d(train, test).size == 0
    assert train.size + test.size == n_samples
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n_samples))


def distribution_row(
    task: str,
    split_method: str,
    outer_seed: int | None,
    subset: str,
    y: np.ndarray,
) -> dict[str, Any]:
    q1, median, q3 = np.quantile(y, [0.25, 0.50, 0.75])
    return {
        "task": task,
        "split_method": split_method,
        "outer_seed": seed_value(outer_seed),
        "subset": subset,
        "n": int(y.size),
        "min": float(np.min(y)),
        "Q1": float(q1),
        "median": float(median),
        "mean": float(np.mean(y)),
        "Q3": float(q3),
        "max": float(np.max(y)),
        "SD": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
    }


def response_sparsity_row(
    task: str,
    split_method: str,
    outer_seed: int | None,
    y_calibration: np.ndarray,
) -> dict[str, Any]:
    if y_calibration.size < 6:
        raise ValueError("At least six calibration responses are required for fifth-neighbor sparsity.")
    standardized = (y_calibration - y_calibration.mean()) / max(float(y_calibration.std(ddof=1)), 1e-12)
    distances = np.abs(standardized[:, None] - standardized[None, :])
    np.fill_diagonal(distances, np.inf)
    fifth = np.partition(distances, kth=4, axis=1)[:, 4]
    return {
        "task": task,
        "split_method": split_method,
        "outer_seed": seed_value(outer_seed),
        "sparsity_median_5nn": float(np.median(fifth)),
        "sparsity_p90_5nn": float(np.quantile(fifth, 0.90)),
        "sparsity_max_5nn": float(np.max(fifth)),
    }


def save_checkpoint(
    task: str,
    split_method: str,
    outer_seed: int | None,
    split_result: dict[str, Any],
    predictions: list[dict[str, Any]],
    y_statistics: list[dict[str, Any]],
    response_sparsity: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_key": list(experiment_key(task, split_method, outer_seed)),
        "split_result": split_result,
        "predictions": predictions,
        "y_distribution_statistics": y_statistics,
        "response_sparsity": response_sparsity,
    }
    destination = checkpoint_path(task, split_method, outer_seed)
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def rebuild_aggregate_outputs() -> None:
    checkpoints = load_checkpoints()
    split_rows = [item["split_result"] for item in checkpoints]
    prediction_rows = [row for item in checkpoints for row in item["predictions"]]
    y_rows = [row for item in checkpoints for row in item["y_distribution_statistics"]]
    sparsity_rows = [item["response_sparsity"] for item in checkpoints]
    atomic_write_csv(RESULTS_DIR / "split_results.csv", SPLIT_RESULT_FIELDS, split_rows)
    atomic_write_csv(RESULTS_DIR / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
    atomic_write_csv(RESULTS_DIR / "y_distribution_statistics.csv", Y_DISTRIBUTION_FIELDS, y_rows)
    atomic_write_csv(RESULTS_DIR / "response_sparsity.csv", SPARSITY_FIELDS, sparsity_rows)
    atomic_write_csv(RESULTS_DIR / "summary.csv", SUMMARY_FIELDS, build_summary(split_rows))


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for task in DATASETS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        spxy = next((row for row in task_rows if row["split_method"] == "spxy"), None)
        ks = next((row for row in task_rows if row["split_method"] == "ks"), None)
        mc = [row for row in task_rows if row["split_method"] == "mc"]
        mc_r2 = np.asarray([float(row["R2"]) for row in mc], dtype=float)
        mc_rmsep = np.asarray([float(row["RMSEP"]) for row in mc], dtype=float)
        summary.append(
            {
                "task": task,
                "spxy_R2": spxy["R2"] if spxy else "",
                "spxy_RMSEP": spxy["RMSEP"] if spxy else "",
                "ks_R2": ks["R2"] if ks else "",
                "ks_RMSEP": ks["RMSEP"] if ks else "",
                "mc_n": int(mc_r2.size),
                "mc_mean_R2": safe_stat(mc_r2, np.mean),
                "mc_SD_R2": safe_sd(mc_r2),
                "mc_median_R2": safe_stat(mc_r2, np.median),
                "mc_mean_RMSEP": safe_stat(mc_rmsep, np.mean),
                "mc_SD_RMSEP": safe_sd(mc_rmsep),
                "mc_median_RMSEP": safe_stat(mc_rmsep, np.median),
                "mc_min_RMSEP": safe_stat(mc_rmsep, np.min),
                "mc_max_RMSEP": safe_stat(mc_rmsep, np.max),
            }
        )
    return summary


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
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
    task_order = {task: index for index, task in enumerate(DATASETS)}
    method_order = {"spxy": 0, "ks": 1, "mc": 2}
    items.sort(
        key=lambda item: (
            task_order[item["split_result"]["task"]],
            method_order[item["split_result"]["split_method"]],
            -1 if item["split_result"]["outer_seed"] == "" else int(item["split_result"]["outer_seed"]),
        )
    )
    return items


def completed_keys() -> set[tuple[str, str, str]]:
    keys = set()
    if not CHECKPOINT_DIR.exists():
        return keys
    for path in CHECKPOINT_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = payload["experiment_key"]
        keys.add((str(raw[0]), str(raw[1]), str(raw[2])))
    return keys


def experiment_key(task: str, split_method: str, outer_seed: int | None) -> tuple[str, str, str]:
    return task, split_method, "NA" if outer_seed is None else str(int(outer_seed))


def checkpoint_path(task: str, split_method: str, outer_seed: int | None) -> Path:
    seed = "deterministic" if outer_seed is None else f"seed{int(outer_seed)}"
    return CHECKPOINT_DIR / f"{task}__{split_method}__{seed}.json"


def seed_value(outer_seed: int | None) -> int | str:
    return "" if outer_seed is None else int(outer_seed)


def seed_text(outer_seed: int | None) -> str:
    return "NA" if outer_seed is None else str(int(outer_seed))


def safe_stat(values: np.ndarray, function) -> float | str:
    return float(function(values)) if values.size else ""


def safe_sd(values: np.ndarray) -> float | str:
    return float(np.std(values, ddof=1)) if values.size > 1 else ""


def is_float_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


if __name__ == "__main__":
    main()
