"""R3.3b Bjerrum et al. (2017) spectral-augmentation baseline.

This isolated script reuses each RenSA outer run's measured-only CARS mask,
preprocessing choice, requested synthetic-sample count, outer split, random
seed, and RBF-SVR search protocol. Bjerrum augmentation is applied only after
the fixed CARS mask, while slope positions retain their coordinates on the
complete original wavelength axis.
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
RESULTS_DIR = EXPERIMENT_ROOT / "results_bjerrum"
LOGS_DIR = EXPERIMENT_ROOT / "logs_bjerrum"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RENSA_RESULTS = EXPERIMENT_ROOT / "results" / "split_results.csv"

sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import run_renvsa as base  # noqa: E402


BASELINE = "bjerrum_2017"
SLOPE_RANGE = (0.95, 1.05)
MULTIPLICATION_RANGE = (0.90, 1.10)
OFFSET_SIGMA_FRACTION = 0.10


@dataclass(frozen=True)
class RenSAReference:
    task: str
    split_method: str
    outer_seed: int | None
    algorithm_seed: int
    preprocessing: str
    best_n_synthetic: int
    requested_synthetic: int
    accepted_synthetic: int
    n_calibration: int
    n_holdout: int
    n_full_wavelengths: int
    n_cars_features: int
    selected_cars_indices: np.ndarray


@dataclass(frozen=True)
class BjerrumResult:
    X_synthetic: np.ndarray
    y_synthetic: np.ndarray
    parent_indices: np.ndarray
    multiplication_factors: np.ndarray
    slope_factors: np.ndarray
    offsets: np.ndarray
    sigma_X: float


RESULT_FIELDS = [
    "baseline",
    "task",
    "split_method",
    "outer_seed",
    "actual_strata",
    "algorithm_seed",
    "n_total_samples",
    "n_measured_samples",
    "n_holdout",
    "n_synthetic_samples",
    "augmentation_count_source",
    "rensa_requested_synthetic",
    "rensa_accepted_synthetic",
    "n_full_wavelengths",
    "n_cars_features",
    "selected_cars_indices",
    "selected_cars_wavelengths",
    "selected_normalized_positions",
    "wavelength_axis_min",
    "wavelength_axis_max",
    "preprocessing",
    "sigma_X",
    "slope_factor_min",
    "slope_factor_max",
    "multiplication_factor_min",
    "multiplication_factor_max",
    "offset_min",
    "offset_max",
    "gaussian_noise",
    "spectral_consistency_filter",
    "best_C",
    "best_gamma",
    "best_epsilon",
    "inner_cv_folds",
    "inner_cv_RMSEP",
    "R2",
    "RMSEP",
    "Bias",
    "Slope",
    "SEP",
    "calibration_indices",
    "holdout_indices",
    "rensa_reference_file",
    "runtime_seconds",
]

PREDICTION_FIELDS = [
    "baseline",
    "task",
    "split_method",
    "outer_seed",
    "sample_index",
    "y_true",
    "y_pred",
    "residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bjerrum et al. (2017) spectral perturbation with each RenSA "
            "run's measured-only CARS mask and requested augmentation count."
        )
    )
    parser.add_argument("--tasks", nargs="+", choices=tuple(base.DATASETS), default=list(base.DATASETS))
    parser.add_argument(
        "--split-methods",
        nargs="+",
        choices=("spxy", "ks", "mc"),
        default=["spxy", "ks", "mc"],
    )
    parser.add_argument("--mc-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--rensa-results",
        type=Path,
        default=RENSA_RESULTS,
        help="Read-only RenSA split_results.csv containing the fixed CARS masks and counts.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_mc_seeds = sorted(set(args.mc_seeds) - {0, 1, 2, 3, 4})
    if invalid_mc_seeds:
        raise ValueError(f"MC outer seeds must be selected from 0..4, got {invalid_mc_seeds}.")

    references = load_rensa_references(args.rensa_results)
    runs = list(requested_runs(args))
    missing = [experiment_key(*run) for run in runs if experiment_key(*run) not in references]
    if missing:
        raise KeyError(f"RenSA reference rows are missing for: {missing}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    completed = completed_keys()
    new_runs = 0
    for task, split_method, outer_seed in runs:
        key = experiment_key(task, split_method, outer_seed)
        if key in completed:
            if args.resume:
                logging.info("skip completed task=%s split=%s seed=%s", task, split_method, seed_text(outer_seed))
                continue
            raise FileExistsError(f"Bjerrum result {key} already exists; use --resume to skip it.")
        if args.max_runs > 0 and new_runs >= args.max_runs:
            break

        run_one(
            task=task,
            split_method=split_method,
            outer_seed=outer_seed,
            reference=references[key],
            reference_path=args.rensa_results,
        )
        completed.add(key)
        new_runs += 1
        rebuild_aggregate_outputs()

    rebuild_aggregate_outputs()
    logging.info("finished new_runs=%d completed=%d", new_runs, len(completed_keys()))


def requested_runs(args: argparse.Namespace) -> Iterable[tuple[str, str, int | None]]:
    yield from base.requested_runs(args)


def run_one(
    task: str,
    split_method: str,
    outer_seed: int | None,
    reference: RenSAReference,
    reference_path: Path,
) -> None:
    started = time.perf_counter()
    wavelengths, X, y = base.load_task(task)
    calibration_idx, holdout_idx, actual_strata = base.outer_split(X, y, split_method, outer_seed)
    base.assert_disjoint_complete_indices(calibration_idx, holdout_idx, X.shape[0])
    validate_reference(
        reference,
        task=task,
        split_method=split_method,
        outer_seed=outer_seed,
        n_calibration=calibration_idx.size,
        n_holdout=holdout_idx.size,
        n_full_wavelengths=X.shape[1],
    )

    X_calibration = X[calibration_idx].copy()
    y_calibration = y[calibration_idx].copy()
    holdout = base.HoldoutGuard(X, y, holdout_idx)

    preprocessor = base.fit_preprocessor_measured_only(
        X_calibration,
        reference.preprocessing,
        expected_measured_count=calibration_idx.size,
    )
    X_measured_full = preprocessor.transform(X_calibration)
    selected_indices = reference.selected_cars_indices.copy()
    selected_wavelengths, selected_positions = selected_wavelength_mapping(
        wavelengths,
        selected_indices,
    )
    Z_measured = X_measured_full[:, selected_indices]

    augmented = bjerrum_augment(
        Z_measured,
        y_calibration,
        selected_positions,
        n_synthetic=reference.requested_synthetic,
        random_state=base.ALGORITHM_SEED,
    )
    X_fit = np.vstack([Z_measured, augmented.X_synthetic])
    y_fit = np.concatenate([y_calibration, augmented.y_synthetic])
    assert augmented.X_synthetic.shape[0] == reference.requested_synthetic
    assert X_fit.shape[0] == calibration_idx.size + reference.requested_synthetic

    search = base.make_svr_search(X_fit.shape[0])
    search.fit(X_fit, y_fit)

    X_holdout_raw, y_holdout = holdout.reveal("final_prediction")
    X_holdout_full = preprocessor.transform(X_holdout_raw)
    Z_holdout = X_holdout_full[:, selected_indices]
    prediction = search.predict(Z_holdout).reshape(-1)
    metrics = regression_metrics(y_holdout, prediction)
    params = search.best_params_
    elapsed = time.perf_counter() - started

    row = {
        "baseline": BASELINE,
        "task": task,
        "split_method": split_method,
        "outer_seed": base.seed_value(outer_seed),
        "actual_strata": actual_strata if actual_strata is not None else "",
        "algorithm_seed": base.ALGORITHM_SEED,
        "n_total_samples": int(X.shape[0]),
        "n_measured_samples": int(calibration_idx.size),
        "n_holdout": int(holdout_idx.size),
        "n_synthetic_samples": int(augmented.X_synthetic.shape[0]),
        "augmentation_count_source": "RenSA best_n_syn/requested_synthetic",
        "rensa_requested_synthetic": reference.requested_synthetic,
        "rensa_accepted_synthetic": reference.accepted_synthetic,
        "n_full_wavelengths": int(X.shape[1]),
        "n_cars_features": int(selected_indices.size),
        "selected_cars_indices": json_int_array(selected_indices),
        "selected_cars_wavelengths": json_float_array(selected_wavelengths),
        "selected_normalized_positions": json_float_array(selected_positions),
        "wavelength_axis_min": float(np.min(wavelengths)),
        "wavelength_axis_max": float(np.max(wavelengths)),
        "preprocessing": reference.preprocessing,
        "sigma_X": augmented.sigma_X,
        "slope_factor_min": SLOPE_RANGE[0],
        "slope_factor_max": SLOPE_RANGE[1],
        "multiplication_factor_min": MULTIPLICATION_RANGE[0],
        "multiplication_factor_max": MULTIPLICATION_RANGE[1],
        "offset_min": -OFFSET_SIGMA_FRACTION * augmented.sigma_X,
        "offset_max": OFFSET_SIGMA_FRACTION * augmented.sigma_X,
        "gaussian_noise": False,
        "spectral_consistency_filter": False,
        "best_C": float(params["svr__C"]),
        "best_gamma": float(params["svr__gamma"]),
        "best_epsilon": float(params["svr__epsilon"]),
        "inner_cv_folds": base.INNER_CV,
        "inner_cv_RMSEP": -float(search.best_score_),
        **metrics,
        "calibration_indices": json_int_array(calibration_idx),
        "holdout_indices": json_int_array(holdout_idx),
        "rensa_reference_file": str(reference_path.resolve()),
        "runtime_seconds": float(elapsed),
    }
    predictions = [
        {
            "baseline": BASELINE,
            "task": task,
            "split_method": split_method,
            "outer_seed": base.seed_value(outer_seed),
            "sample_index": int(sample_idx),
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "residual": float(y_true - y_pred),
        }
        for sample_idx, y_true, y_pred in zip(holdout_idx, y_holdout, prediction)
    ]
    augmentation_audit = {
        "formula": "x_prime = m*x - (s-1)*(t-0.5) - beta",
        "sigma_scope": "measured_calibration_after_preprocessing_and_fixed_CARS",
        "parent_indices_within_calibration": augmented.parent_indices.tolist(),
        "parent_original_sample_indices": calibration_idx[augmented.parent_indices].tolist(),
        "multiplication_factors": augmented.multiplication_factors.tolist(),
        "slope_factors": augmented.slope_factors.tolist(),
        "offsets": augmented.offsets.tolist(),
    }
    save_checkpoint(task, split_method, outer_seed, row, predictions, augmentation_audit)
    logging.info(
        "done task=%s split=%s seed=%s n_measured=%d n_synthetic=%d R2=%.6f RMSEP=%.6f",
        task,
        split_method,
        seed_text(outer_seed),
        calibration_idx.size,
        augmented.X_synthetic.shape[0],
        row["R2"],
        row["RMSEP"],
    )


def bjerrum_augment(
    X_measured: np.ndarray,
    y_measured: np.ndarray,
    normalized_positions: np.ndarray,
    *,
    n_synthetic: int,
    random_state: int,
) -> BjerrumResult:
    """Generate exact-count Bjerrum spectra from measured parents only.

    ``sigma_X`` is the population standard deviation of the current measured
    calibration matrix after preprocessing and application of the fixed CARS
    mask. No validation/holdout values enter this calculation.
    """

    X_measured = np.asarray(X_measured, dtype=float)
    y_measured = np.asarray(y_measured, dtype=float).reshape(-1)
    positions = np.asarray(normalized_positions, dtype=float).reshape(-1)
    if X_measured.ndim != 2 or X_measured.shape[0] != y_measured.size:
        raise ValueError("X_measured must be 2D with one y value per measured spectrum.")
    if X_measured.shape[0] < 1:
        raise ValueError("At least one measured parent spectrum is required.")
    if positions.size != X_measured.shape[1]:
        raise ValueError("normalized_positions must match the CARS-selected variables.")
    if not np.isfinite(X_measured).all() or not np.isfinite(y_measured).all():
        raise ValueError("Measured calibration data must be finite.")
    if not np.isfinite(positions).all() or np.any((positions < 0.0) | (positions > 1.0)):
        raise ValueError("Normalized wavelength positions must be finite and within [0, 1].")
    if int(n_synthetic) != n_synthetic or int(n_synthetic) < 1:
        raise ValueError("n_synthetic must be a positive integer.")

    n_synthetic = int(n_synthetic)
    rng = np.random.default_rng(int(random_state))
    parent_indices = rng.integers(0, X_measured.shape[0], size=n_synthetic)
    multiplication = rng.uniform(*MULTIPLICATION_RANGE, size=n_synthetic)
    slopes = rng.uniform(*SLOPE_RANGE, size=n_synthetic)
    sigma_X = float(np.std(X_measured, ddof=0))
    offsets = rng.uniform(
        -OFFSET_SIGMA_FRACTION * sigma_X,
        OFFSET_SIGMA_FRACTION * sigma_X,
        size=n_synthetic,
    )

    parents = X_measured[parent_indices]
    X_synthetic = (
        multiplication[:, None] * parents
        - (slopes[:, None] - 1.0) * (positions[None, :] - 0.5)
        - offsets[:, None]
    )
    y_synthetic = y_measured[parent_indices].copy()
    if X_synthetic.shape != (n_synthetic, X_measured.shape[1]):
        raise AssertionError("Bjerrum augmentation returned an unexpected shape.")
    if not np.isfinite(X_synthetic).all() or not np.isfinite(y_synthetic).all():
        raise ValueError("Bjerrum augmentation produced non-finite values.")

    return BjerrumResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        parent_indices=parent_indices,
        multiplication_factors=multiplication,
        slope_factors=slopes,
        offsets=offsets,
        sigma_X=sigma_X,
    )


def selected_wavelength_mapping(
    full_wavelengths: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map zero-based CARS indices to the complete original wavelength axis."""

    axis = np.asarray(full_wavelengths, dtype=float).reshape(-1)
    selected = np.asarray(selected_indices, dtype=int).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all():
        raise ValueError("The complete original wavelength axis must contain finite coordinates.")
    if selected.size < 1 or selected[0] < 0 or selected[-1] >= axis.size:
        raise ValueError("CARS indices fall outside the complete original wavelength axis.")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("CARS indices must be unique and strictly increasing.")
    axis_min = float(np.min(axis))
    axis_max = float(np.max(axis))
    if axis_max <= axis_min:
        raise ValueError("The complete original wavelength axis has no positive span.")
    selected_wavelengths = axis[selected]
    positions = (selected_wavelengths - axis_min) / (axis_max - axis_min)
    return selected_wavelengths, positions


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return external metrics; Bias uses prediction minus reference.

    Slope is from the least-squares regression of predicted values on true
    values. SEP is the sample standard deviation (ddof=1) of prediction errors.
    """

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size or y_true.size < 2:
        raise ValueError("External metrics require at least two paired observations.")
    errors = y_pred - y_true
    centered_true = y_true - np.mean(y_true)
    denominator = float(centered_true @ centered_true)
    if denominator <= 0.0:
        raise ValueError("Slope is undefined for a constant external reference response.")
    slope = float(centered_true @ (y_pred - np.mean(y_pred)) / denominator)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSEP": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Bias": float(np.mean(errors)),
        "Slope": slope,
        "SEP": float(np.std(errors, ddof=1)),
    }


def load_rensa_references(path: Path) -> dict[tuple[str, str, str], RenSAReference]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RenSA split results were not found: {path}")
    required = {
        "task",
        "split_method",
        "outer_seed",
        "algorithm_seed",
        "preprocessing",
        "best_n_syn",
        "requested_synthetic",
        "accepted_synthetic",
        "n_calibration",
        "n_holdout",
        "n_full_wavelengths",
        "n_cars_features",
        "selected_cars_indices",
    }
    references: dict[tuple[str, str, str], RenSAReference] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"RenSA split results are missing columns: {sorted(missing)}")
        for row in reader:
            seed = None if str(row["outer_seed"]).strip() == "" else int(row["outer_seed"])
            best_n_synthetic = int(row["best_n_syn"])
            requested_synthetic = int(row["requested_synthetic"])
            if best_n_synthetic != requested_synthetic:
                raise ValueError(
                    "RenSA best_n_syn and requested_synthetic disagree for "
                    f"{row['task']}/{row['split_method']}/{row['outer_seed']}."
                )
            raw_indices = json.loads(row["selected_cars_indices"])
            selected_indices = np.asarray(raw_indices, dtype=int)
            reference = RenSAReference(
                task=str(row["task"]),
                split_method=str(row["split_method"]),
                outer_seed=seed,
                algorithm_seed=int(row["algorithm_seed"]),
                preprocessing=str(row["preprocessing"]),
                best_n_synthetic=best_n_synthetic,
                requested_synthetic=requested_synthetic,
                accepted_synthetic=int(row["accepted_synthetic"]),
                n_calibration=int(row["n_calibration"]),
                n_holdout=int(row["n_holdout"]),
                n_full_wavelengths=int(row["n_full_wavelengths"]),
                n_cars_features=int(row["n_cars_features"]),
                selected_cars_indices=selected_indices,
            )
            key = experiment_key(reference.task, reference.split_method, reference.outer_seed)
            if key in references:
                raise ValueError(f"Duplicate RenSA reference row: {key}")
            references[key] = reference
    return references


def validate_reference(
    reference: RenSAReference,
    *,
    task: str,
    split_method: str,
    outer_seed: int | None,
    n_calibration: int,
    n_holdout: int,
    n_full_wavelengths: int,
) -> None:
    expected_key = experiment_key(task, split_method, outer_seed)
    if experiment_key(reference.task, reference.split_method, reference.outer_seed) != expected_key:
        raise ValueError("RenSA reference row does not match the requested outer run.")
    if reference.algorithm_seed != base.ALGORITHM_SEED:
        raise ValueError("RenSA and Bjerrum algorithm seeds do not match.")
    if reference.preprocessing not in base.PREPROCESSING:
        raise ValueError(f"Unsupported referenced preprocessing: {reference.preprocessing}")
    if reference.requested_synthetic < 1:
        raise ValueError("The referenced RenSA augmentation count must be positive.")
    observed_shape = (n_calibration, n_holdout, n_full_wavelengths)
    referenced_shape = (
        reference.n_calibration,
        reference.n_holdout,
        reference.n_full_wavelengths,
    )
    if observed_shape != referenced_shape:
        raise ValueError(
            f"Current data/split shape {observed_shape} differs from RenSA {referenced_shape}."
        )
    selected = reference.selected_cars_indices
    if selected.ndim != 1 or selected.size != reference.n_cars_features:
        raise ValueError("Stored CARS index count does not match n_cars_features.")
    if selected.size < 1 or selected[0] < 0 or selected[-1] >= n_full_wavelengths:
        raise ValueError("Stored CARS indices are outside the complete spectral dimension.")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("Stored CARS indices must be unique and strictly increasing.")


def save_checkpoint(
    task: str,
    split_method: str,
    outer_seed: int | None,
    result: dict[str, Any],
    predictions: list[dict[str, Any]],
    augmentation_audit: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_key": list(experiment_key(task, split_method, outer_seed)),
        "result": result,
        "predictions": predictions,
        "augmentation_audit": augmentation_audit,
    }
    destination = checkpoint_path(task, split_method, outer_seed)
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def rebuild_aggregate_outputs() -> None:
    checkpoints = load_checkpoints()
    rows = [item["result"] for item in checkpoints]
    predictions = [row for item in checkpoints for row in item["predictions"]]
    base.atomic_write_csv(
        RESULTS_DIR / "bjerrum_baseline_results.csv",
        RESULT_FIELDS,
        rows,
    )
    base.atomic_write_csv(
        RESULTS_DIR / "bjerrum_baseline_predictions.csv",
        PREDICTION_FIELDS,
        predictions,
    )
    base.atomic_write_csv(
        RESULTS_DIR / "bjerrum_baseline_summary.csv",
        base.SUMMARY_FIELDS,
        base.build_summary(rows),
    )


def load_checkpoints() -> list[dict[str, Any]]:
    if not CHECKPOINT_DIR.exists():
        return []
    items = []
    for path in CHECKPOINT_DIR.glob("bjerrum__*.json"):
        with path.open(encoding="utf-8") as handle:
            items.append(json.load(handle))
    task_order = {task: index for index, task in enumerate(base.DATASETS)}
    method_order = {"spxy": 0, "ks": 1, "mc": 2}
    items.sort(
        key=lambda item: (
            task_order[item["result"]["task"]],
            method_order[item["result"]["split_method"]],
            -1 if item["result"]["outer_seed"] == "" else int(item["result"]["outer_seed"]),
        )
    )
    return items


def completed_keys() -> set[tuple[str, str, str]]:
    return {tuple(str(value) for value in item["experiment_key"]) for item in load_checkpoints()}


def experiment_key(task: str, split_method: str, outer_seed: int | None) -> tuple[str, str, str]:
    return task, split_method, "NA" if outer_seed is None else str(int(outer_seed))


def checkpoint_path(task: str, split_method: str, outer_seed: int | None) -> Path:
    seed = "deterministic" if outer_seed is None else f"seed{int(outer_seed)}"
    return CHECKPOINT_DIR / f"bjerrum__{task}__{split_method}__{seed}.json"


def json_int_array(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values, dtype=int).tolist(), separators=(",", ":"))


def json_float_array(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values, dtype=float).tolist(), separators=(",", ":"))


def seed_text(outer_seed: int | None) -> str:
    return "NA" if outer_seed is None else str(int(outer_seed))


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
