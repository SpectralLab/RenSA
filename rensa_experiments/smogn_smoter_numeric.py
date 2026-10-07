"""Audited numeric implementation of the published SMOTER and SMOGN algorithms.

The implementation is restricted to finite, continuous predictor matrices, as
required by the spectral experiments.  It follows the SMOTER paper algorithm,
the SMOGN paper pseudocode, and UBL 0.0.9 for relevance-defined response bumps
and the ``balance``/``extreme`` sampling factors.

The two available Python ports (``smogn==0.1.2`` and
``ImbalancedLearningRegression==0.0.2``) are retained as provenance references,
but their over-sampling kernels are not called: both include the target in the
numeric-feature loop and read its uninitialised output cell while calculating a
synthetic target.  This module applies the published predictor-only distance and
fully assigns every synthetic row before it is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import smogn


UBL_VERSION = "0.0.9"
UBL_SMOTER_SOURCE = "https://rdrr.io/cran/UBL/src/R/smoteRegress.R"
UBL_SMOGN_SOURCE = "https://rdrr.io/cran/UBL/src/R/smognRegress.R"
SMOTER_PAPER_DOI = "10.1007/978-3-642-40669-0_33"
SMOGN_PAPER_URL = "https://proceedings.mlr.press/v74/branco17a.html"
IMPLEMENTATION_VERSION = "audited-numeric-1.0"


@dataclass(frozen=True)
class NumericSamplingResult:
    X: np.ndarray
    y: np.ndarray
    n_interpolated: int
    n_gaussian: int
    n_replicated: int
    bump_sizes: tuple[int, ...]
    sampling_factors: tuple[float, ...]


def resample_numeric(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    relevance_control_points: Sequence[Sequence[float]],
    relevance_threshold: float,
    k: int,
    sampling_strategy: str,
    perturbation: float,
    random_state: int,
) -> NumericSamplingResult:
    """Apply standard SMOTER or SMOGN to a numeric training fold."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    _validate_inputs(
        method, X, y, relevance_threshold, k, sampling_strategy, perturbation
    )

    ordered_indices, bumps = relevance_bumps(
        y, relevance_control_points, relevance_threshold
    )
    bump_sizes = np.asarray([indices.size for indices in bumps], dtype=int)
    factors = sampling_factors(y.size, bump_sizes, sampling_strategy)
    rng = np.random.default_rng(int(random_state))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_interpolated = 0
    n_gaussian = 0
    n_replicated = 0

    for indices, factor in zip(bumps, factors):
        X_bump = X[indices]
        y_bump = y[indices]
        if factor == 1.0:
            X_parts.append(X_bump.copy())
            y_parts.append(y_bump.copy())
            continue
        if factor < 1.0:
            keep = int(np.rint(factor * indices.size))
            if keep:
                selected = rng.choice(indices.size, size=keep, replace=False)
                X_parts.append(X_bump[selected].copy())
                y_parts.append(y_bump[selected].copy())
            continue

        anchors = _synthetic_anchor_indices(indices.size, factor, rng)
        if anchors.size:
            if indices.size == 1:
                X_synth = np.repeat(X_bump, anchors.size, axis=0)
                y_synth = np.repeat(y_bump, anchors.size)
                n_replicated += int(anchors.size)
            else:
                X_synth, y_synth, interpolated, gaussian = _generate_synthetic(
                    method, X_bump, y_bump, anchors, k, perturbation, rng
                )
                n_interpolated += interpolated
                n_gaussian += gaussian
            X_parts.append(X_synth)
            y_parts.append(y_synth)
        X_parts.append(X_bump.copy())
        y_parts.append(y_bump.copy())

    if not X_parts:
        raise RuntimeError("Sampling removed every training observation.")
    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    if X_out.shape[0] != y_out.size or X_out.shape[1] != X.shape[1]:
        raise RuntimeError("Resampler produced incompatible output dimensions.")
    if not np.isfinite(X_out).all() or not np.isfinite(y_out).all():
        raise RuntimeError("Resampler produced non-finite values.")

    # This also asserts that the response sort used for the bumps is a complete
    # permutation, making accidental row loss before sampling detectable.
    if not np.array_equal(np.sort(ordered_indices), np.arange(y.size)):
        raise AssertionError("Response ordering is not a complete permutation.")
    return NumericSamplingResult(
        X=X_out,
        y=y_out,
        n_interpolated=n_interpolated,
        n_gaussian=n_gaussian,
        n_replicated=n_replicated,
        bump_sizes=tuple(int(value) for value in bump_sizes),
        sampling_factors=tuple(float(value) for value in factors),
    )


def relevance_bumps(
    y: np.ndarray,
    control_points: Sequence[Sequence[float]],
    threshold: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return response-sorted indices and contiguous relevance partitions."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(y, kind="stable")
    y_sorted = pd.Series(y[order])
    phi_parameters = smogn.phi_ctrl_pts(
        y_sorted,
        method="manual",
        ctrl_pts=[list(map(float, point)) for point in control_points],
    )
    relevance = np.asarray(smogn.phi(y_sorted, phi_parameters), dtype=float)
    if relevance.shape != y.shape or not np.isfinite(relevance).all():
        raise RuntimeError("The relevance function returned invalid values.")
    rare = relevance >= float(threshold)
    cut_positions = np.flatnonzero(rare[:-1] != rare[1:]) + 1
    bumps = tuple(part for part in np.split(order, cut_positions) if part.size)
    return order, bumps


def sampling_factors(
    n_samples: int, bump_sizes: np.ndarray, strategy: str
) -> np.ndarray:
    """Reproduce UBL's automatic ``balance`` and ``extreme`` C.perc rules."""
    sizes = np.asarray(bump_sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes <= 0):
        raise ValueError("bump_sizes must contain positive counts.")
    target = float(np.rint(float(n_samples) / sizes.size))
    if strategy == "balance":
        factors = target / sizes
    elif strategy == "extreme":
        rescale = sizes.size * target / np.sum(target**2 / sizes)
        objectives = np.round((target**2 / sizes) * rescale, 2)
        factors = np.round(objectives / sizes, 1)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if not np.isfinite(factors).all() or np.any(factors < 0):
        raise RuntimeError("Invalid automatic sampling factors.")
    return factors.astype(float)


def predictor_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances using predictors only (never the target)."""
    X = np.asarray(X, dtype=np.float64)
    squared_norm = np.einsum("ij,ij->i", X, X)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (X @ X.T)
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared, out=squared)
    np.fill_diagonal(distances, 0.0)
    return distances


def _synthetic_anchor_indices(
    bump_size: int, factor: float, rng: np.random.Generator
) -> np.ndarray:
    integer_passes = int(np.floor(factor - 1.0))
    fractional = float(factor - 1.0 - integer_passes)
    extra = int(np.rint(bump_size * fractional))
    repeated = np.tile(np.arange(bump_size, dtype=int), integer_passes)
    if extra:
        selected = rng.choice(bump_size, size=extra, replace=False).astype(int)
        return np.concatenate([repeated, selected])
    return repeated


def _generate_synthetic(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    anchors: np.ndarray,
    k: int,
    perturbation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    distances = predictor_distance_matrix(X)
    neighbour_distances = distances.copy()
    np.fill_diagonal(neighbour_distances, np.inf)
    k_effective = min(int(k), X.shape[0] - 1)
    neighbours = np.argsort(neighbour_distances, axis=1, kind="stable")[:, :k_effective]

    choices = rng.integers(0, k_effective, size=anchors.size)
    selected_neighbours = neighbours[anchors, choices]
    selected_distances = distances[anchors, selected_neighbours]
    if method == "smogn":
        off_diagonal = distances.copy()
        np.fill_diagonal(off_diagonal, np.nan)
        safe_radius = 0.5 * np.nanmedian(off_diagonal, axis=1)
        interpolate_mask = selected_distances < safe_radius[anchors]
    else:
        safe_radius = np.zeros(X.shape[0], dtype=float)
        interpolate_mask = np.ones(anchors.size, dtype=bool)

    X_new = np.empty((anchors.size, X.shape[1]), dtype=np.float64)
    y_new = np.empty(anchors.size, dtype=np.float64)
    interpolation_rows = np.flatnonzero(interpolate_mask)
    if interpolation_rows.size:
        anchor_ids = anchors[interpolation_rows]
        neighbour_ids = selected_neighbours[interpolation_rows]
        weights = rng.random(interpolation_rows.size)
        X_delta = X[neighbour_ids] - X[anchor_ids]
        X_new[interpolation_rows] = X[anchor_ids] + weights[:, None] * X_delta
        feature_distance = np.linalg.norm(X_delta, axis=1)
        y_interpolated = (
            (1.0 - weights) * y[anchor_ids] + weights * y[neighbour_ids]
        )
        identical = feature_distance <= np.finfo(float).eps
        if np.any(identical):
            y_interpolated[identical] = 0.5 * (
                y[anchor_ids[identical]] + y[neighbour_ids[identical]]
            )
        y_new[interpolation_rows] = y_interpolated

    gaussian_rows = np.flatnonzero(~interpolate_mask)
    if gaussian_rows.size:
        anchor_ids = anchors[gaussian_rows]
        caps = np.minimum(safe_radius[anchor_ids], float(perturbation))
        X_scale = np.std(X, axis=0, ddof=0)
        y_scale = float(np.std(y, ddof=0))
        X_noise = rng.normal(size=(gaussian_rows.size, X.shape[1]))
        y_noise = rng.normal(size=gaussian_rows.size)
        X_new[gaussian_rows] = X[anchor_ids] + X_noise * X_scale * caps[:, None]
        y_new[gaussian_rows] = y[anchor_ids] + y_noise * y_scale * caps

    return (
        X_new,
        y_new,
        int(interpolation_rows.size),
        int(gaussian_rows.size),
    )


def _validate_inputs(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    k: int,
    strategy: str,
    perturbation: float,
) -> None:
    if method not in {"smoter", "smogn"}:
        raise ValueError(f"Unknown method: {method}")
    if X.ndim != 2 or X.shape[0] != y.size or X.shape[0] < 2:
        raise ValueError("X and y have incompatible or insufficient dimensions.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Standard resamplers require finite training data.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("relevance_threshold must be between zero and one.")
    if int(k) < 1:
        raise ValueError("k must be positive.")
    if strategy not in {"balance", "extreme"}:
        raise ValueError("sampling_strategy must be balance or extreme.")
    if not 0.0 < float(perturbation) <= 1.0:
        raise ValueError("perturbation must be in (0, 1].")
