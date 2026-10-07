"""Response-driven spectrum generation and augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class AugmentationResult:
    """Container returned by :class:`ResponseDrivenAugmenter`."""

    X: np.ndarray
    y: np.ndarray
    synthetic_mask: np.ndarray
    metadata: dict

    @property
    def n_original(self) -> int:
        return int((~self.synthetic_mask).sum())

    @property
    def n_synthetic(self) -> int:
        return int(self.synthetic_mask.sum())


@dataclass
class ResponseDrivenAugmenter:
    """Generate spectra in the joint spectrum-response space.

    The algorithm draws anchors preferentially from sparse response bins, pairs
    them with local response neighbors, interpolates both spectrum and response,
    adds a small local perturbation, then rejects samples that violate basic
    spectral consistency checks.
    """

    n_synthetic: int = 100
    response_bins: int = 6
    neighbors: int = 5
    alpha_min: float = 0.15
    alpha_max: float = 0.85
    noise_scale: float = 0.015
    max_spectral_angle: float = 0.18
    min_derivative_corr: float = 0.70
    envelope_margin: float = 0.08
    max_attempt_multiplier: int = 40
    random_state: int | None = None
    neighbor_space: str = "joint"
    spectrum_weight: float = 0.35
    response_bin_strategy: str = "quantile"
    perturbation_mode: str = "local_std"
    response_consistency: bool = False
    max_response_deviation: float = 0.35

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> AugmentationResult:
        X, y = self._validate_inputs(X, y)
        if self.n_synthetic <= 0:
            return AugmentationResult(
                X=X.copy(),
                y=y.copy(),
                synthetic_mask=np.zeros(X.shape[0], dtype=bool),
                metadata={"accepted": 0, "attempted": 0, "acceptance_rate": 0.0},
            )

        rng = np.random.default_rng(self.random_state)
        bins = self._response_bins(y)
        anchor_weights = self._anchor_weights(bins)
        neighbor_indices = self._local_neighbors(X, y)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[float] = []
        diagnostics = _AugmentationDiagnostics()
        attempted = 0
        max_attempts = max(self.n_synthetic * self.max_attempt_multiplier, self.n_synthetic)

        while len(synthetic_X) < self.n_synthetic and attempted < max_attempts:
            attempted += 1
            i = int(rng.choice(X.shape[0], p=anchor_weights))
            candidates = neighbor_indices[i]
            if candidates.size == 0:
                diagnostics.rejections["no_neighbor"] += 1
                continue
            j = int(rng.choice(candidates))
            alpha = float(rng.uniform(self.alpha_min, self.alpha_max))

            base_x = (1.0 - alpha) * X[i] + alpha * X[j]
            base_y = (1.0 - alpha) * y[i] + alpha * y[j]
            x_new = base_x + self._local_perturbation(X, candidates, rng)

            check = self._consistency_report(x_new, base_x, X[i], X[j], base_y, X, y)
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
            "neighbor_space": self.neighbor_space,
            "perturbation_mode": self.perturbation_mode,
            **diagnostics.as_metadata(),
        }
        return AugmentationResult(X=X_out, y=y_out, synthetic_mask=synthetic_mask, metadata=metadata)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array shaped as samples x variables.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be a 1D array with one response per spectrum.")
        if X.shape[0] < 3:
            raise ValueError("At least three spectra are required for response-driven augmentation.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")
        if not (0.0 <= self.alpha_min < self.alpha_max <= 1.0):
            raise ValueError("alpha_min and alpha_max must satisfy 0 <= min < max <= 1.")
        if self.neighbor_space not in {"joint", "spectrum", "response"}:
            raise ValueError("neighbor_space must be one of: joint, spectrum, response.")
        if self.response_bin_strategy not in {"quantile", "uniform"}:
            raise ValueError("response_bin_strategy must be one of: quantile, uniform.")
        if self.perturbation_mode not in {"local_std", "difference", "none"}:
            raise ValueError("perturbation_mode must be one of: local_std, difference, none.")
        return X, y

    def _response_bins(self, y: np.ndarray) -> np.ndarray:
        if np.allclose(y.min(), y.max()):
            return np.zeros_like(y, dtype=int)
        if self.response_bin_strategy == "uniform":
            edges = np.linspace(y.min(), y.max(), self.response_bins + 1)
        else:
            edges = np.quantile(y, np.linspace(0.0, 1.0, self.response_bins + 1))
        edges = np.unique(edges)
        if edges.size <= 2:
            return np.zeros_like(y, dtype=int)
        return np.clip(np.digitize(y, edges[1:-1], right=True), 0, edges.size - 2)

    def _anchor_weights(self, bins: np.ndarray) -> np.ndarray:
        counts = np.bincount(bins)
        weights = 1.0 / counts[bins]
        return weights / weights.sum()

    def _local_neighbors(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        y_scaled = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
        if self.neighbor_space == "spectrum":
            joint = X_scaled
        elif self.neighbor_space == "response":
            joint = y_scaled[:, None]
        else:
            joint = np.column_stack([float(self.spectrum_weight) * X_scaled, y_scaled])
        k = min(self.neighbors + 1, X.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k).fit(joint)
        indices = nbrs.kneighbors(joint, return_distance=False)
        return [row[row != idx] for idx, row in enumerate(indices)]

    def _local_perturbation(
        self,
        X: np.ndarray,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.noise_scale <= 0 or candidates.size < 2 or self.perturbation_mode == "none":
            return np.zeros(X.shape[1], dtype=float)
        if self.perturbation_mode == "difference":
            a, b = rng.choice(candidates, size=2, replace=False)
            return rng.normal(0.0, self.noise_scale) * (X[a] - X[b])
        local_std = X[candidates].std(axis=0, ddof=1)
        return rng.normal(0.0, self.noise_scale, size=X.shape[1]) * local_std

    def _consistency_report(
        self,
        x_new: np.ndarray,
        base_x: np.ndarray,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        base_y: float,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "_ConsistencyReport":
        angle = _spectral_angle(x_new, base_x)
        if angle > self.max_spectral_angle:
            return _ConsistencyReport(False, "spectral_angle", angle, np.nan, np.nan)

        corr = _derivative_corr(x_new, base_x)
        if corr < self.min_derivative_corr:
            return _ConsistencyReport(False, "derivative_corr", angle, corr, np.nan)

        low = np.minimum(parent_a, parent_b)
        high = np.maximum(parent_a, parent_b)
        span = np.maximum(high - low, np.std(np.vstack([parent_a, parent_b]), axis=0))
        margin = self.envelope_margin * (span + 1e-12)
        if np.any(x_new < low - margin) or np.any(x_new > high + margin):
            return _ConsistencyReport(False, "envelope", angle, corr, np.nan)

        response_deviation = np.nan
        if self.response_consistency:
            response_deviation = _knn_response_deviation(x_new, base_y, X, y, self.neighbors)
            if response_deviation > self.max_response_deviation:
                return _ConsistencyReport(
                    False,
                    "response_consistency",
                    angle,
                    corr,
                    response_deviation,
                )

        return _ConsistencyReport(True, "accepted", angle, corr, response_deviation)


def _spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    cosine = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.arccos(cosine))


def _derivative_corr(a: np.ndarray, b: np.ndarray) -> float:
    da = np.diff(a)
    db = np.diff(b)
    da = da - da.mean()
    db = db - db.mean()
    denom = np.linalg.norm(da) * np.linalg.norm(db)
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(da, db) / denom)


def _knn_response_deviation(
    x_new: np.ndarray,
    y_new: float,
    X: np.ndarray,
    y: np.ndarray,
    neighbors: int,
) -> float:
    k = min(max(int(neighbors), 1), X.shape[0])
    distances = np.linalg.norm(X - x_new, axis=1)
    idx = np.argsort(distances)[:k]
    local_y = y[idx]
    predicted = float(np.mean(local_y))
    scale = float(np.std(y, ddof=1) + 1e-12)
    return abs(predicted - y_new) / scale


@dataclass(frozen=True)
class _ConsistencyReport:
    passed: bool
    reason: str
    spectral_angle: float
    derivative_corr: float
    response_deviation: float


@dataclass
class _AugmentationDiagnostics:
    rejections: Counter
    attempted_parent_y_gaps: list[float]
    accepted_parent_y_gaps: list[float]
    synthetic_y: list[float]
    accepted_angles: list[float]
    accepted_derivative_corrs: list[float]
    accepted_response_deviations: list[float]

    def __init__(self) -> None:
        self.rejections = Counter()
        self.attempted_parent_y_gaps = []
        self.accepted_parent_y_gaps = []
        self.synthetic_y = []
        self.accepted_angles = []
        self.accepted_derivative_corrs = []
        self.accepted_response_deviations = []

    def observe_attempt(
        self,
        parent_y_a: float,
        parent_y_b: float,
        synthetic_y: float,
        check: _ConsistencyReport,
    ) -> None:
        self.attempted_parent_y_gaps.append(abs(float(parent_y_a) - float(parent_y_b)))

    def observe_accept(self, synthetic_y: float, check: _ConsistencyReport) -> None:
        self.synthetic_y.append(float(synthetic_y))
        self.accepted_angles.append(float(check.spectral_angle))
        self.accepted_derivative_corrs.append(float(check.derivative_corr))
        if np.isfinite(check.response_deviation):
            self.accepted_response_deviations.append(float(check.response_deviation))

    def as_metadata(self) -> dict:
        return {
            "rejections": dict(self.rejections),
            "attempted_parent_y_gap_mean": _safe_mean(self.attempted_parent_y_gaps),
            "synthetic_y_min": _safe_min(self.synthetic_y),
            "synthetic_y_max": _safe_max(self.synthetic_y),
            "accepted_spectral_angle_mean": _safe_mean(self.accepted_angles),
            "accepted_derivative_corr_mean": _safe_mean(self.accepted_derivative_corrs),
            "accepted_response_deviation_mean": _safe_mean(self.accepted_response_deviations),
        }


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_min(values: list[float]) -> float | None:
    return float(np.min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None
