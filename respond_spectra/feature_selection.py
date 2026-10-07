"""Feature selection utilities for spectral regression."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """Select individual variables with the largest absolute response correlation."""

    def __init__(self, k: int = 300):
        self.k = k

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        scores = _absolute_correlations(X, y)
        k = min(max(int(self.k), 1), X.shape[1])
        selected = np.argsort(scores)[-k:]
        self.selected_indices_ = np.sort(selected)
        self.scores_ = scores
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]


class CorrelationIntervalSelector(BaseEstimator, TransformerMixin):
    """Select contiguous spectral intervals by response correlation.

    This is more suitable for 1D-CNN/ResNet than scattered top-k variables,
    because the selected input still preserves local spectral neighborhoods.
    """

    def __init__(self, n_intervals: int = 30, n_select: int = 10):
        self.n_intervals = n_intervals
        self.n_select = n_select

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        intervals = np.array_split(np.arange(X.shape[1]), int(self.n_intervals))
        scores = _absolute_correlations(X, y)
        interval_scores = np.asarray([float(np.nanmean(scores[idx])) for idx in intervals])
        n_select = min(max(int(self.n_select), 1), len(intervals))
        selected_intervals = np.sort(np.argsort(interval_scores)[-n_select:])

        self.interval_scores_ = interval_scores
        self.selected_intervals_ = selected_intervals
        self.selected_indices_ = np.concatenate([intervals[i] for i in selected_intervals])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]


class CARSFeatureSelector(BaseEstimator, TransformerMixin):
    """Competitive adaptive reweighted sampling for spectral variables.

    The selector is designed to be fitted inside each training fold. It performs
    Monte Carlo sample resampling, uses PLS coefficients as variable weights,
    gradually reduces the retained variable count, and keeps the subset with
    the lowest inner-CV PLS RMSE.
    """

    def __init__(
        self,
        n_sampling: int = 50,
        sample_ratio: float = 0.8,
        max_components: int = 10,
        cv: int = 5,
        min_features: int = 20,
        random_state: int = 42,
    ):
        self.n_sampling = n_sampling
        self.sample_ratio = sample_ratio
        self.max_components = max_components
        self.cv = cv
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        rng = np.random.default_rng(int(self.random_state))
        n_samples, n_features = X.shape
        n_sampling = max(int(self.n_sampling), 1)
        min_features = min(max(int(self.min_features), 1), n_features)
        sample_size = min(max(int(round(n_samples * float(self.sample_ratio))), 2), n_samples)

        current = np.arange(n_features)
        candidates: list[np.ndarray] = []
        candidate_rmse: list[float] = []

        for iteration in range(n_sampling):
            if current.size <= min_features:
                subset = np.sort(current.copy())
            else:
                row_idx = rng.choice(n_samples, size=sample_size, replace=False)
                coef = _pls_abs_coefficients(
                    X[row_idx][:, current],
                    y[row_idx],
                    max_components=int(self.max_components),
                )
                retain_count = _cars_retain_count(
                    n_features=n_features,
                    min_features=min_features,
                    iteration=iteration,
                    n_sampling=n_sampling,
                )
                retain_count = min(retain_count, current.size)
                weights = coef / np.maximum(coef.sum(), 1e-12)
                if np.isfinite(weights).all() and weights.sum() > 0:
                    picked = rng.choice(current, size=retain_count, replace=False, p=weights)
                else:
                    picked = rng.choice(current, size=retain_count, replace=False)
                subset = np.sort(picked)
                current = subset

            rmse = _pls_cv_rmse(
                X[:, subset],
                y,
                max_components=int(self.max_components),
                cv=int(self.cv),
                random_state=int(self.random_state),
            )
            candidates.append(subset)
            candidate_rmse.append(rmse)

        best_idx = int(np.argmin(candidate_rmse))
        self.selected_indices_ = candidates[best_idx]
        self.rmse_path_ = np.asarray(candidate_rmse, dtype=float)
        self.n_features_path_ = np.asarray([idx.size for idx in candidates], dtype=int)
        self.best_iteration_ = best_idx
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]


def _validate_xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and y must have one value per row.")
    return X, y


def _absolute_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = np.abs(X_centered.T @ y_centered)
    denominator = np.linalg.norm(X_centered, axis=0) * np.linalg.norm(y_centered)
    return numerator / np.maximum(denominator, 1e-12)


def _cars_retain_count(
    n_features: int,
    min_features: int,
    iteration: int,
    n_sampling: int,
) -> int:
    if n_sampling <= 1:
        return min_features
    ratio = (min_features / n_features) ** (iteration / (n_sampling - 1))
    return max(int(round(n_features * ratio)), min_features)


def _pls_abs_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
) -> np.ndarray:
    upper = min(max_components, X.shape[0] - 1, X.shape[1])
    if upper < 1:
        return np.ones(X.shape[1], dtype=float)
    model = PLSRegression(n_components=upper)
    model.fit(X, y)
    coef = np.abs(np.asarray(model.coef_).reshape(-1))
    if coef.shape[0] != X.shape[1]:
        coef = coef[: X.shape[1]]
    return coef + 1e-12


def _pls_cv_rmse(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    cv: int,
    random_state: int,
) -> float:
    n_splits = min(cv, X.shape[0])
    if n_splits < 2:
        return float("inf")
    upper = min(max_components, X.shape[0] - 1, X.shape[1])
    if upper < 1:
        return float("inf")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = PLSRegression(n_components=upper)
    pred = cross_val_predict(model, X, y, cv=splitter).reshape(-1)
    return float(np.sqrt(mean_squared_error(y, pred)))
