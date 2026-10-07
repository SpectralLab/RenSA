"""Common spectral preprocessing routines.

Input matrices use the common chemometric shape: samples x spectral variables.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.signal import savgol_filter


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array shaped as samples x variables.")
    return X


def snv(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply standard normal variate correction to each spectrum."""

    X = _as_2d_float(X)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    return (X - mean) / np.maximum(std, eps)


def msc(X: np.ndarray, reference: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    """Apply multiplicative scatter correction.

    Each spectrum is regressed against a reference spectrum and corrected by
    subtracting the intercept and dividing by the slope.
    """

    X = _as_2d_float(X)
    ref = X.mean(axis=0) if reference is None else np.asarray(reference, dtype=float)
    if ref.ndim != 1 or ref.shape[0] != X.shape[1]:
        raise ValueError("reference must be a 1D array with one value per spectral variable.")

    design = np.column_stack([np.ones_like(ref), ref])
    coef, *_ = np.linalg.lstsq(design, X.T, rcond=None)
    intercept = coef[0][:, None]
    slope = coef[1][:, None]
    return (X - intercept) / np.maximum(np.abs(slope), eps) * np.sign(slope)


def detrend(X: np.ndarray, order: int = 1) -> np.ndarray:
    """Remove constant or linear trend from each spectrum."""

    X = _as_2d_float(X)
    if order not in (0, 1):
        raise ValueError("Only constant and linear detrending are supported.")
    return scipy_detrend(X, axis=1, type="constant" if order == 0 else "linear")


def savgol(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing or derivative filtering along spectra."""

    X = _as_2d_float(X)
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd.")
    if window_length <= polyorder:
        raise ValueError("window_length must be larger than polyorder.")
    if window_length > X.shape[1]:
        raise ValueError("window_length cannot exceed the number of spectral variables.")
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
