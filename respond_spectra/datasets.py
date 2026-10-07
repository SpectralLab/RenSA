"""Dataset loading helpers for tabular spectral files."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def load_spectrum_csv(
    path: str | Path,
    target_column: str = "y",
    spectral_prefix: str = "x",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CSV where rows are samples and columns are spectral variables.

    The coal heating-value file in ``data/coal_Q.csv`` follows this layout:
    ``x1`` ... ``x1500`` are NIR spectral variables and ``y`` is the response.
    The returned wavelength axis is the numeric suffix of each spectral column.
    """

    path = Path(path)
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        frame = _read_headerless_spectrum_csv(path, target_column, spectral_prefix)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r} in {path}.")

    spectral_columns = _spectral_columns(frame.columns, spectral_prefix)
    if not spectral_columns:
        raise ValueError(
            f"No spectral columns matching {spectral_prefix!r} + number were found in {path}."
        )

    X = frame.loc[:, spectral_columns].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float).reshape(-1)
    wavelengths = np.asarray([_column_index(name, spectral_prefix) for name in spectral_columns])

    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("Loaded X must be 2D and y must have one value per row.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{path} contains missing, infinite, or non-numeric spectral values.")

    return wavelengths.astype(float), X, y


def _read_headerless_spectrum_csv(
    path: Path,
    target_column: str,
    spectral_prefix: str,
) -> pd.DataFrame:
    """Read numeric spectra without a header row, using the final column as y."""

    frame = pd.read_csv(path, header=None, encoding="utf-8-sig")
    if frame.shape[1] < 2:
        return frame
    columns = [f"{spectral_prefix}{idx}" for idx in range(1, frame.shape[1])] + [target_column]
    frame.columns = columns
    return frame


def _spectral_columns(columns: pd.Index, spectral_prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(spectral_prefix)}(\d+)$", re.IGNORECASE)
    indexed_columns: list[tuple[int, str]] = []
    for column in columns:
        name = str(column)
        match = pattern.match(name)
        if match:
            indexed_columns.append((int(match.group(1)), name))
    return [name for _, name in sorted(indexed_columns)]


def _column_index(column: str, spectral_prefix: str) -> int:
    return int(str(column)[len(spectral_prefix) :])
