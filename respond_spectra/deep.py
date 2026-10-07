"""Small PyTorch regressors for one-dimensional spectra."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class _TrainingState:
    best_loss: float
    best_state: dict | None
    stale_epochs: int


class _BaseTorchSpectrumRegressor(BaseEstimator, RegressorMixin):
    """Shared sklearn-style wrapper for small 1D spectral networks."""

    def __init__(
        self,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        validation_fraction: float = 0.15,
        patience: int = 12,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        _require_torch()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("X must be 2D and y must have one value per row.")

        self._set_seed()
        self.n_features_in_ = int(X.shape[1])
        self.x_mean_ = X.mean(axis=0, keepdims=True)
        self.x_std_ = np.maximum(X.std(axis=0, keepdims=True), 1e-6)
        self.y_mean_ = y.mean(axis=0, keepdims=True)
        self.y_std_ = np.maximum(y.std(axis=0, keepdims=True), 1e-6)

        X_scaled = (X - self.x_mean_) / self.x_std_
        y_scaled = (y - self.y_mean_) / self.y_std_

        device = self._resolved_device()
        self.model_ = self._build_model(self.n_features_in_).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        train_idx, valid_idx = self._train_valid_indices(X_scaled.shape[0])
        train_loader = self._loader(X_scaled[train_idx], y_scaled[train_idx], shuffle=True)
        valid_loader = None
        if valid_idx.size:
            valid_loader = self._loader(X_scaled[valid_idx], y_scaled[valid_idx], shuffle=False)

        state = _TrainingState(best_loss=float("inf"), best_state=None, stale_epochs=0)
        self.history_ = []
        for epoch in range(int(self.epochs)):
            train_loss = self._train_epoch(train_loader, optimizer, loss_fn, device)
            valid_loss = self._eval_loss(valid_loader, loss_fn, device) if valid_loader else train_loss
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

            if valid_loss < state.best_loss - 1e-6:
                state.best_loss = valid_loss
                state.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                state.stale_epochs = 0
            else:
                state.stale_epochs += 1
                if state.stale_epochs >= int(self.patience):
                    break

        if state.best_state is not None:
            self.model_.load_state_dict(state.best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        _require_torch()
        if not hasattr(self, "model_"):
            raise ValueError("This estimator is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the same number of spectral variables as fit.")

        X_scaled = (X - self.x_mean_) / self.x_std_
        device = self._resolved_device()
        tensor = torch.as_tensor(X_scaled[:, None, :], dtype=torch.float32, device=device)
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, tensor.shape[0], int(self.batch_size)):
                batch = tensor[start : start + int(self.batch_size)]
                preds.append(self.model_(batch).detach().cpu().numpy())
        y_scaled = np.vstack(preds)
        return (y_scaled * self.y_std_ + self.y_mean_).reshape(-1)

    def _build_model(self, n_features: int):
        raise NotImplementedError

    def _set_seed(self) -> None:
        seed = int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolved_device(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def _train_valid_indices(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(self.random_state))
        indices = rng.permutation(n_samples)
        n_valid = int(round(n_samples * float(self.validation_fraction)))
        if n_samples < 8:
            n_valid = 0
        else:
            n_valid = min(max(n_valid, 1), n_samples - 2)
        return indices[n_valid:], indices[:n_valid]

    def _loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool):
        dataset = TensorDataset(
            torch.as_tensor(X[:, None, :], dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=int(self.batch_size), shuffle=shuffle)

    def _train_epoch(self, loader, optimizer, loss_fn, device) -> float:
        self.model_.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model_(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * batch_x.shape[0]
            count += batch_x.shape[0]
        return total / max(count, 1)

    def _eval_loss(self, loader, loss_fn, device) -> float:
        self.model_.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = loss_fn(self.model_(batch_x), batch_y)
                total += float(loss.detach().cpu()) * batch_x.shape[0]
                count += batch_x.shape[0]
        return total / max(count, 1)


class Shallow1DCNNRegressor(_BaseTorchSpectrumRegressor):
    """Compact 1D-CNN for small-sample spectral regression."""

    def __init__(
        self,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        validation_fraction: float = 0.15,
        patience: int = 12,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
        dropout: float = 0.2,
        pool_bins: int = 16,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            patience=patience,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        self.dropout = dropout
        self.pool_bins = pool_bins

    def _build_model(self, n_features: int):
        return _Shallow1DCNN(dropout=float(self.dropout), pool_bins=int(self.pool_bins))


class MiniResNet1DRegressor(_BaseTorchSpectrumRegressor):
    """Small residual 1D-CNN for spectral regression."""

    def __init__(
        self,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 8e-4,
        weight_decay: float = 1e-3,
        validation_fraction: float = 0.15,
        patience: int = 12,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
        dropout: float = 0.3,
        pool_bins: int = 16,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            patience=patience,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        self.dropout = dropout
        self.pool_bins = pool_bins

    def _build_model(self, n_features: int):
        return _MiniResNet1D(dropout=float(self.dropout), pool_bins=int(self.pool_bins))


class _Shallow1DCNN(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_bins),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * pool_bins, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


class _MiniResNet1D(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock1D(32, 32),
            _ResidualBlock1D(32, 32),
            _ResidualBlock1D(32, 64, stride=2),
            _ResidualBlock1D(64, 64),
            nn.AdaptiveAvgPool1d(pool_bins),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * pool_bins, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regressor(self.blocks(self.stem(x)))


class _ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for deep spectral regressors.") from _TORCH_IMPORT_ERROR
