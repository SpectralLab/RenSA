"""Model evaluation helpers for spectral regression."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .augment import ResponseDrivenAugmenter


def evaluate_plsr_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 15,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Select PLSR components by cross-validation and report RMSE/R2."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and y must have one value per row.")

    n_splits = min(cv, X.shape[0])
    if n_splits < 2:
        raise ValueError("At least two samples are required for cross-validation.")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    upper = min(max_components, X.shape[1], X.shape[0] - 1)
    best = None
    history = []

    for n_components in range(1, upper + 1):
        model = PLSRegression(n_components=n_components)
        pred = cross_val_predict(model, X, y, cv=splitter).reshape(-1)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        item = {"n_components": n_components, "rmse": float(rmse), "r2": float(r2)}
        history.append(item)
        if best is None or rmse < best["rmse"]:
            best = item

    return {"best": best, "history": history}


def evaluate_plsr_cv_with_train_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    augmenter: ResponseDrivenAugmenter,
    max_components: int = 15,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate PLSR with augmentation performed inside each training fold.

    Validation folds always contain only original measured samples. This avoids
    the leakage caused by generating synthetic samples from the full dataset
    before cross-validation.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and y must have one value per row.")

    n_splits = min(cv, X.shape[0])
    if n_splits < 2:
        raise ValueError("At least two samples are required for cross-validation.")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X))
    upper = min(max_components, X.shape[1], min(len(train) for train, _ in folds) - 1)
    if upper < 1:
        raise ValueError("Not enough training samples to fit a PLSR model.")

    augmented_folds = []
    fold_metadata = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
        result = fold_augmenter.fit_resample(X[train_idx], y[train_idx])
        augmented_folds.append((result.X, result.y, test_idx))
        fold_metadata.append(
            {
                "fold": fold_idx,
                "train_original": int(train_idx.size),
                "validation_original": int(test_idx.size),
                "synthetic_accepted": result.n_synthetic,
                "acceptance_rate": float(result.metadata["acceptance_rate"]),
                "rejections": result.metadata.get("rejections", {}),
                "synthetic_y_min": result.metadata.get("synthetic_y_min"),
                "synthetic_y_max": result.metadata.get("synthetic_y_max"),
                "accepted_spectral_angle_mean": result.metadata.get(
                    "accepted_spectral_angle_mean"
                ),
                "accepted_derivative_corr_mean": result.metadata.get(
                    "accepted_derivative_corr_mean"
                ),
            }
        )

    best = None
    history = []
    for n_components in range(1, upper + 1):
        pred = np.empty_like(y, dtype=float)
        for X_train_aug, y_train_aug, test_idx in augmented_folds:
            components = min(n_components, X_train_aug.shape[0] - 1, X_train_aug.shape[1])
            model = PLSRegression(n_components=components)
            model.fit(X_train_aug, y_train_aug)
            pred[test_idx] = model.predict(X[test_idx]).reshape(-1)

        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        item = {"n_components": n_components, "rmse": float(rmse), "r2": float(r2)}
        history.append(item)
        if best is None or rmse < best["rmse"]:
            best = item

    return {"best": best, "history": history, "folds": fold_metadata}


def default_tolerant_regressors(random_state: int = 42) -> dict:
    """Return nonlinear / regularized regressors for augmentation validation."""

    return {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "rbf_svr": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
        ),
    }


def default_tuned_tolerant_regressors(
    random_state: int = 42,
    inner_cv: int = 3,
) -> dict:
    """Return GridSearchCV-wrapped regressors for nested CV style evaluation."""

    inner = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    return {
        "extra_trees_tuned": GridSearchCV(
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            param_grid={
                "n_estimators": [300],
                "max_features": ["sqrt", 0.5, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 8, 16],
            },
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        ),
        "random_forest_tuned": GridSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            param_grid={
                "n_estimators": [300],
                "max_features": ["sqrt", 0.5, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 8, 16],
            },
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        ),
        "rbf_svr_tuned": GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            param_grid={
                "svr__C": [1.0, 10.0, 100.0],
                "svr__epsilon": [0.05, 0.1, 0.2],
                "svr__gamma": ["scale", 0.01, 0.1],
            },
            scoring="neg_root_mean_squared_error",
            cv=inner,
            n_jobs=1,
        ),
    }


def default_deep_regressors(
    random_state: int = 42,
    epochs: int = 80,
    device: str = "auto",
) -> dict:
    """Return compact 1D neural regressors for small spectral datasets."""

    from .deep import MiniResNet1DRegressor, Shallow1DCNNRegressor

    return {
        "shallow_1d_cnn": Shallow1DCNNRegressor(
            epochs=epochs,
            batch_size=32,
            patience=max(8, epochs // 5),
            random_state=random_state,
            device=device,
        ),
        "mini_resnet_1d": MiniResNet1DRegressor(
            epochs=epochs,
            batch_size=32,
            patience=max(8, epochs // 5),
            random_state=random_state,
            device=device,
        ),
    }


def evaluate_regressor_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    transformer=None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate a regression estimator by CV on original measured samples."""

    X, y, splitter = _validated_cv_inputs(X, y, cv, random_state)
    if transformer is None:
        pred = cross_val_predict(clone(estimator), X, y, cv=splitter).reshape(-1)
    else:
        pred = np.empty_like(y, dtype=float)
        for train_idx, test_idx in splitter.split(X):
            fold_transformer = clone(transformer)
            X_train = fold_transformer.fit_transform(X[train_idx], y[train_idx])
            X_test = fold_transformer.transform(X[test_idx])
            model = clone(estimator)
            model.fit(X_train, y[train_idx])
            pred[test_idx] = model.predict(X_test).reshape(-1)
    return _regression_scores(y, pred)


def evaluate_regressor_cv_with_train_augmentation(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    augmenter: ResponseDrivenAugmenter,
    transformer=None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate an estimator with synthetic spectra generated only in train folds."""

    X, y, splitter = _validated_cv_inputs(X, y, cv, random_state)
    pred = np.empty_like(y, dtype=float)
    fold_metadata = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        if transformer is None:
            X_train = X[train_idx]
            X_test = X[test_idx]
        else:
            fold_transformer = clone(transformer)
            X_train = fold_transformer.fit_transform(X[train_idx], y[train_idx])
            X_test = fold_transformer.transform(X[test_idx])

        fold_augmenter = _clone_augmenter_for_fold(augmenter, fold_idx)
        result = fold_augmenter.fit_resample(X_train, y[train_idx])
        model = clone(estimator)
        model.fit(result.X, result.y)
        pred[test_idx] = model.predict(X_test).reshape(-1)
        fold_metadata.append(
            {
                "fold": fold_idx,
                "train_original": int(train_idx.size),
                "validation_original": int(test_idx.size),
                "synthetic_accepted": result.n_synthetic,
                "acceptance_rate": float(result.metadata["acceptance_rate"]),
            }
        )

    scores = _regression_scores(y, pred)
    scores["folds"] = fold_metadata
    return scores


def compare_regressors_with_train_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    augmenter: ResponseDrivenAugmenter,
    regressors: dict | None = None,
    transformer=None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Compare original-only CV against train-fold augmentation for regressors."""

    models = default_tolerant_regressors(random_state) if regressors is None else regressors
    comparison = {}
    for name, estimator in models.items():
        baseline = evaluate_regressor_cv(
            estimator,
            X,
            y,
            transformer=transformer,
            cv=cv,
            random_state=random_state,
        )
        augmented = evaluate_regressor_cv_with_train_augmentation(
            estimator,
            X,
            y,
            augmenter,
            transformer=transformer,
            cv=cv,
            random_state=random_state,
        )
        comparison[name] = {
            "baseline": baseline,
            "fold_train_augmented": augmented,
            "delta_rmse": augmented["rmse"] - baseline["rmse"],
            "delta_r2": augmented["r2"] - baseline["r2"],
        }
    return comparison


def _validated_cv_inputs(
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, KFold]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and y must have one value per row.")
    n_splits = min(cv, X.shape[0])
    if n_splits < 2:
        raise ValueError("At least two samples are required for cross-validation.")
    return X, y, KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _clone_augmenter_for_fold(
    augmenter: ResponseDrivenAugmenter,
    fold_idx: int,
) -> ResponseDrivenAugmenter:
    params = augmenter.__dict__.copy()
    if augmenter.random_state is not None:
        params["random_state"] = int(augmenter.random_state) + fold_idx
    return ResponseDrivenAugmenter(**params)
