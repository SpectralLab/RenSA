"""Response-driven spectral augmentation tools."""

from .augment import AugmentationResult, ResponseDrivenAugmenter
from .datasets import load_spectrum_csv
from .evaluation import (
    compare_regressors_with_train_augmentation,
    default_deep_regressors,
    default_tolerant_regressors,
    default_tuned_tolerant_regressors,
    evaluate_plsr_cv,
    evaluate_plsr_cv_with_train_augmentation,
    evaluate_regressor_cv,
    evaluate_regressor_cv_with_train_augmentation,
)
from .feature_selection import (
    CARSFeatureSelector,
    CorrelationFeatureSelector,
    CorrelationIntervalSelector,
)
from .preprocessing import detrend, msc, savgol, snv

__all__ = [
    "AugmentationResult",
    "ResponseDrivenAugmenter",
    "detrend",
    "evaluate_plsr_cv",
    "evaluate_plsr_cv_with_train_augmentation",
    "compare_regressors_with_train_augmentation",
    "CARSFeatureSelector",
    "CorrelationFeatureSelector",
    "CorrelationIntervalSelector",
    "default_deep_regressors",
    "default_tolerant_regressors",
    "default_tuned_tolerant_regressors",
    "evaluate_regressor_cv",
    "evaluate_regressor_cv_with_train_augmentation",
    "load_spectrum_csv",
    "msc",
    "savgol",
    "snv",
]
