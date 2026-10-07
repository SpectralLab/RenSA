from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from respond_spectra import (
    CARSFeatureSelector,
    CorrelationFeatureSelector,
    CorrelationIntervalSelector,
    ResponseDrivenAugmenter,
    compare_regressors_with_train_augmentation,
    default_deep_regressors,
    default_tolerant_regressors,
    default_tuned_tolerant_regressors,
    evaluate_plsr_cv,
    evaluate_plsr_cv_with_train_augmentation,
    load_spectrum_csv,
    savgol,
    msc,
    snv,
)


def make_synthetic_coal_spectra(n_samples: int = 36, n_variables: int = 180, seed: int = 7):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(1000.0, 2500.0, n_variables)
    y = np.sort(rng.uniform(15.0, 31.0, size=n_samples))

    peaks = [
        (1220.0, 45.0, 0.45),
        (1480.0, 75.0, -0.30),
        (1920.0, 55.0, 0.52),
        (2240.0, 90.0, -0.18),
    ]
    X = []
    for value in y:
        spectrum = 0.15 + 0.00008 * (wavelengths - wavelengths.mean())
        response_factor = (value - y.mean()) / y.std(ddof=1)
        for center, width, weight in peaks:
            spectrum += (1.0 + weight * response_factor) * np.exp(
                -0.5 * ((wavelengths - center) / width) ** 2
            )
        spectrum += rng.normal(0.0, 0.018, size=n_variables)
        X.append(spectrum)
    return wavelengths, np.asarray(X), y


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "data" / "coal_Q.csv"
    parser = argparse.ArgumentParser(
        description="Run response-driven augmentation on coal NIR heating-value spectra."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="CSV with x1...xN spectral columns and a y response column.",
    )
    parser.add_argument(
        "--target-column",
        default="y",
        help="Response column name. The coal_Q.csv file uses y.",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=200,
        help="Number of synthetic spectrum-response pairs to request.",
    )
    parser.add_argument(
        "--neighbor-space",
        choices=["joint", "spectrum", "response"],
        default="joint",
        help="Space used to choose parent neighbors for synthetic spectra.",
    )
    parser.add_argument(
        "--spectrum-weight",
        type=float,
        default=0.35,
        help="Spectral block weight when --neighbor-space joint is used.",
    )
    parser.add_argument(
        "--perturbation-mode",
        choices=["local_std", "difference", "none"],
        default="local_std",
        help="How to perturb interpolated synthetic spectra.",
    )
    parser.add_argument(
        "--response-consistency",
        action="store_true",
        help="Reject synthetic spectra whose local spectral neighbors imply inconsistent y.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["raw", "snv", "sg", "snv-sg", "snv-sg1", "msc", "msc-sg"],
        default="raw",
        help="Spectral preprocessing before modeling. raw is often a strong baseline for this coal_Q data.",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "corr-topk", "corr-interval", "cars"],
        default="none",
        help="Feature selector fitted inside each training fold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Number of variables for --feature-selection corr-topk.",
    )
    parser.add_argument(
        "--n-intervals",
        type=int,
        default=30,
        help="Number of spectral intervals for --feature-selection corr-interval.",
    )
    parser.add_argument(
        "--select-intervals",
        type=int,
        default=10,
        help="Number of intervals retained for --feature-selection corr-interval.",
    )
    parser.add_argument(
        "--cars-sampling",
        type=int,
        default=40,
        help="Monte Carlo sampling iterations for --feature-selection cars.",
    )
    parser.add_argument(
        "--cars-min-features",
        type=int,
        default=30,
        help="Minimum number of variables retained by CARS.",
    )
    parser.add_argument(
        "--cars-components",
        type=int,
        default=10,
        help="Maximum PLS components used inside CARS.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=15,
        help="Maximum PLSR components to evaluate by cross-validation.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--use-synthetic-demo",
        action="store_true",
        help="Ignore --data and run the old generated-data smoke demo.",
    )
    parser.add_argument(
        "--skip-tolerant-models",
        action="store_true",
        help="Only run the PLSR diagnostic and skip tree/SVR validation models.",
    )
    parser.add_argument(
        "--tune-models",
        action="store_true",
        help="Use inner-CV GridSearchCV versions of RF/ExtraTrees/SVR.",
    )
    parser.add_argument(
        "--inner-cv",
        type=int,
        default=3,
        help="Inner CV folds for --tune-models.",
    )
    parser.add_argument(
        "--include-deep-models",
        action="store_true",
        help="Also run Shallow 1D-CNN and mini 1D-ResNet validation models.",
    )
    parser.add_argument(
        "--deep-epochs",
        type=int,
        default=80,
        help="Training epochs for deep spectral models when --include-deep-models is set.",
    )
    parser.add_argument(
        "--deep-device",
        default="auto",
        help="Torch device for deep models: auto, cpu, mps, or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_synthetic_demo:
        wavelengths, X, y = make_synthetic_coal_spectra()
    else:
        wavelengths, X, y = load_spectrum_csv(args.data, target_column=args.target_column)

    X_pre = preprocess_spectra(X, args.preprocess)
    transformer = build_feature_selector(args)

    baseline = evaluate_plsr_cv(X_pre, y, max_components=args.max_components, cv=args.cv)

    augmenter = ResponseDrivenAugmenter(
        n_synthetic=args.n_synthetic,
        response_bins=6,
        neighbors=5,
        noise_scale=0.008,
        random_state=42,
        neighbor_space=args.neighbor_space,
        spectrum_weight=args.spectrum_weight,
        perturbation_mode=args.perturbation_mode,
        response_consistency=args.response_consistency,
    )
    result = augmenter.fit_resample(X_pre, y)

    fold_augmented = evaluate_plsr_cv_with_train_augmentation(
        X_pre,
        y,
        augmenter,
        max_components=args.max_components,
        cv=args.cv,
    )
    tolerant_comparison = {}
    if not args.skip_tolerant_models:
        regressors = (
            default_tuned_tolerant_regressors(random_state=42, inner_cv=args.inner_cv)
            if args.tune_models
            else default_tolerant_regressors(random_state=42)
        )
        tolerant_comparison = compare_regressors_with_train_augmentation(
            X_pre,
            y,
            augmenter,
            regressors=regressors,
            transformer=transformer,
            cv=args.cv,
            random_state=42,
        )
    deep_comparison = {}
    if args.include_deep_models:
        deep_comparison = compare_regressors_with_train_augmentation(
            X_pre,
            y,
            augmenter,
            regressors=default_deep_regressors(
                random_state=42,
                epochs=args.deep_epochs,
                device=args.deep_device,
            ),
            transformer=transformer,
            cv=args.cv,
            random_state=42,
        )

    print("Spectral variables:", X.shape[1])
    print("Spectral axis:", f"{wavelengths[0]:g} ... {wavelengths[-1]:g}")
    print("Original samples:", X.shape[0])
    print("Response range:", f"{y.min():.4g} ... {y.max():.4g}")
    print("Preprocess:", args.preprocess)
    print("Feature selection:", args.feature_selection)
    print("Full-data synthetic accepted:", result.n_synthetic)
    print("Full-data acceptance rate:", round(result.metadata["acceptance_rate"], 3))
    print("Augmentation diagnostics:", result.metadata)
    print("PLSR diagnostic baseline:", baseline["best"])
    print("PLSR diagnostic fold-train augmented:", fold_augmented["best"])
    for name, scores in tolerant_comparison.items():
        print(f"{name} baseline:", scores["baseline"])
        print(f"{name} fold-train augmented:", scores["fold_train_augmented"])
        print(
            f"{name} delta:",
            {
                "delta_rmse": scores["delta_rmse"],
                "delta_r2": scores["delta_r2"],
            },
        )
    for name, scores in deep_comparison.items():
        print(f"{name} baseline:", scores["baseline"])
        print(f"{name} fold-train augmented:", scores["fold_train_augmented"])
        print(
            f"{name} delta:",
            {
                "delta_rmse": scores["delta_rmse"],
                "delta_r2": scores["delta_r2"],
            },
        )
    print("Fold augmentation:", fold_augmented["folds"])


def preprocess_spectra(X: np.ndarray, method: str) -> np.ndarray:
    window_length = min(21, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
    if method == "raw":
        return X
    if method == "snv":
        return snv(X)
    if method == "sg":
        return savgol(X, window_length=window_length, polyorder=2)
    if method == "snv-sg":
        return savgol(snv(X), window_length=window_length, polyorder=2)
    if method == "snv-sg1":
        return savgol(snv(X), window_length=window_length, polyorder=2, deriv=1)
    if method == "msc":
        return msc(X)
    if method == "msc-sg":
        return savgol(msc(X), window_length=window_length, polyorder=2)
    raise ValueError(f"Unknown preprocessing method: {method}")


def build_feature_selector(args):
    if args.feature_selection == "none":
        return None
    if args.feature_selection == "corr-topk":
        return CorrelationFeatureSelector(k=args.top_k)
    if args.feature_selection == "corr-interval":
        return CorrelationIntervalSelector(
            n_intervals=args.n_intervals,
            n_select=args.select_intervals,
        )
    if args.feature_selection == "cars":
        return CARSFeatureSelector(
            n_sampling=args.cars_sampling,
            min_features=args.cars_min_features,
            max_components=args.cars_components,
            cv=min(args.cv, 5),
            random_state=42,
        )
    raise ValueError(f"Unknown feature selection method: {args.feature_selection}")


if __name__ == "__main__":
    main()
