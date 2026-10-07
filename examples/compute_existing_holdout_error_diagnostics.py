from __future__ import annotations

"""Compute holdout error diagnostics from existing prediction vectors only.

This script deliberately does not import or call any splitting, preprocessing,
CARS, RenSA, augmentation, tuning, or model-training code. It reads the locked
outer-SPXY holdout prediction CSVs used for the manuscript's Table 5, verifies
their provenance against companion summaries and the displayed Table 5 values,
and then calculates RMSEP, Bias, prediction slope, SEP, and the RMSEP error
decomposition.
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Endpoint:
    task: str
    dataset: str
    predictions: str
    summary: str
    baseline_column: str
    rensa_column: str
    summary_baseline_rmsep: str
    summary_rensa_rmsep: str
    summary_baseline_r2: str
    summary_rensa_r2: str
    expected_n: int
    table5_baseline_rmsep: float
    table5_rensa_rmsep: float
    table5_baseline_r2: float
    table5_rensa_r2: float
    table5_delta_r2: float


ENDPOINTS = (
    Endpoint(
        task="GCV",
        dataset="coal_Q",
        predictions="results/holdout_unified_trainonly_primary_predictions.csv",
        summary="results/holdout_unified_trainonly_primary.csv",
        baseline_column="baseline_pred",
        rensa_column="selected_pred",
        summary_baseline_rmsep="baseline_test_rmse",
        summary_rensa_rmsep="selected_test_rmse",
        summary_baseline_r2="baseline_test_r2",
        summary_rensa_r2="selected_test_r2",
        expected_n=33,
        table5_baseline_rmsep=0.92,
        table5_rensa_rmsep=0.84,
        table5_baseline_r2=0.88,
        table5_rensa_r2=0.90,
        table5_delta_r2=0.0187,
    ),
    Endpoint(
        task="Ash",
        dataset="coal_ash",
        predictions="results/holdout_unified_trainonly_coal_ash_raw_predictions.csv",
        summary="results/holdout_unified_trainonly_coal_ash_raw.csv",
        baseline_column="baseline_pred",
        rensa_column="selected_pred",
        summary_baseline_rmsep="baseline_test_rmse",
        summary_rensa_rmsep="selected_test_rmse",
        summary_baseline_r2="baseline_test_r2",
        summary_rensa_r2="selected_test_r2",
        expected_n=33,
        table5_baseline_rmsep=3.28,
        table5_rensa_rmsep=2.29,
        table5_baseline_r2=0.85,
        table5_rensa_r2=0.93,
        table5_delta_r2=0.0746,
    ),
    Endpoint(
        task="SOC",
        dataset="soil_SOC",
        predictions="results/final_soil_snvsg_spxy25_locked_protocol_predictions.csv",
        summary="results/final_soil_snvsg_spxy25_locked_protocol_summary.csv",
        baseline_column="baseline_pred",
        rensa_column="augmented_pred",
        summary_baseline_rmsep="baseline_rmse",
        summary_rensa_rmsep="augmented_rmse",
        summary_baseline_r2="baseline_r2",
        summary_rensa_r2="augmented_r2",
        expected_n=21,
        table5_baseline_rmsep=0.19,
        table5_rensa_rmsep=0.17,
        table5_baseline_r2=0.82,
        table5_rensa_r2=0.86,
        table5_delta_r2=0.0384,
    ),
    Endpoint(
        task="CN",
        dataset="diesel_CN",
        predictions="results/final_diesel_cn_snvsg_n80_none_for_violin_predictions.csv",
        summary="results/final_diesel_cn_snvsg_n80_none_for_violin_summary.csv",
        baseline_column="baseline_pred",
        rensa_column="augmented_pred",
        summary_baseline_rmsep="baseline_rmse",
        summary_rensa_rmsep="augmented_rmse",
        summary_baseline_r2="baseline_r2",
        summary_rensa_r2="augmented_r2",
        expected_n=95,
        table5_baseline_rmsep=1.74,
        table5_rensa_rmsep=1.61,
        table5_baseline_r2=0.58,
        table5_rensa_r2=0.64,
        table5_delta_r2=0.0596,
    ),
    Endpoint(
        task="FREEZE",
        dataset="diesel_FREEZE",
        predictions="results/final_diesel_freeze_snv_cars_n40_localstd_for_violin_predictions.csv",
        summary="results/final_diesel_freeze_snv_cars_n40_localstd_for_violin_summary.csv",
        baseline_column="baseline_pred",
        rensa_column="augmented_pred",
        summary_baseline_rmsep="baseline_rmse",
        summary_rensa_rmsep="augmented_rmse",
        summary_baseline_r2="baseline_r2",
        summary_rensa_r2="augmented_r2",
        expected_n=99,
        table5_baseline_rmsep=2.90,
        table5_rensa_rmsep=2.83,
        table5_baseline_r2=0.92,
        table5_rensa_r2=0.92,
        table5_delta_r2=0.0036,
    ),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compute Bias, Slope, and SEP from existing Table 5 holdout predictions."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "holdout_error_diagnostics",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmsep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum = float(np.sum(np.square(y_pred - y_true)))
    total_sum = float(np.sum(np.square(y_true - np.mean(y_true))))
    return 1.0 - residual_sum / total_sum


def diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    n = len(residual)
    bias = float(np.mean(residual))
    sep = float(np.sqrt(np.sum(np.square(residual - bias)) / (n - 1)))
    design = np.column_stack((np.ones(n, dtype=float), y_true))
    intercept, slope = np.linalg.lstsq(design, y_pred, rcond=None)[0]
    value_rmsep = rmsep(y_true, y_pred)
    decomposition = ((n - 1) / n) * sep**2 + bias**2
    return {
        "RMSEP": value_rmsep,
        "Bias": bias,
        "Slope": float(slope),
        "Intercept": float(intercept),
        "SEP": sep,
        "RMSEP_squared": value_rmsep**2,
        "decomposition_value": decomposition,
        "absolute_difference": abs(value_rmsep**2 - decomposition),
    }


def one_summary_row(frame: pd.DataFrame, endpoint: Endpoint) -> pd.Series:
    if "dataset" in frame.columns:
        frame = frame.loc[frame["dataset"].eq(endpoint.dataset)]
    if len(frame) != 1:
        raise ValueError(
            f"Expected one summary row for {endpoint.task}, found {len(frame)}"
        )
    return frame.iloc[0]


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label}: expected {expected:.16g}, found {actual:.16g}")


def verify_table5_rounding(
    endpoint: Endpoint,
    baseline_rmsep: float,
    rensa_rmsep: float,
    baseline_r2: float,
    rensa_r2: float,
) -> None:
    displayed = (
        (round(baseline_rmsep, 2), endpoint.table5_baseline_rmsep, "baseline RMSEP"),
        (round(rensa_rmsep, 2), endpoint.table5_rensa_rmsep, "RenSA RMSEP"),
        (round(baseline_r2, 2), endpoint.table5_baseline_r2, "baseline R2"),
        (round(rensa_r2, 2), endpoint.table5_rensa_r2, "RenSA R2"),
        (round(rensa_r2 - baseline_r2, 4), endpoint.table5_delta_r2, "delta R2"),
    )
    for actual, expected, name in displayed:
        assert_close(actual, expected, f"{endpoint.task} Table 5 {name}", atol=5e-12)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    result_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []

    for endpoint in ENDPOINTS:
        prediction_path = project_root / endpoint.predictions
        summary_path = project_root / endpoint.summary
        if not prediction_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing locked input for {endpoint.task}: "
                f"{prediction_path} or {summary_path}"
            )

        frame = pd.read_csv(prediction_path)
        if "dataset" in frame.columns:
            frame = frame.loc[frame["dataset"].eq(endpoint.dataset)].copy()
        required = {"y_true", endpoint.baseline_column, endpoint.rensa_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
        if len(frame) != endpoint.expected_n:
            raise ValueError(
                f"{endpoint.task}: expected {endpoint.expected_n} holdout rows, found {len(frame)}"
            )
        if "sample_index" in frame and frame["sample_index"].duplicated().any():
            raise ValueError(f"{endpoint.task}: duplicate holdout sample_index values")

        arrays = frame[["y_true", endpoint.baseline_column, endpoint.rensa_column]].to_numpy(
            dtype=float
        )
        if not np.isfinite(arrays).all():
            raise ValueError(f"{endpoint.task}: non-finite prediction values")
        y_true = arrays[:, 0]
        baseline_pred = arrays[:, 1]
        rensa_pred = arrays[:, 2]

        baseline_rmsep = rmsep(y_true, baseline_pred)
        rensa_rmsep = rmsep(y_true, rensa_pred)
        baseline_r2 = r2_score(y_true, baseline_pred)
        rensa_r2 = r2_score(y_true, rensa_pred)

        summary = one_summary_row(pd.read_csv(summary_path), endpoint)
        if str(summary["split_method"]).lower() != "spxy":
            raise ValueError(f"{endpoint.task}: companion summary is not an SPXY split")
        assert_close(float(summary["test_size"]), 0.25, f"{endpoint.task} test_size")
        assert_close(float(summary["test_count"]), endpoint.expected_n, f"{endpoint.task} test_count")
        assert_close(
            baseline_rmsep,
            float(summary[endpoint.summary_baseline_rmsep]),
            f"{endpoint.task} baseline RMSEP vs summary",
        )
        assert_close(
            rensa_rmsep,
            float(summary[endpoint.summary_rensa_rmsep]),
            f"{endpoint.task} RenSA RMSEP vs summary",
        )
        assert_close(
            baseline_r2,
            float(summary[endpoint.summary_baseline_r2]),
            f"{endpoint.task} baseline R2 vs summary",
        )
        assert_close(
            rensa_r2,
            float(summary[endpoint.summary_rensa_r2]),
            f"{endpoint.task} RenSA R2 vs summary",
        )
        verify_table5_rounding(
            endpoint, baseline_rmsep, rensa_rmsep, baseline_r2, rensa_r2
        )

        for method, prediction in (("Baseline", baseline_pred), ("RenSA", rensa_pred)):
            values = diagnostics(y_true, prediction)
            result_rows.append(
                {
                    "Task": endpoint.task,
                    "Method": method,
                    "n_holdout": endpoint.expected_n,
                    "RMSEP": values["RMSEP"],
                    "Bias": values["Bias"],
                    "Slope": values["Slope"],
                    "SEP": values["SEP"],
                }
            )
            decomposition_rows.append(
                {
                    "Task": endpoint.task,
                    "Method": method,
                    "RMSEP_squared": values["RMSEP_squared"],
                    "decomposition_value": values["decomposition_value"],
                    "absolute_difference": values["absolute_difference"],
                }
            )

        provenance_rows.append(
            {
                "Task": endpoint.task,
                "prediction_file": endpoint.predictions,
                "prediction_sha256": sha256(prediction_path),
                "summary_file": endpoint.summary,
                "n_holdout": endpoint.expected_n,
                "split_method": str(summary["split_method"]),
                "test_size": float(summary["test_size"]),
                "baseline_RMSEP": baseline_rmsep,
                "RenSA_RMSEP": rensa_rmsep,
                "baseline_R2": baseline_r2,
                "RenSA_R2": rensa_r2,
                "delta_R2": rensa_r2 - baseline_r2,
                "Table5_rounding_verified": True,
            }
        )

    results = pd.DataFrame(result_rows)
    audit = pd.DataFrame(decomposition_rows)
    provenance = pd.DataFrame(provenance_rows)
    if len(results) != 10 or len(audit) != 10 or len(provenance) != 5:
        raise RuntimeError("Unexpected output row count")
    if float(audit["absolute_difference"].max()) > 1e-12:
        raise ArithmeticError("RMSEP decomposition exceeded floating-point tolerance")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "holdout_bias_slope_sep.csv", index=False)
    audit.to_csv(args.output_dir / "holdout_rmsep_decomposition_audit.csv", index=False)
    provenance.to_csv(args.output_dir / "prediction_provenance_audit.csv", index=False)

    print(results.to_string(index=False))
    print(
        "Maximum RMSEP decomposition absolute difference: "
        f"{audit['absolute_difference'].max():.3e}"
    )


if __name__ == "__main__":
    main()
