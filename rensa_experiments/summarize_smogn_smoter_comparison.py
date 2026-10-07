"""Build R3.3a comparison tables and paired holdout-bootstrap evidence."""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RENSA_RESULTS = ROOT / "results"
STANDARD_RESULTS = ROOT / "results_standard_smogn_smoter"
EXPECTED_SPLITS = 35
EXPECTED_STANDARD_ROWS = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize standard SMOGN/SMOTER versus RenSA.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    standard = pd.read_csv(STANDARD_RESULTS / "split_results.csv")
    rensa = pd.read_csv(RENSA_RESULTS / "split_results.csv")
    standard_pred = pd.read_csv(STANDARD_RESULTS / "predictions.csv")
    rensa_pred = pd.read_csv(RENSA_RESULTS / "predictions.csv")
    for frame in (standard, rensa, standard_pred, rensa_pred):
        frame["seed_key"] = frame["outer_seed"].map(normalize_seed)

    validate_inputs(standard, rensa, args.allow_incomplete)
    keys = ["task", "split_method", "seed_key"]
    comparison = standard.merge(
        rensa.loc[:, keys + ["R2", "RMSEP"]].rename(
            columns={"R2": "RenSA_R2", "RMSEP": "RenSA_RMSEP"}
        ),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if comparison[["RenSA_R2", "RenSA_RMSEP"]].isna().any().any():
        raise RuntimeError("At least one standard-baseline split has no matching RenSA result.")
    comparison = comparison.rename(columns={"R2": "baseline_R2", "RMSEP": "baseline_RMSEP"})
    comparison["delta_RMSEP_baseline_minus_RenSA"] = (
        comparison["baseline_RMSEP"] - comparison["RenSA_RMSEP"]
    )
    comparison["relative_RMSEP_reduction_vs_baseline_pct"] = (
        100.0
        * comparison["delta_RMSEP_baseline_minus_RenSA"]
        / comparison["baseline_RMSEP"]
    )
    comparison["delta_R2_baseline_minus_RenSA"] = comparison["baseline_R2"] - comparison["RenSA_R2"]

    bootstrap_rows = []
    for row in comparison.itertuples(index=False):
        key = (row.method, row.task, row.split_method, row.seed_key)
        baseline_subset = select_predictions(standard_pred, key)
        rensa_subset = select_predictions(rensa_pred, key[1:])
        merged = baseline_subset.merge(
            rensa_subset,
            on="sample_index",
            how="inner",
            suffixes=("_baseline", "_rensa"),
            validate="one_to_one",
        ).sort_values("sample_index")
        if merged.shape[0] != baseline_subset.shape[0] or merged.shape[0] != rensa_subset.shape[0]:
            raise RuntimeError(f"Prediction rows do not align for {key}.")
        if not np.allclose(merged["y_true_baseline"], merged["y_true_rensa"]):
            raise RuntimeError(f"Holdout targets differ for {key}.")
        seed = args.random_state + zlib.crc32("|".join(key).encode("utf-8"))
        bootstrap_rows.append(
            paired_rmsep_bootstrap(
                merged["y_true_baseline"].to_numpy(float),
                merged["y_pred_baseline"].to_numpy(float),
                merged["y_pred_rensa"].to_numpy(float),
                args.bootstrap,
                seed,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison = pd.concat([comparison.reset_index(drop=True), bootstrap], axis=1)
    comparison["RenSA_lower_RMSEP"] = (
        comparison["delta_RMSEP_baseline_minus_RenSA"] > 0
    ).astype(int)
    comparison["RenSA_significantly_lower_RMSEP"] = (
        comparison["delta_RMSEP_ci_low"] > 0
    ).astype(int)

    keep = [
        "method", "task", "split_method", "outer_seed", "selected_config_id",
        "preprocessing", "relevance_threshold", "k", "sampling_strategy",
        "perturbation", "RenSA_RMSEP", "baseline_RMSEP",
        "delta_RMSEP_baseline_minus_RenSA", "delta_RMSEP_ci_low",
        "delta_RMSEP_ci_high", "relative_RMSEP_reduction_vs_baseline_pct",
        "bootstrap_two_sided_p", "RenSA_R2",
        "baseline_R2", "delta_R2_baseline_minus_RenSA", "RenSA_lower_RMSEP",
        "RenSA_significantly_lower_RMSEP",
    ]
    comparison_out = comparison.loc[:, keep].sort_values(
        ["method", "task", "split_method", "outer_seed"], na_position="first"
    )
    summary = build_summary(comparison)
    atomic_csv(STANDARD_RESULTS / "r3_3a_comparison_by_split.csv", comparison_out)
    atomic_csv(STANDARD_RESULTS / "r3_3a_comparison_summary.csv", summary)
    write_evidence_json(comparison, summary, args.bootstrap)
    print(
        f"R3.3a summary complete: {comparison.shape[0]} paired method-splits, "
        f"{args.bootstrap} bootstrap replicates each."
    )


def normalize_seed(value: object) -> str:
    if pd.isna(value) or value == "":
        return "NA"
    return str(int(float(value)))


def validate_inputs(standard: pd.DataFrame, rensa: pd.DataFrame, allow_incomplete: bool) -> None:
    standard_keys = standard[["method", "task", "split_method", "seed_key"]]
    rensa_keys = rensa[["task", "split_method", "seed_key"]]
    if standard_keys.duplicated().any() or rensa_keys.duplicated().any():
        raise RuntimeError("Duplicate experiment keys found in aggregate results.")
    if not allow_incomplete:
        if rensa.shape[0] != EXPECTED_SPLITS:
            raise RuntimeError(f"Expected {EXPECTED_SPLITS} RenSA rows, found {rensa.shape[0]}.")
        if standard.shape[0] != EXPECTED_STANDARD_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_STANDARD_ROWS} standard-baseline rows, found {standard.shape[0]}."
            )
        counts = standard.groupby("method").size().to_dict()
        if counts != {"smogn": EXPECTED_SPLITS, "smoter": EXPECTED_SPLITS}:
            raise RuntimeError(f"Incomplete per-method matrix: {counts}")


def select_predictions(frame: pd.DataFrame, key: tuple[str, ...]) -> pd.DataFrame:
    if len(key) == 4:
        method, task, split_method, seed = key
        mask = (
            (frame["method"] == method)
            & (frame["task"] == task)
            & (frame["split_method"] == split_method)
            & (frame["seed_key"] == seed)
        )
    else:
        task, split_method, seed = key
        mask = (
            (frame["task"] == task)
            & (frame["split_method"] == split_method)
            & (frame["seed_key"] == seed)
        )
    columns = ["sample_index", "y_true", "y_pred"]
    result = frame.loc[mask, columns].copy()
    if result.empty:
        raise RuntimeError(f"No predictions found for {key}.")
    return result


def paired_rmsep_bootstrap(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    rensa_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = y_true.size
    deltas = np.empty(n_bootstrap, dtype=float)
    chunk = 1000
    baseline_error = (baseline_pred - y_true) ** 2
    rensa_error = (rensa_pred - y_true) ** 2
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        baseline_rmsep = np.sqrt(np.mean(baseline_error[indices], axis=1))
        rensa_rmsep = np.sqrt(np.mean(rensa_error[indices], axis=1))
        deltas[start:stop] = baseline_rmsep - rensa_rmsep
    low, high = np.quantile(deltas, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return {
        "delta_RMSEP_ci_low": float(low),
        "delta_RMSEP_ci_high": float(high),
        "bootstrap_two_sided_p": float(p_value),
    }


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, task), group in comparison.groupby(["method", "task"], sort=False):
        rows.append(summary_row(method, task, group))
    for method, group in comparison.groupby("method", sort=False):
        rows.append(summary_row(method, "ALL_TASKS", group))
    return pd.DataFrame(rows)


def summary_row(method: str, task: str, group: pd.DataFrame) -> dict[str, object]:
    spxy = group[group["split_method"] == "spxy"]
    ks = group[group["split_method"] == "ks"]
    mc = group[group["split_method"] == "mc"]

    def one(frame: pd.DataFrame, column: str) -> float | str:
        return float(frame.iloc[0][column]) if frame.shape[0] == 1 else ""

    return {
        "method": method,
        "task": task,
        "n_splits": int(group.shape[0]),
        "RenSA_lower_RMSEP_n": int((group["delta_RMSEP_baseline_minus_RenSA"] > 0).sum()),
        "RenSA_significantly_lower_RMSEP_n": int((group["delta_RMSEP_ci_low"] > 0).sum()),
        "baseline_significantly_lower_RMSEP_n": int((group["delta_RMSEP_ci_high"] < 0).sum()),
        "mean_delta_RMSEP_baseline_minus_RenSA": float(
            group["delta_RMSEP_baseline_minus_RenSA"].mean()
        ),
        "median_delta_RMSEP_baseline_minus_RenSA": float(
            group["delta_RMSEP_baseline_minus_RenSA"].median()
        ),
        "mean_relative_RMSEP_reduction_vs_baseline_pct": float(
            group["relative_RMSEP_reduction_vs_baseline_pct"].mean()
        ),
        "median_relative_RMSEP_reduction_vs_baseline_pct": float(
            group["relative_RMSEP_reduction_vs_baseline_pct"].median()
        ),
        "spxy_RenSA_RMSEP": one(spxy, "RenSA_RMSEP"),
        "spxy_baseline_RMSEP": one(spxy, "baseline_RMSEP"),
        "spxy_delta_RMSEP": one(spxy, "delta_RMSEP_baseline_minus_RenSA"),
        "spxy_delta_RMSEP_ci_low": one(spxy, "delta_RMSEP_ci_low"),
        "spxy_delta_RMSEP_ci_high": one(spxy, "delta_RMSEP_ci_high"),
        "ks_RenSA_RMSEP": one(ks, "RenSA_RMSEP"),
        "ks_baseline_RMSEP": one(ks, "baseline_RMSEP"),
        "ks_delta_RMSEP": one(ks, "delta_RMSEP_baseline_minus_RenSA"),
        "mc_n": int(mc.shape[0]),
        "mc_mean_RenSA_RMSEP": float(mc["RenSA_RMSEP"].mean()) if not mc.empty else "",
        "mc_mean_baseline_RMSEP": float(mc["baseline_RMSEP"].mean()) if not mc.empty else "",
        "mc_mean_delta_RMSEP": float(mc["delta_RMSEP_baseline_minus_RenSA"].mean())
        if not mc.empty else "",
    }


def write_evidence_json(comparison: pd.DataFrame, summary: pd.DataFrame, n_bootstrap: int) -> None:
    payload = {
        "review_item": "R3.3a",
        "paired_method_split_comparisons": int(comparison.shape[0]),
        "bootstrap_replicates_per_comparison": int(n_bootstrap),
        "sign_convention": "delta_RMSEP = standard baseline - RenSA; positive favors RenSA",
        "cross_task_aggregation": (
            "Raw RMSEP units are task-specific; interpret cross-task win counts and relative "
            "RMSEP reductions, and use raw deltas only within task."
        ),
        "overall": summary[summary["task"] == "ALL_TASKS"].to_dict(orient="records"),
    }
    path = STANDARD_RESULTS / "r3_3a_evidence.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)




def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
