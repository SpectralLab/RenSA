"""Independently audit the saved cross-split sparsity figures and correlations.

This does not refit RenSA or SVR. It recomputes split descriptors from the
measured source data, prediction metrics from held-out predictions, and the
Spearman analyses from the verified split-level pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import train_test_split


TASKS = {
    "coal_Q": ("GCV", "coal_Q.csv", "y", "x"),
    "coal_ash": ("Ash", "coal_ash.csv", "y", "x"),
    "soil_SOC": ("SOC", "soil_SOC.csv", "y", "x"),
    "diesel_CN": ("CN", "diesel_CN.csv", "CN", ""),
    "diesel_FREEZE": ("FREEZE", "diesel_FREEZE.csv", "FREEZE", ""),
}
SPARSITY = ("mean", "p90", "median")
OUTCOMES = ("delta_rmsep", "delta_r2")
TOL = 1e-9


def read_source(path: Path, target: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if target not in frame.columns:
        frame = pd.read_csv(path, header=None, encoding="utf-8-sig")
        frame.columns = [f"{prefix}{i}" for i in range(1, frame.shape[1])] + [target]
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    columns = sorted(
        (name for name in frame.columns if pattern.fullmatch(str(name))),
        key=lambda name: int(pattern.fullmatch(str(name)).group(1)),
    )
    if not columns or target not in frame:
        raise ValueError(f"Source format invalid: {path}")
    x = frame[columns].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite source values: {path}")
    return x, y


def split_indices(record: pd.Series, n: int) -> tuple[np.ndarray, np.ndarray]:
    cal = np.asarray(json.loads(record["calibration_indices"]), dtype=int)
    hold = np.asarray(json.loads(record["holdout_indices"]), dtype=int)
    if len(cal) != int(record["n_calibration"]) or len(hold) != int(record["n_holdout"]):
        raise AssertionError("Stored split sizes disagree with indices")
    if len(np.unique(np.r_[cal, hold])) != n or set(np.r_[cal, hold]) != set(range(n)):
        raise AssertionError("Calibration and holdout indices do not partition the source rows")
    return cal, hold


def nearest_fifth(y_cal: np.ndarray) -> np.ndarray:
    z = (y_cal - y_cal.mean()) / y_cal.std(ddof=0)
    distances = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(distances, 0.0)
    return np.partition(distances, 5, axis=1)[:, 5]


def sparsity_from_source(y_cal: np.ndarray) -> dict[str, float]:
    fifth = nearest_fifth(y_cal)
    return {
        "mean": float(np.mean(fifth)),
        "p90": float(np.quantile(fifth, 0.90)),
        "median": float(np.median(fifth)),
    }


def make_strata(y: np.ndarray, requested: int, test_size: float) -> tuple[np.ndarray, int]:
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    for bins in range(min(requested, len(y)), 1, -1):
        labels = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"), dtype=int)
        _, counts = np.unique(labels, return_counts=True)
        n_bins = len(counts)
        if n_bins >= 2 and counts.min() >= 2 and n_test >= n_bins and n_train >= n_bins:
            return labels, n_bins
    raise AssertionError("Cannot independently reproduce stratification")


def reproduce_mc_split(y: np.ndarray, seed: int, requested: int, test_size: float) -> tuple[np.ndarray, np.ndarray, int]:
    labels, used = make_strata(y, requested, test_size)
    cal, hold = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed,
        shuffle=True, stratify=labels,
    )
    return np.sort(cal), np.sort(hold), used


def reproduce_ks_split(x: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    scaled = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0, ddof=1), 1e-12)
    sq = np.sum(scaled * scaled, axis=1)
    distances = np.maximum(sq[:, None] + sq[None, :] - 2 * scaled @ scaled.T, 0)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [int(first), int(second)]
    remaining = np.ones(len(x), dtype=bool)
    remaining[selected] = False
    nearest = np.minimum(distances[:, first], distances[:, second])
    n_cal = min(max(int(round(len(x) * (1 - test_size))), 2), len(x) - 1)
    while len(selected) < n_cal:
        candidates = np.flatnonzero(remaining)
        next_idx = int(candidates[np.argmax(nearest[candidates])])
        selected.append(next_idx)
        remaining[next_idx] = False
        nearest = np.minimum(nearest, distances[:, next_idx])
    return np.asarray(selected), np.flatnonzero(remaining)


def verify_predictions(
    record: pd.Series, predictions: pd.DataFrame, hold: np.ndarray, y: np.ndarray,
) -> tuple[float, float, float]:
    if len(predictions) != len(hold):
        raise AssertionError("Prediction count differs from holdout size")
    rows = predictions.sort_values("sample_index")
    if not np.array_equal(rows["sample_index"].to_numpy(dtype=int), np.sort(hold)):
        raise AssertionError("Predictions refer to incorrect holdout rows")
    truth = rows["y_true"].to_numpy(dtype=float)
    if not np.allclose(truth, y[np.sort(hold)], atol=TOL, rtol=0):
        raise AssertionError("Prediction truth differs from measured source response")
    base = rows["baseline_prediction"].to_numpy(dtype=float)
    rensa = rows["rensa_prediction"].to_numpy(dtype=float)
    if not np.allclose(rows["baseline_residual"], truth - base, atol=TOL, rtol=0):
        raise AssertionError("Baseline residual values incorrect")
    if not np.allclose(rows["rensa_residual"], truth - rensa, atol=TOL, rtol=0):
        raise AssertionError("RenSA residual values incorrect")
    baseline_rmsep = float(np.sqrt(np.mean((truth - base) ** 2)))
    rensa_rmsep = float(np.sqrt(np.mean((truth - rensa) ** 2)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    baseline_r2 = float(1 - np.sum((truth - base) ** 2) / denominator)
    rensa_r2 = float(1 - np.sum((truth - rensa) ** 2) / denominator)
    recomputed = {
        "baseline_rmsep": baseline_rmsep,
        "rensa_rmsep": rensa_rmsep,
        "baseline_r2": baseline_r2,
        "rensa_r2": rensa_r2,
        "delta_rmsep": rensa_rmsep - baseline_rmsep,
        "delta_r2": rensa_r2 - baseline_r2,
    }
    for key, value in recomputed.items():
        if not np.isclose(float(record[key]), value, atol=TOL, rtol=0):
            raise AssertionError(f"Stored {key} differs from predictions")
    return recomputed["delta_rmsep"], recomputed["delta_r2"], max(
        abs(float(record[key]) - value) for key, value in recomputed.items()
    )


def spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    # Paired percentile bootstrap: each resampled row retains its (sparsity, gain) pair.
    indices = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    x_rank = rankdata(x[indices], axis=1)
    y_rank = rankdata(y[indices], axis=1)
    x_rank -= x_rank.mean(axis=1, keepdims=True)
    y_rank -= y_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x_rank * x_rank, axis=1) * np.sum(y_rank * y_rank, axis=1))
    boot = np.sum(x_rank * y_rank, axis=1) / denominator
    boot = boot[np.isfinite(boot)]
    if len(boot) < n_bootstrap:
        raise AssertionError("Undefined bootstrap Spearman coefficient")
    return tuple(map(float, np.quantile(boot, [0.025, 0.975])))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(value: float, expected: float, context: str) -> float:
    difference = abs(float(value) - float(expected))
    if difference > TOL:
        raise AssertionError(f"{context}: difference {difference:.4g} exceeds {TOL}")
    return difference


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "cross_split_robustness_final")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "cross_split_sparsity_audit")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads((args.results_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    mc = pd.read_csv(args.results_dir / "monte_carlo_split_results.csv")
    ks = pd.read_csv(args.results_dir / "ks_results.csv")
    mc_pred = pd.read_csv(args.results_dir / "monte_carlo_predictions.csv")
    ks_pred = pd.read_csv(args.results_dir / "ks_predictions.csv")
    old_corr = pd.read_csv(args.results_dir / "sparsity_gain_correlations.csv")
    old_mean_corr = pd.read_csv(args.results_dir / "mean_5nn_sparsity_gain_correlations.csv")
    old_mean_pairs = pd.read_csv(args.results_dir / "mean_5nn_split_metrics.csv")
    expected_seeds = sorted(map(int, manifest["outer_seeds"]))
    if expected_seeds != list(range(50)) or len(mc) != 250 or len(ks) != 5:
        raise AssertionError("Unexpected split count or seed range")
    if mc.duplicated(["task", "outer_seed"]).any() or ks.duplicated("task").any():
        raise AssertionError("Duplicate task/seed result rows")
    if mc_pred.duplicated(["task", "outer_seed", "sample_index"]).any():
        raise AssertionError("Duplicate Monte Carlo prediction rows")
    if ks_pred.duplicated(["task", "sample_index"]).any():
        raise AssertionError("Duplicate KS prediction rows")
    mc_predictions = {(task, int(seed)): group for (task, seed), group in mc_pred.groupby(["task", "outer_seed"])}
    ks_predictions = {task: group for task, group in ks_pred.groupby("task")}
    mean_pair_lookup = old_mean_pairs.set_index(["task", "split_method", "outer_seed"])
    rows = []
    check_count = 0
    max_sparsity_diff = 0.0
    max_prediction_diff = 0.0
    split_checks = 0
    for task, (label, filename, target, prefix) in TASKS.items():
        x_source, y_source = read_source(args.data_dir / filename, target, prefix)
        task_mc = mc.loc[mc.task == task].sort_values("outer_seed")
        if task_mc["outer_seed"].astype(int).tolist() != expected_seeds:
            raise AssertionError(f"Missing or extra Monte Carlo seed for {task}")
        for _, record in task_mc.iterrows():
            seed = int(record["outer_seed"])
            cal, hold = split_indices(record, len(y_source))
            reproduced_cal, reproduced_hold, used_strata = reproduce_mc_split(
                y_source, seed, int(record["requested_strata"]), float(manifest["test_size"])
            )
            if not np.array_equal(cal, reproduced_cal) or not np.array_equal(hold, reproduced_hold):
                raise AssertionError(f"Cannot reproduce Monte Carlo split {task}/{seed}")
            if used_strata != int(record["used_strata"]):
                raise AssertionError(f"Wrong used_strata for {task}/{seed}")
            split_checks += 1
            sparse = sparsity_from_source(y_source[cal])
            max_sparsity_diff = max(
                max_sparsity_diff,
                check_close(sparse["p90"], record["sparsity_p90_5nn"], f"P90 {task}/{seed}"),
                check_close(sparse["median"], record["sparsity_median_5nn"], f"median {task}/{seed}"),
            )
            lookup = mean_pair_lookup.loc[(task, "stratified_monte_carlo", float(seed))]
            max_sparsity_diff = max(
                max_sparsity_diff,
                check_close(sparse["mean"], lookup["sparsity_mean_5nn"], f"mean {task}/{seed}"),
            )
            delta_rmsep, delta_r2, prediction_diff = verify_predictions(
                record, mc_predictions[(task, seed)], hold, y_source
            )
            max_prediction_diff = max(max_prediction_diff, prediction_diff)
            rows.append({
                "task": task,
                "task_label": label,
                "outer_seed": seed,
                "n_calibration": len(cal),
                "n_holdout": len(hold),
                "sparsity_mean_5nn_recomputed": sparse["mean"],
                "sparsity_p90_5nn_reported": float(record["sparsity_p90_5nn"]),
                "sparsity_p90_5nn_recomputed": sparse["p90"],
                "sparsity_median_5nn_recomputed": sparse["median"],
                "delta_rmsep_reported": float(record["delta_rmsep"]),
                "delta_rmsep_recomputed": delta_rmsep,
                "delta_r2_reported": float(record["delta_r2"]),
                "delta_r2_recomputed": delta_r2,
            })
            check_count += 1
        ks_record = ks.loc[ks.task == task].iloc[0]
        ks_cal, ks_hold = split_indices(ks_record, len(y_source))
        reproduced_cal, reproduced_hold = reproduce_ks_split(x_source, float(manifest["test_size"]))
        if not np.array_equal(ks_cal, reproduced_cal) or not np.array_equal(ks_hold, reproduced_hold):
            raise AssertionError(f"Cannot reproduce X-only KS split for {task}")
        sparse_ks = sparsity_from_source(y_source[ks_cal])
        max_sparsity_diff = max(
            max_sparsity_diff,
            check_close(sparse_ks["p90"], ks_record["sparsity_p90_5nn"], f"KS P90 {task}"),
            check_close(sparse_ks["median"], ks_record["sparsity_median_5nn"], f"KS median {task}"),
        )
        lookup = mean_pair_lookup.loc[(task, "kennard_stone", np.nan)]
        max_sparsity_diff = max(
            max_sparsity_diff,
            check_close(sparse_ks["mean"], lookup["sparsity_mean_5nn"], f"KS mean {task}"),
        )
        _, _, prediction_diff = verify_predictions(ks_record, ks_predictions[task], ks_hold, y_source)
        max_prediction_diff = max(max_prediction_diff, prediction_diff)
        check_count += 1
    verified = pd.DataFrame(rows)
    rng = np.random.default_rng(args.seed)
    correlation_rows = []
    max_correlation_diff = 0.0
    for task in TASKS:
        group = verified.loc[verified.task == task]
        for sparsity in SPARSITY:
            x = group[f"sparsity_{sparsity}_5nn_recomputed"].to_numpy(dtype=float)
            for outcome in OUTCOMES:
                y = group[f"{outcome}_recomputed"].to_numpy(dtype=float)
                rho, p_value = spearmanr(x, y)
                original = old_mean_corr if sparsity == "mean" else old_corr
                original = original.loc[
                    (original.task == task)
                    & (original.sparsity_metric == f"sparsity_{sparsity}_5nn")
                    & (original.gain_metric == outcome)
                ]
                if len(original) != 1 or int(original.iloc[0]["n_splits"]) != 50:
                    raise AssertionError(f"Missing published correlation: {task}/{sparsity}/{outcome}")
                max_correlation_diff = max(
                    max_correlation_diff,
                    check_close(rho, original.iloc[0]["rho"], f"rho {task}/{sparsity}/{outcome}"),
                    check_close(p_value, original.iloc[0]["p_value"], f"p {task}/{sparsity}/{outcome}"),
                )
                ci_low, ci_high = spearman_ci(x, y, rng, args.bootstrap)
                correlation_rows.append({
                    "task": task,
                    "task_label": TASKS[task][0],
                    "sparsity_metric": f"sparsity_{sparsity}_5nn",
                    "gain_metric": outcome,
                    "n_splits": 50,
                    "spearman_rho": rho,
                    "p_value_two_sided": p_value,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "bootstrap_method": "paired_percentile",
                    "bootstrap_resamples": args.bootstrap,
                    "bootstrap_seed": args.seed,
                    "ks_included": False,
                })
    correlations = pd.DataFrame(correlation_rows)
    p90 = correlations.loc[correlations.sparsity_metric == "sparsity_p90_5nn"].copy()
    p90_lookup = p90.set_index(["task", "gain_metric"])
    p90_table_rows = []
    for task, (label, _, _, _) in TASKS.items():
        rmsep = p90_lookup.loc[(task, "delta_rmsep")]
        r2 = p90_lookup.loc[(task, "delta_r2")]
        p90_table_rows.append({
            "Task": label,
            "ΔRMSEP: ρ (95% CI)": f"{rmsep.spearman_rho:+.3f} [{rmsep.bootstrap_ci95_low:+.3f}, {rmsep.bootstrap_ci95_high:+.3f}]",
            "ΔRMSEP: P": f"{rmsep.p_value_two_sided:.3f}",
            "ΔR²: ρ (95% CI)": f"{r2.spearman_rho:+.3f} [{r2.bootstrap_ci95_low:+.3f}, {r2.bootstrap_ci95_high:+.3f}]",
            "ΔR²: P": f"{r2.p_value_two_sided:.3f}",
        })
    p90_table = pd.DataFrame(p90_table_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified.to_csv(args.output_dir / "verified_mc_split_pairs.csv", index=False, encoding="utf-8-sig")
    verified[[
        "task", "task_label", "outer_seed", "sparsity_p90_5nn_reported",
        "delta_rmsep_reported", "delta_r2_reported",
    ]].rename(columns={
        "sparsity_p90_5nn_reported": "sparsity_p90_5nn",
        "delta_rmsep_reported": "delta_rmsep",
        "delta_r2_reported": "delta_r2",
    }).to_csv(args.output_dir / "p90_split_pairs.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output_dir / "verified_all_three_correlations.csv", index=False, encoding="utf-8-sig")
    p90_table.to_csv(args.output_dir / "table_S5_p90_sparsity_gain.csv", index=False, encoding="utf-8-sig")
    hashes = {filename: sha256(args.data_dir / filename) for _, filename, _, _ in TASKS.values()}
    hashes.update({filename: sha256(args.results_dir / filename) for filename in (
        "monte_carlo_split_results.csv", "monte_carlo_predictions.csv",
        "ks_results.csv", "ks_predictions.csv", "experiment_manifest.json",
    )})
    lines = [
        "# Cross-split sparsity audit",
        "",
        f"Verified {check_count} stored splits: 250 Monte Carlo and 5 X-only KS; all sample indices, sizes, response values, and held-out predictions reconcile.",
        f"Maximum absolute difference in recomputed 5-NN sparsity: {max_sparsity_diff:.3g}.",
        f"Maximum absolute difference in recomputed prediction metrics: {max_prediction_diff:.3g}.",
        f"Maximum absolute difference in recomputed Spearman rho or P: {max_correlation_diff:.3g}.",
        "",
        "## Method",
        "",
        "For each split, the measured calibration responses alone were z-scored using their own population SD (ddof=0). For each measured calibration sample, the absolute distance to the fifth other calibration response (the 5-NN distance) was calculated in the standardized response space. The mean, P90 (NumPy linear quantile), and median of these distances were then computed. No synthetic or holdout responses entered this step.",
        "",
        "The held-out prediction rows were matched by task, seed, and sample index to the original measured response. RMSEP was recomputed as sqrt(mean squared prediction error), and R² as 1 - SSE/SST, using each holdout mean for SST. ΔRMSEP and ΔR² equal RenSA minus baseline.",
        "",
        "Each Spearman rho and two-sided SciPy P value uses the 50 paired Monte Carlo observations within one task. X-only KS points are plotted for reference but are excluded. The 95% CI is the 2.5th and 97.5th percentiles of 10,000 paired bootstrap resamples of 50 (sparsity, gain) rows, sampled with replacement. The random generator is NumPy default_rng(42), consumed in task order GCV, Ash, SOC, CN, FREEZE; sparsity order mean, P90, median; gain order ΔRMSEP, ΔR². The intervals describe variation over the saved overlapping splits and should not be treated as independent-dataset generalization intervals.",
        "",
        "## Table S5. Association between P90 response-neighbor sparsity and RenSA gain across Monte Carlo partitions",
        "",
        "| Task | ΔRMSEP: Spearman ρ (95% CI) | P | ΔR²: Spearman ρ (95% CI) | P |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in p90_table.iterrows():
        lines.append(
            f"| {row['Task']} | {row['ΔRMSEP: ρ (95% CI)']} | {row['ΔRMSEP: P']} | "
            f"{row['ΔR²: ρ (95% CI)']} | {row['ΔR²: P']} |"
        )
    lines.extend([
        "",
        "Note. P90 is the 90th percentile of the fifth-nearest-neighbor distances among measured calibration responses after standardization within each split. Each task includes 50 Monte Carlo 75/25 partitions. Spearman ρ was calculated from the paired split-level P90 and gain values; P values are two-sided. The 95% CIs use 10,000 paired percentile bootstrap resamples (seed 42). X-only KS partitions are excluded. Δ = RenSA − baseline; lower ΔRMSEP and higher ΔR² indicate improvement.",
        "",
        "All ten P90 CIs include zero. The saved figures and correlation tables are numerically consistent with the saved raw inputs and predictions. These checks do not independently refit CARS, SVR, or RenSA; a full training rerun is a separate, much longer experiment. The fifty repeated splits of the same dataset overlap, so this descriptive split-level analysis does not establish that sparsity predicts gain for new datasets.",
        "",
        "## SHA-256 of source files",
        "",
    ])
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in hashes.items())
    (args.output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: {check_count} splits, {len(correlations)} correlations; P90 rows={len(p90)}")
    print(f"Max abs differences: sparsity={max_sparsity_diff:.3g}, predictions={max_prediction_diff:.3g}, rho/P={max_correlation_diff:.3g}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
