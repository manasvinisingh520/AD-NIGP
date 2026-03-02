"""
Analyze F1 scores from expression (exp_*) vs gene embedding (ge_*) runs.

Directories:
  exp_1L, exp_3L, exp_all_3L  -> raw expression (expression_experiments.csv)
  ge_1L, ge_3L, ge_all_3L    -> gene embedding (overfitting_experiments.csv)

1L = 1 hidden layer (minimal model), 3L = 3 hidden layers.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Default base dir for CSVs (ranking_model_runs)
EXP_DIRS = ["exp_all_3L_shrinking", "exp_1L"]
GE_DIRS = ["ge_all_3L_shrinking", "ge_1L"]
EXP_CSV = "expression_experiments.csv"
GE_CSV = "overfitting_experiments.csv"


def load_f1_series(base_dir: str, subdir: str, csv_name: str) -> pd.Series:
    path = os.path.join(base_dir, subdir, csv_name)
    if not os.path.isfile(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    if "avg_f1" not in df.columns:
        return pd.Series(dtype=float)
    # Drop rows with missing F1 or non-empty error
    valid = df["avg_f1"].notna()
    if "error" in df.columns:
        valid = valid & (df["error"].isna() | (df["error"].astype(str).str.strip() == ""))
    return df.loc[valid, "avg_f1"].astype(float)


def load_all(base_dir: str):
    """Load F1 for all 6 directories. Returns dict label -> Series of F1."""
    out = {}
    for d in EXP_DIRS:
        s = load_f1_series(base_dir, d, EXP_CSV)
        out[d] = s
    for d in GE_DIRS:
        s = load_f1_series(base_dir, d, GE_CSV)
        out[d] = s
    return out


def summary_table(data: dict) -> pd.DataFrame:
    rows = []
    for label in EXP_DIRS + GE_DIRS:
        s = data.get(label, pd.Series(dtype=float))
        if len(s) == 0:
            rows.append({"dir": label, "n": 0, "mean": np.nan, "std": np.nan, "var": np.nan, "min": np.nan, "max": np.nan, "median": np.nan})
        else:
            rows.append({
                "dir": label,
                "n": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "var": s.var(),
                "min": s.min(),
                "max": s.max(),
                "median": s.median(),
            })
    return pd.DataFrame(rows)


def plot_box(data: dict, out_path: str):
    labels = EXP_DIRS + GE_DIRS
    series = [data.get(l, pd.Series(dtype=float)) for l in labels]
    # Filter to non-empty for plot
    plot_labels = [l for l, s in zip(labels, series) if len(s) > 0]
    plot_series = [data[l] for l in plot_labels]
    if not plot_series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(plot_series, labels=plot_labels, patch_artist=True)
    colors = ["#7eb8da"] * 3 + ["#e8a87c"] * 3  # blue exp, orange ge
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i] if i < len(colors) else "white")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 score by run directory (exp = expression, ge = gene embedding)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved box plot: {out_path}")


def plot_mean_bars(summary: pd.DataFrame, out_path: str):
    df = summary[summary["n"] > 0]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["mean"], yerr=df["std"], capsize=5, color=["#7eb8da"] * 3 + ["#e8a87c"] * 3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dir"], rotation=15)
    ax.set_ylabel("Mean F1 ± std")
    ax.set_title("Mean F1 by directory (exp = expression, ge = gene embedding)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved bar plot: {out_path}")


def plot_exp_vs_ge(summary: pd.DataFrame, out_path: str):
    """Group by 1L / 3L / all_3L: compare exp vs ge mean F1."""
    pairs = [("exp_1L", "ge_1L"), ("exp_3L", "ge_3L"), ("exp_all_3L", "ge_all_3L")]
    x_labels = ["1L (baseline only)", "3L (baseline only)", "All 3L (full grid)"]
    exp_means, ge_means = [], []
    exp_stds, ge_stds = [], []
    for (e, g) in pairs:
        se = summary[summary["dir"] == e]
        sg = summary[summary["dir"] == g]
        exp_means.append(se["mean"].values[0] if len(se) and se["n"].values[0] > 0 else np.nan)
        ge_means.append(sg["mean"].values[0] if len(sg) and sg["n"].values[0] > 0 else np.nan)
        exp_stds.append(se["std"].values[0] if len(se) and se["n"].values[0] > 0 else 0)
        ge_stds.append(sg["std"].values[0] if len(sg) and sg["n"].values[0] > 0 else 0)
    x = np.arange(len(x_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, exp_means, w, yerr=exp_stds, capsize=4, label="Expression (exp)", color="#7eb8da")
    ax.bar(x + w / 2, ge_means, w, yerr=ge_stds, capsize=4, label="Gene embedding (ge)", color="#e8a87c")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean F1 ± std")
    ax.set_title("Expression vs Gene embedding: mean F1 by model size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved exp vs ge plot: {out_path}")


def print_conclusions(summary: pd.DataFrame, data: dict):
    print("\n" + "=" * 60)
    print("CONCLUSIONS (qualitative)")
    print("=" * 60)
    exp_1 = summary[summary["dir"] == "exp_1L"]
    ge_1 = summary[summary["dir"] == "ge_1L"]
    exp_3 = summary[summary["dir"] == "exp_3L"]
    ge_3 = summary[summary["dir"] == "ge_3L"]
    exp_all = summary[summary["dir"] == "exp_all_3L"]
    ge_all = summary[summary["dir"] == "ge_all_3L"]

    def _mean_n(df):
        if df.empty or df["n"].values[0] == 0:
            return np.nan, 0
        return df["mean"].values[0], int(df["n"].values[0])

    m_exp_1, n_exp_1 = _mean_n(exp_1)
    m_ge_1, n_ge_1 = _mean_n(ge_1)
    m_exp_3, n_exp_3 = _mean_n(exp_3)
    m_ge_3, n_ge_3 = _mean_n(ge_3)
    m_exp_all, n_exp_all = _mean_n(exp_all)
    m_ge_all, n_ge_all = _mean_n(ge_all)

    print("\n1. Expression vs Gene embedding (same layer size)")
    if not np.isnan(m_exp_1) and not np.isnan(m_ge_1):
        diff_1 = m_ge_1 - m_exp_1
        print(f"   1L: exp mean F1 = {m_exp_1:.4f} (n={n_exp_1}), ge mean F1 = {m_ge_1:.4f} (n={n_ge_1}) -> ge - exp = {diff_1:+.4f}")
    if not np.isnan(m_exp_3) and not np.isnan(m_ge_3):
        diff_3 = m_ge_3 - m_exp_3
        print(f"   3L: exp mean F1 = {m_exp_3:.4f} (n={n_exp_3}), ge mean F1 = {m_ge_3:.4f} (n={n_ge_3}) -> ge - exp = {diff_3:+.4f}")
    if not np.isnan(m_exp_all) and not np.isnan(m_ge_all):
        diff_all = m_ge_all - m_exp_all
        print(f"   All 3L: exp mean F1 = {m_exp_all:.4f} (n={n_exp_all}), ge mean F1 = {m_ge_all:.4f} (n={n_ge_all}) -> ge - exp = {diff_all:+.4f}")

    print("\n2. Best run per directory (max F1)")
    for d in EXP_DIRS + GE_DIRS:
        s = data.get(d, pd.Series(dtype=float))
        if len(s) > 0:
            print(f"   {d}: max F1 = {s.max():.4f}")

    print("\n3. Interpretation")
    best_exp = max((data.get(d, pd.Series(dtype=float)).max() for d in EXP_DIRS), default=np.nan)
    best_ge = max((data.get(d, pd.Series(dtype=float)).max() for d in GE_DIRS), default=np.nan)
    if not np.isnan(m_ge_all) and not np.isnan(m_exp_all):
        if m_ge_all > m_exp_all:
            print("   Gene embedding (ge) achieves higher mean F1 than raw expression (exp) in the full grid (all 3L), suggesting the embedding may capture useful signal for disease classification.")
        elif m_ge_all < m_exp_all:
            print("   Raw expression (exp) achieves higher mean F1 than gene embedding (ge) in the full grid; with n~20, variance is high — consider label-shuffle null and confidence intervals.")
        else:
            print("   Mean F1 is similar between exp and ge; more runs or statistical tests (e.g. permutation) would help.")
    if not np.isnan(best_exp) and not np.isnan(best_ge):
        print(f"   Best single run: exp max F1 = {best_exp:.4f}, ge max F1 = {best_ge:.4f}.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze F1 from exp_* and ge_* CSV runs")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory containing exp_1L, ge_1L, etc. (default: ranking_model_runs next to script)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save plots (default: same as base_dir)")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    if args.base_dir is None:
        # Default: project root / ranking_model_runs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.join(os.path.dirname(script_dir), "ranking_model_runs")
    if args.output_dir is None:
        args.output_dir = args.base_dir

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading CSVs from: {args.base_dir}")
    data = load_all(args.base_dir)

    summary = summary_table(data)
    print("\nSummary (F1 per directory):")
    print(summary.to_string(index=False))

    print_conclusions(summary, data)

    if not args.no_plots:
        plot_box(data, os.path.join(args.output_dir, "f1_boxplot_by_dir.png"))
        plot_mean_bars(summary, os.path.join(args.output_dir, "f1_mean_bars.png"))
        plot_exp_vs_ge(summary, os.path.join(args.output_dir, "f1_exp_vs_ge.png"))

    # Save summary CSV
    out_csv = os.path.join(args.output_dir, "f1_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
