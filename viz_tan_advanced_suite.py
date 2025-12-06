import argparse
import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def make_output_dir(base: str, label: str) -> str:
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"tan_suite_{label}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def load_summary(path: str) -> pd.DataFrame:
    """Load summary_metrics.csv from a reports directory."""
    summary_path = os.path.join(path, "summary_metrics.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_metrics.csv not found at {summary_path}")
    return pd.read_csv(summary_path)


def filter_models(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Return a copy of df restricted to the given model names (in that order)."""
    df_f = df[df["model"].isin(models)].copy()
    # preserve input order
    df_f["__order"] = pd.Categorical(df_f["model"], categories=models, ordered=True)
    df_f = df_f.sort_values("__order").drop(columns="__order")
    missing = [m for m in models if m not in df_f["model"].tolist()]
    if missing:
        print(f"Warning: missing models in summary: {missing}")
    return df_f


def copy_roc_curves(fig_src_dir: str, fig_dst_dir: str, models: List[str], multiclass: bool) -> None:
    """
    Copy existing ROC OOF PNGs for the specified models into fig_dst_dir.

    For binary:  source filenames are 'roc_oof_<model>.png'
    For multiclass: source filenames are 'roc_oof_multiclass_<model>.png'
    """
    os.makedirs(fig_dst_dir, exist_ok=True)
    prefix = "roc_oof_multiclass_" if multiclass else "roc_oof_"
    for m in models:
        src = os.path.join(fig_src_dir, f"{prefix}{m}.png")
        if os.path.exists(src):
            dst = os.path.join(fig_dst_dir, f"{prefix}{m}.png")
            with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())
            print(f"Copied ROC: {src} -> {dst}")
        else:
            print(f"Warning: ROC file not found for model '{m}': {src}")


def plot_binary_bar(df_bin: pd.DataFrame, out_dir: str, label: str) -> None:
    """Bar plot of binary AUROC (roc_auc_mean) with 95% CI for selected models."""
    required_cols = ["model", "roc_auc_mean", "roc_auc_ci95_lo", "roc_auc_ci95_hi"]
    if not all(c in df_bin.columns for c in required_cols):
        print("Warning: Skipping binary bar plot: required AUROC columns not found.")
        return

    df = df_bin.copy()
    df = df.sort_values("roc_auc_mean", ascending=True)
    means = df["roc_auc_mean"].values
    ci_lo = df["roc_auc_ci95_lo"].values
    ci_hi = df["roc_auc_ci95_hi"].values
    yerr = [means - ci_lo, ci_hi - means]

    plt.figure(figsize=(10, 6))
    y_pos = range(len(df))
    plt.barh(y_pos, means, xerr=yerr, capsize=4, color="#2ECC71", alpha=0.85)
    plt.yticks(y_pos, df["model"])
    plt.xlabel("Binary AUROC")
    plt.title(f"Binary AUROC with 95% CI ({label})")
    plt.xlim(0.5, 1.0)

    for i, m in enumerate(means):
        plt.text(m + 0.005, i, f"{m:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "binary_bar_auroc_with_ci.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_multiclass_bar(df_multi: pd.DataFrame, out_dir: str, label: str) -> None:
    """Bar plot of multiclass AUROC (roc_auc_macro_mean) with 95% CI for selected models."""
    required_cols = ["model", "roc_auc_macro_mean", "roc_auc_macro_ci95_lo", "roc_auc_macro_ci95_hi"]
    if not all(c in df_multi.columns for c in required_cols):
        print("Warning: Skipping multiclass bar plot: required AUROC columns not found.")
        return

    df = df_multi.copy()
    df = df.sort_values("roc_auc_macro_mean", ascending=True)
    means = df["roc_auc_macro_mean"].values
    ci_lo = df["roc_auc_macro_ci95_lo"].values
    ci_hi = df["roc_auc_macro_ci95_hi"].values
    yerr = [means - ci_lo, ci_hi - means]

    plt.figure(figsize=(10, 6))
    y_pos = range(len(df))
    plt.barh(y_pos, means, xerr=yerr, capsize=4, color="#3498DB", alpha=0.85)
    plt.yticks(y_pos, df["model"])
    plt.xlabel("Multiclass AUROC (macro OVR)")
    plt.title(f"Multiclass AUROC with 95% CI ({label})")
    plt.xlim(0.5, 1.0)

    for i, m in enumerate(means):
        plt.text(m + 0.005, i, f"{m:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "multiclass_bar_auroc_with_ci.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_binary_heatmap(df_bin: pd.DataFrame, out_dir: str, label: str) -> None:
    """Heatmap of selected binary metrics for the chosen models."""
    metrics = ["roc_auc_mean", "pr_auc_mean", "brier_mean", "ece_mean"]
    present = [m for m in metrics if m in df_bin.columns]
    if not present:
        print("Warning: Skipping binary heatmap: no expected metric columns found.")
        return

    data = df_bin.set_index("model")[present]
    data = data.sort_values(present[0], ascending=False)

    plt.figure(figsize=(max(8, len(present) * 1.6), 0.4 * len(data) + 2))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
    )
    plt.title(f"Binary Metrics Heatmap ({label})")
    plt.ylabel("Model")
    plt.tight_layout()
    path = os.path.join(out_dir, "binary_metrics_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_multiclass_heatmap(df_multi: pd.DataFrame, out_dir: str, label: str) -> None:
    """Heatmap of selected multiclass metrics for the chosen models."""
    metrics = ["roc_auc_macro_mean", "roc_auc_weighted_mean", "f1_macro_mean", "ece_mean"]
    present = [m for m in metrics if m in df_multi.columns]
    if not present:
        print("Warning: Skipping multiclass heatmap: no expected metric columns found.")
        return

    data = df_multi.set_index("model")[present]
    data = data.sort_values(present[0], ascending=False)

    plt.figure(figsize=(max(8, len(present) * 1.6), 0.4 * len(data) + 2))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
    )
    plt.title(f"Multiclass Metrics Heatmap ({label})")
    plt.ylabel("Model")
    plt.tight_layout()
    path = os.path.join(out_dir, "multiclass_metrics_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_binary_vs_multiclass_scatter(
    df_bin: pd.DataFrame, df_multi: pd.DataFrame, out_dir: str, label: str
) -> None:
    """Scatter plot of binary vs multiclass AUROC for overlapping models."""
    if "roc_auc_mean" not in df_bin.columns or "roc_auc_macro_mean" not in df_multi.columns:
        print("Warning: Skipping binary vs multiclass scatter: AUROC columns missing.")
        return

    df_b = df_bin[["model", "roc_auc_mean"]].rename(columns={"roc_auc_mean": "auroc_bin"})
    df_m = df_multi[["model", "roc_auc_macro_mean"]].rename(
        columns={"roc_auc_macro_mean": "auroc_multi"}
    )
    df = pd.merge(df_b, df_m, on="model", how="inner")
    if df.empty:
        print("Warning: No overlapping models for binary vs multiclass scatter.")
        return

    plt.figure(figsize=(7, 7))
    sns.scatterplot(data=df, x="auroc_bin", y="auroc_multi", s=80, color="#9B59B6")

    lo = min(df["auroc_bin"].min(), df["auroc_multi"].min())
    hi = max(df["auroc_bin"].max(), df["auroc_multi"].max())
    lo = max(0.5, lo - 0.02)
    hi = min(1.0, hi + 0.02)
    plt.plot([lo, hi], [lo, hi], ls="--", color="gray", label="y = x")

    for _, row in df.iterrows():
        plt.text(row["auroc_bin"] + 0.002, row["auroc_multi"], row["model"], fontsize=8)

    plt.xlabel("Binary AUROC")
    plt.ylabel("Multiclass AUROC (macro OVR)")
    plt.title(f"Binary vs Multiclass AUROC ({label})")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(out_dir, "scatter_binary_vs_multiclass_auroc.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_binary_vs_multiclass_bar(
    df_bin: pd.DataFrame, df_multi: pd.DataFrame, out_dir: str, label: str
) -> None:
    """Side-by-side bar chart of binary vs multiclass AUROC for overlapping models."""
    if "roc_auc_mean" not in df_bin.columns or "roc_auc_macro_mean" not in df_multi.columns:
        print("Warning: Skipping binary vs multiclass bar: AUROC columns missing.")
        return

    df_b = df_bin[["model", "roc_auc_mean"]].rename(columns={"roc_auc_mean": "auroc_bin"})
    df_m = df_multi[["model", "roc_auc_macro_mean"]].rename(
        columns={"roc_auc_macro_mean": "auroc_multi"}
    )
    df = pd.merge(df_b, df_m, on="model", how="inner")
    if df.empty:
        print("Warning: No overlapping models for binary vs multiclass bar plot.")
        return

    df = df.sort_values("auroc_bin", ascending=True)
    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], df["auroc_bin"], width=width, label="Binary", color="#2ECC71")
    plt.bar(
        [i + width / 2 for i in x],
        df["auroc_multi"],
        width=width,
        label="Multiclass (macro AUROC)",
        color="#3498DB",
    )

    plt.xticks(x, df["model"], rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.title(f"Binary vs Multiclass AUROC per Model ({label})")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "bar_binary_vs_multiclass_auroc.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate TAN Bayes + advanced model visualizations from original CV reports "
            "(uses only existing summary_metrics.csv and ROC PNGs)."
        )
    )
    parser.add_argument(
        "--binary-reports-dir",
        required=True,
        help="Path to binary reports dir from run_experiments.py (contains summary_metrics.csv).",
    )
    parser.add_argument(
        "--multiclass-reports-dir",
        required=True,
        help="Path to multiclass reports dir from run_experiments.py (contains summary_metrics.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_visualizations",
        help="Parent directory where all plots will be saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "random_forest",
            "stacking",
            "logreg_L2",
            "logreg_L1",
            "categorical_nb_discretized",
            "tan_bayes",
        ],
        help="List of model names (as they appear in summary_metrics.csv) to include.",
    )
    parser.add_argument(
        "--label",
        default="tan_suite",
        help="Label used in plot titles and output folder name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    out_dir = make_output_dir(args.output_dir, args.label)
    print(f"Binary reports:     {args.binary_reports_dir}")
    print(f"Multiclass reports: {args.multiclass_reports_dir}")
    print(f"Output directory:   {out_dir}")

    # Load summary metrics
    df_bin_full = load_summary(args.binary_reports_dir)
    df_multi_full = load_summary(args.multiclass_reports_dir)

    df_bin = filter_models(df_bin_full, args.models)
    df_multi = filter_models(df_multi_full, args.models)

    # 1 & 4: ROC curves (copied from existing figures)
    bin_fig_src = os.path.join(os.path.dirname(args.binary_reports_dir), "figures")
    multi_fig_src = os.path.join(os.path.dirname(args.multiclass_reports_dir), "figures")
    copy_roc_curves(bin_fig_src, os.path.join(out_dir, "binary_roc"), args.models, multiclass=False)
    copy_roc_curves(
        multi_fig_src, os.path.join(out_dir, "multiclass_roc"), args.models, multiclass=True
    )

    # 2: Binary metrics heatmap
    plot_binary_heatmap(df_bin, out_dir, args.label)

    # 3: Binary AUROC bar plot
    plot_binary_bar(df_bin, out_dir, args.label)

    # 5: Multiclass metrics heatmap
    plot_multiclass_heatmap(df_multi, out_dir, args.label)

    # 6: Multiclass AUROC bar plot
    plot_multiclass_bar(df_multi, out_dir, args.label)

    # 7: Combined binary vs multiclass comparisons
    plot_binary_vs_multiclass_scatter(df_bin, df_multi, out_dir, args.label)
    plot_binary_vs_multiclass_bar(df_bin, df_multi, out_dir, args.label)

    print("TAN + advanced model visualization suite complete.")



