import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
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


def load_optimized_results(reports_dir: str) -> pd.DataFrame:
    """Load and normalize optimized_model_results.csv from a reports directory."""
    csv_path = os.path.join(reports_dir, "optimized_model_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"optimized_model_results.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    col_map = {}
    if "Model" in df.columns:
        col_map["Model"] = "model"
    if "AUROC_Mean" in df.columns:
        col_map["AUROC_Mean"] = "roc_auc_mean"
    if "AUROC_Std" in df.columns:
        col_map["AUROC_Std"] = "roc_auc_std"
    if "AUROC_CI_Lower" in df.columns:
        col_map["AUROC_CI_Lower"] = "roc_auc_ci95_lo"
    if "AUROC_CI_Upper" in df.columns:
        col_map["AUROC_CI_Upper"] = "roc_auc_ci95_hi"
    df = df.rename(columns=col_map)
    return df


def make_output_dir(base: str, label: str) -> str:
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"adv_single_{label}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def plot_bar_auroc_with_ci(df: pd.DataFrame, out_dir: str, label: str) -> None:
    """Bar chart of AUROC with 95% CI error bars."""
    # Sort by AUROC descending for nicer ordering
    df_sorted = df.sort_values("roc_auc_mean", ascending=True).copy()

    means = df_sorted["roc_auc_mean"].values
    ci_lo = df_sorted["roc_auc_ci95_lo"].values
    ci_hi = df_sorted["roc_auc_ci95_hi"].values
    # Asymmetric errors for errorbar
    yerr = [means - ci_lo, ci_hi - means]

    plt.figure(figsize=(10, 6))
    y_pos = range(len(df_sorted))

    plt.barh(
        y_pos,
        means,
        xerr=yerr,
        capsize=4,
        color="#2ECC71",
        alpha=0.8,
    )
    plt.yticks(y_pos, df_sorted["model"])
    plt.xlabel("AUROC")
    plt.title(f"Advanced Experiment AUROC with 95% CI ({label})")
    plt.xlim(0.5, 1.0)

    # Annotate values
    for i, m in enumerate(means):
        plt.text(
            m + 0.005,
            i,
            f"{m:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "bar_auroc_with_ci.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scatter_auroc_vs_std(df: pd.DataFrame, out_dir: str, label: str) -> None:
    """Scatter of AUROC vs AUROC_Std, annotated by model."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="roc_auc_mean",
        y="roc_auc_std",
        s=80,
        color="#3498DB",
    )

    for _, row in df.iterrows():
        plt.text(
            row["roc_auc_mean"] + 0.001,
            row["roc_auc_std"],
            row["model"],
            fontsize=8,
        )

    plt.xlabel("AUROC Mean")
    plt.ylabel("AUROC Std")
    plt.title(f"Advanced Experiment AUROC vs Std ({label})")
    plt.xlim(max(0.5, df["roc_auc_mean"].min() - 0.02), min(1.0, df["roc_auc_mean"].max() + 0.02))
    plt.tight_layout()

    path = os.path.join(out_dir, "scatter_auroc_vs_std.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for a single advanced experiment (optimized_model_results.csv)."
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="Path to reports directory containing optimized_model_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_visualizations",
        help="Parent directory where plots will be saved.",
    )
    parser.add_argument(
        "--label",
        default="adv_single",
        help="Label used in plot titles and output folder name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_optimized_results(args.reports_dir)
    out_dir = make_output_dir(args.output_dir, args.label)

    print(f"Loaded optimized results from: {args.reports_dir}")
    print(f"Output directory: {out_dir}")

    plot_bar_auroc_with_ci(df, out_dir, args.label)
    plot_scatter_auroc_vs_std(df, out_dir, args.label)
    print("Advanced single-experiment visualizations generated.")


