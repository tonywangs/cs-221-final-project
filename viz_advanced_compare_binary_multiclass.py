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


def load_optimized_results(path: str) -> pd.DataFrame:
    """Load optimized_model_results.csv and normalize column names."""
    csv_path = os.path.join(path, "optimized_model_results.csv")
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
    out = os.path.join(base, f"adv_binary_vs_multiclass_{label}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def plot_bar_comparison(df_merged: pd.DataFrame, out_dir: str, label: str) -> None:
    """Bar chart: AUROC (binary vs multiclass) per model."""
    plt.figure(figsize=(10, 6))
    x = range(len(df_merged))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        df_merged["roc_auc_mean_bin"],
        width=width,
        label="Binary",
        color="#2ECC71",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df_merged["roc_auc_mean_multi"],
        width=width,
        label="Multiclass (macro AUROC)",
        color="#3498DB",
    )

    plt.xticks(x, df_merged["model"], rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.title(f"Advanced Models: Binary vs Multiclass AUROC ({label})")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "bar_binary_vs_multiclass_auroc.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scatter_comparison(df_merged: pd.DataFrame, out_dir: str, label: str) -> None:
    """Scatter: binary vs multiclass AUROC per model."""
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=df_merged,
        x="roc_auc_mean_bin",
        y="roc_auc_mean_multi",
        s=80,
        color="#9B59B6",
    )

    # Diagonal y=x line
    lo = min(df_merged["roc_auc_mean_bin"].min(), df_merged["roc_auc_mean_multi"].min())
    hi = max(df_merged["roc_auc_mean_bin"].max(), df_merged["roc_auc_mean_multi"].max())
    lo = max(0.5, lo - 0.02)
    hi = min(1.0, hi + 0.02)
    plt.plot([lo, hi], [lo, hi], ls="--", color="gray", label="y = x")

    # Annotate points with model names
    for _, row in df_merged.iterrows():
        plt.text(
            row["roc_auc_mean_bin"] + 0.002,
            row["roc_auc_mean_multi"],
            row["model"],
            fontsize=8,
        )

    plt.xlabel("Binary AUROC")
    plt.ylabel("Multiclass AUROC (macro OVR)")
    plt.title(f"Binary vs Multiclass AUROC per Model ({label})")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(out_dir, "scatter_binary_vs_multiclass_auroc.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare advanced binary and multiclass experiments."
    )
    parser.add_argument(
        "--binary-reports-dir",
        required=True,
        help="Path to binary advanced reports dir (contains optimized_model_results.csv).",
    )
    parser.add_argument(
        "--multiclass-reports-dir",
        required=True,
        help="Path to multiclass advanced reports dir (contains optimized_model_results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_visualizations",
        help="Parent directory where comparison plots will be saved.",
    )
    parser.add_argument(
        "--label",
        default="adv_202512",
        help="Label used in plot titles/output folder name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df_bin = load_optimized_results(args.binary_reports_dir)
    df_multi = load_optimized_results(args.multiclass_reports_dir)

    # Align on common models
    df_bin = df_bin[["model", "roc_auc_mean", "roc_auc_std"]].rename(
        columns={
            "roc_auc_mean": "roc_auc_mean_bin",
            "roc_auc_std": "roc_auc_std_bin",
        }
    )
    df_multi = df_multi[["model", "roc_auc_mean", "roc_auc_std"]].rename(
        columns={
            "roc_auc_mean": "roc_auc_mean_multi",
            "roc_auc_std": "roc_auc_std_multi",
        }
    )

    df_merged = pd.merge(df_bin, df_multi, on="model", how="inner")
    if df_merged.empty:
        raise ValueError(
            "No overlapping models between binary and multiclass runs. "
            "Check that both optimized_model_results.csv files use the same 'Model' names."
        )

    out_dir = make_output_dir(args.output_dir, args.label)
    print(f"Binary reports:     {args.binary_reports_dir}")
    print(f"Multiclass reports: {args.multiclass_reports_dir}")
    print(f"Comparison outputs: {out_dir}")

    plot_bar_comparison(df_merged, out_dir, args.label)
    plot_scatter_comparison(df_merged, out_dir, args.label)
    print("Advanced binary vs multiclass visualizations generated.")


