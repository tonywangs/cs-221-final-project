import argparse
import os
from datetime import datetime
from typing import List

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


class RealMetricsPlotter:
    """Helper that loads report CSVs and produces real-data plots."""

    def __init__(
        self,
        reports_dir: str,
        output_dir: str = "paper_visualizations",
        label: str = "original",
    ):
        self.reports_dir = reports_dir
        self.label = label

        self.summary_path = os.path.join(reports_dir, "summary_metrics.csv")
        self.fold_path = os.path.join(reports_dir, "fold_metrics.csv")
        self.leaderboard_path = os.path.join(reports_dir, "leaderboard.csv")
        self.optimized_path = os.path.join(reports_dir, "optimized_model_results.csv")

        self.summary_df: pd.DataFrame = None
        self.fold_df: pd.DataFrame = None
        self.mode: str = None  # 'cv' for original pipeline, 'advanced' for optimized-only

        self._load_data()

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"metrics_{self.label}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Reports directory: {self.reports_dir}")
        print(f"Output directory:  {self.output_dir}")

    def _load_data(self):
        # Case 1: Original CV pipeline (summary_metrics + fold_metrics)
        if os.path.exists(self.summary_path) and os.path.exists(self.fold_path):
            self.summary_df = pd.read_csv(self.summary_path)
            self.fold_df = pd.read_csv(self.fold_path)
            self.mode = "cv"
            return

        # Case 2: Advanced run with only optimized_model_results.csv
        if os.path.exists(self.optimized_path):
            df = pd.read_csv(self.optimized_path)
            # Normalize column names to match the CV-style expectations where possible
            # Expected: 'model', 'roc_auc_mean', 'roc_auc_std', 'roc_auc_ci95_lo', 'roc_auc_ci95_hi'
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
            self.summary_df = df
            self.fold_df = None
            self.mode = "advanced"
            print("Warning: Using optimized_model_results.csv (no per-fold metrics available).")
            return

        # If neither format is found, raise a clear error
        raise FileNotFoundError(
            f"Could not find either summary_metrics.csv/fold_metrics.csv or "
            f"optimized_model_results.csv under {self.reports_dir}"
        )

    # ------------------------------------------------------------------
    # Plot generators
    # ------------------------------------------------------------------
    def plot_leaderboard(self, metric: str = "roc_auc_mean", top_n: int = 10):
        """Horizontal bar chart of the top-N models for a given metric."""
        if metric not in self.summary_df.columns:
            raise ValueError(f"Metric '{metric}' not in summary_metrics.csv columns.")

        top = (
            self.summary_df.sort_values(metric, ascending=False)
            .head(top_n)
            .copy()
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=top,
            y="model",
            x=metric,
            color="#2ECC71",
        )
        plt.xlabel(metric.replace("_", " ").upper())
        plt.ylabel("Model")
        plt.title(f"Top {top_n} Models by {metric.replace('_', ' ').title()} ({self.label})")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"leaderboard_{metric}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    def plot_fold_metric_distribution(
        self,
        metric: str = "roc_auc",
        focus_metric: str = "roc_auc_mean",
        top_n: int = 8,
    ):
        """Boxplot of per-fold metrics for the top-N models (based on focus_metric)."""
        if self.mode != "cv" or self.fold_df is None:
            print("Warning: Skipping fold metric distribution: no per-fold metrics available for this run.")
            return
        if metric not in self.fold_df.columns:
            raise ValueError(f"Metric '{metric}' not found in fold_metrics.csv.")
        if focus_metric not in self.summary_df.columns:
            raise ValueError(f"Metric '{focus_metric}' not in summary_metrics.csv.")

        top_models = (
            self.summary_df.sort_values(focus_metric, ascending=False)["model"]
            .head(top_n)
            .tolist()
        )
        data = self.fold_df[self.fold_df["model"].isin(top_models)]
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=data,
            x="model",
            y=metric,
            palette="Blues",
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(
            f"{metric.replace('_', ' ').title()} Distribution per Fold "
            f"(Top {top_n} models by {focus_metric.replace('_', ' ').title()})"
        )
        plt.ylabel(metric.replace("_", " ").upper())
        plt.xlabel("Model")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"fold_distribution_{metric}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    def plot_metric_scatter(
        self,
        x_metric: str = "roc_auc_mean",
        y_metric: str = "brier_mean",
        annotate: bool = True,
    ):
        """Scatter plot comparing two summary metrics."""
        for col in [x_metric, y_metric]:
            if col not in self.summary_df.columns:
                print(f"Warning: Skipping scatter {x_metric} vs {y_metric}: '{col}' not in summary metrics.")
                return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.summary_df,
            x=x_metric,
            y=y_metric,
            s=80,
            color="#3498DB",
        )
        if annotate:
            for _, row in self.summary_df.iterrows():
                plt.text(
                    row[x_metric] + 0.001,
                    row[y_metric],
                    row["model"],
                    fontsize=8,
                )
        plt.xlabel(x_metric.replace("_", " ").title())
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()} ({self.label})")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"scatter_{x_metric}_vs_{y_metric}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    def plot_metric_heatmap(self, metrics: List[str]):
        """Heatmap of selected metrics (rows=models, cols=metrics)."""
        # Keep only metrics that actually exist
        metrics_present = [m for m in metrics if m in self.summary_df.columns]
        if not metrics_present:
            print("Warning: Skipping heatmap: none of the requested metrics are present.")
            return

        heatmap_data = (
            self.summary_df.set_index("model")[metrics_present]
            .sort_values(metrics_present[0], ascending=False)
        )
        plt.figure(figsize=(max(8, len(metrics_present) * 1.6), 0.4 * len(heatmap_data) + 2))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Score"},
        )
        plt.title(f"Metric Heatmap ({self.label})")
        plt.ylabel("Model")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"heatmap_{'_'.join(metrics_present)}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    def generate_all(
        self,
        leaderboard_metric: str = "roc_auc_mean",
        box_metric: str = "roc_auc",
        heatmap_metrics: List[str] = None,
    ):
        """Generate a default suite of real-data plots."""
        if heatmap_metrics is None:
            if self.mode == "cv":
                heatmap_metrics = [
                    "roc_auc_mean",
                    "pr_auc_mean",
                    "brier_mean",
                    "ece_mean",
                ]
            else:
                # For advanced runs we typically only have AUROC stats
                heatmap_metrics = [
                    "roc_auc_mean",
                    "roc_auc_std",
                ]
        print("🎨 Generating real-data metric plots...")
        self.plot_leaderboard(metric=leaderboard_metric)
        self.plot_fold_metric_distribution(metric=box_metric)
        # Only scatter metrics that exist
        if self.mode == "cv":
            self.plot_metric_scatter("roc_auc_mean", "brier_mean")
            self.plot_metric_scatter("roc_auc_mean", "ece_mean")
        self.plot_metric_heatmap(heatmap_metrics)
        print("All plots generated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-data metric plots.")
    parser.add_argument(
        "--reports-dir",
        default="outputs/exp_binary_cv10_seed42_20251112_201223/reports",
        help="Directory containing summary_metrics.csv and fold_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_visualizations",
        help="Parent directory for saving plots.",
    )
    parser.add_argument(
        "--label",
        default="original",
        help="Label used in plot titles/output folder names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plotter = RealMetricsPlotter(
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
        label=args.label,
    )
    plotter.generate_all()

