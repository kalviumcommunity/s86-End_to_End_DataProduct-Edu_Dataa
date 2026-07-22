"""Analyze relationships using Pearson and Spearman correlation.

This script generates a synthetic business-style dataset, computes Pearson and
Spearman correlation matrices, visualizes relationships with heatmaps, and
identifies the strongest feature pairs. It also prints interpretation hints
about correlation strength and causation caution.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


OUTPUT_DIR = os.path.join("..", "output", "correlation_analysis")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_sample_business_df(n_rows: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    revenue = rng.lognormal(mean=8.5, sigma=1.5, size=n_rows)
    discount_rate = np.clip(rng.normal(loc=0.1, scale=0.05, size=n_rows), 0.0, 0.35)
    base_margin = np.clip(0.5 - discount_rate * 1.5 + rng.normal(scale=0.05, size=n_rows), 0.1, 0.75)
    support_tickets = np.random.poisson(lam=np.clip(revenue / 10000 + 1, 1, 10), size=n_rows)
    engagement_score = np.clip(0.3 + np.log1p(revenue) / 15 + rng.normal(scale=0.1, size=n_rows), 0.0, 1.0)
    retention_score = np.clip(0.9 - 0.2 * support_tickets / (support_tickets + 5) + rng.normal(scale=0.05, size=n_rows), 0.0, 1.0)
    churn_risk = np.clip(1.0 - retention_score + support_tickets * 0.03 + rng.normal(scale=0.03, size=n_rows), 0.0, 1.0)

    return pd.DataFrame(
        {
            "revenue": revenue,
            "discount_rate": discount_rate,
            "margin": base_margin,
            "support_tickets": support_tickets,
            "engagement_score": engagement_score,
            "retention_score": retention_score,
            "churn_risk": churn_risk,
        }
    )


def compute_correlation_matrices(df: pd.DataFrame) -> dict:
    return {
        "pearson": df.corr(method="pearson"),
        "spearman": df.corr(method="spearman"),
    }


def extract_strong_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    stacked = corr_matrix.unstack()
    strong = stacked[stacked.abs() >= threshold].reset_index()
    strong.columns = ["feature_1", "feature_2", "correlation"]
    strong = strong[strong["feature_1"] != strong["feature_2"]]
    strong["abs_correlation"] = strong["correlation"].abs()
    strong = strong.sort_values(["abs_correlation", "feature_1", "feature_2"], ascending=[False, True, True])
    strong = strong.drop_duplicates(subset=["abs_correlation", "feature_1", "feature_2"])
    return strong.drop(columns=["abs_correlation"]).reset_index(drop=True)


def plot_heatmap(corr_matrix: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_top_relationships(df: pd.DataFrame, strong_pairs: pd.DataFrame, max_pairs: int = 4) -> None:
    pairs = strong_pairs.head(max_pairs)
    for index, row in pairs.iterrows():
        x, y = row["feature_1"], row["feature_2"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[x], y=df[y], alpha=0.4, edgecolor=None, ax=ax)
        ax.set_title(f"{x} vs {y} (corr={row['correlation']:.2f})")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"scatter_{x}_vs_{y}.png"))
        plt.close(fig)


def analyze_relationships(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    correlations = compute_correlation_matrices(df)

    for method, matrix in correlations.items():
        print(f"\n=== {method.title()} Correlation ===")
        print(matrix.round(2).to_string())
        heatmap_path = os.path.join(OUTPUT_DIR, f"correlation_heatmap_{method}.png")
        plot_heatmap(matrix, f"{method.title()} Correlation Matrix", heatmap_path)

        strong_pairs = extract_strong_pairs(matrix)
        if strong_pairs.empty:
            print("No relationships exceed the threshold for strong correlation.")
        else:
            print("\nTop strong relationships:")
            print(strong_pairs.head(10).to_string(index=False))
            plot_top_relationships(df, strong_pairs)

    print("\nNote: correlation measures association, not causation.")
    print("Ask whether a third factor could explain the relationship before drawing business conclusions.")


def summarize_causation_guidance() -> None:
    print("\nCausation guidance:")
    print("  - Correlation says features move together, not that one causes the other.")
    print("  - Strong correlation can indicate redundancy for modeling.")
    print("  - Use domain knowledge to decide which correlated feature is more meaningful.")


if __name__ == "__main__":
    df = build_sample_business_df()
    analyze_relationships(df)
    summarize_causation_guidance()
