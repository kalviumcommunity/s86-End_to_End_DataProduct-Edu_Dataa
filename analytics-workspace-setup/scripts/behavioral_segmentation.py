"""Compare segment behavior using summary tables and visualizations.

This script simulates user/customer segments, computes behaviour metrics by
segment, and creates side-by-side comparisons using tables, a heatmap, and box
plots. It highlights how Enterprise, SMB, and Startup segments differ in
lifetime value, churn, support demand, and revenue.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_DIR = os.path.join("..", "output", "behavioral_segmentation")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_segment_data(n_rows: int = 15_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    customer_type = np.random.choice(
        ["enterprise", "SMB", "startup"],
        size=n_rows,
        p=[0.10, 0.35, 0.55],
    )
    lifetime_value = np.where(
        customer_type == "enterprise",
        rng.normal(loc=200_000, scale=40_000, size=n_rows),
        np.where(
            customer_type == "SMB",
            rng.normal(loc=20_000, scale=8_000, size=n_rows),
            rng.normal(loc=5_000, scale=2_000, size=n_rows),
        ),
    )
    lifetime_value = np.round(np.clip(lifetime_value, 500.0, None), 2)

    churn = np.where(
        customer_type == "enterprise",
        rng.random(n_rows) < 0.01,
        np.where(customer_type == "SMB", rng.random(n_rows) < 0.12, rng.random(n_rows) < 0.08),
    ).astype(int)
    support_tickets = np.where(
        customer_type == "enterprise",
        rng.poisson(lam=4.0, size=n_rows),
        np.where(customer_type == "SMB", rng.poisson(lam=3.0, size=n_rows), rng.poisson(lam=2.0, size=n_rows)),
    )
    feature_usage = np.round(
        np.where(
            customer_type == "enterprise",
            rng.normal(loc=75, scale=8, size=n_rows),
            np.where(customer_type == "SMB", rng.normal(loc=55, scale=10, size=n_rows), rng.normal(loc=40, scale=12, size=n_rows)),
        ),
        1,
    )
    feature_usage = np.clip(feature_usage, 5, 100)

    purchase_frequency = np.where(
        customer_type == "enterprise",
        rng.poisson(lam=12, size=n_rows),
        np.where(customer_type == "SMB", rng.poisson(lam=6, size=n_rows), rng.poisson(lam=3, size=n_rows)),
    )

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "customer_type": customer_type,
            "lifetime_value": lifetime_value,
            "churn": churn,
            "support_tickets": support_tickets,
            "feature_usage": feature_usage,
            "purchase_frequency": purchase_frequency,
        }
    )


def compute_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("customer_type").agg(
        avg_ltv=("lifetime_value", "mean"),
        churn_rate=("churn", "mean"),
        avg_support_tickets=("support_tickets", "mean"),
        avg_feature_usage=("feature_usage", "mean"),
        avg_purchase_frequency=("purchase_frequency", "mean"),
        customer_count=("customer_id", "count"),
    )
    summary["avg_ltv"] = summary["avg_ltv"].round(2)
    summary["churn_rate"] = summary["churn_rate"].round(4)
    summary["avg_support_tickets"] = summary["avg_support_tickets"].round(2)
    summary["avg_feature_usage"] = summary["avg_feature_usage"].round(1)
    summary["avg_purchase_frequency"] = summary["avg_purchase_frequency"].round(2)
    return summary


def plot_segment_heatmap(summary: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    heatmap_data = summary.copy()
    heatmap_data["avg_ltv"] = heatmap_data["avg_ltv"] / 1000
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_title("Segment Comparison Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "segment_comparison_heatmap.png"))
    plt.close(fig)


def plot_segment_boxplots(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    metrics = ["lifetime_value", "support_tickets", "feature_usage"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="customer_type", y=metric, data=df, ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Segment")
        ax.set_xlabel("Customer Segment")
        ax.set_ylabel(metric.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{metric}.png"))
        plt.close(fig)


def analyze_behavioral_segments(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    summary = compute_segment_metrics(df)
    print("Segment summary metrics:")
    print(summary.to_string())

    highest_ltv = summary["avg_ltv"].idxmax()
    lowest_ltv = summary["avg_ltv"].idxmin()
    highest_churn = summary["churn_rate"].idxmax()
    lowest_churn = summary["churn_rate"].idxmin()

    print("\nKey segment comparisons:")
    print(f"  - Highest average LTV: {highest_ltv}")
    print(f"  - Lowest average LTV: {lowest_ltv}")
    print(f"  - Highest churn rate: {highest_churn}")
    print(f"  - Lowest churn rate: {lowest_churn}")

    plot_segment_heatmap(summary)
    plot_segment_boxplots(df)

    print("\nBusiness implications:")
    print("  - Enterprise requires premium support and retention focus despite high LTV.")
    print("  - SMB may need more self-service options and churn-reduction efforts.")
    print("  - Startup segment benefits from education and product adoption support.")


def main() -> None:
    df = build_segment_data()
    analyze_behavioral_segments(df)


if __name__ == "__main__":
    main()
