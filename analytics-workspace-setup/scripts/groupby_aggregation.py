"""Demonstrate Pandas groupby aggregation and segment insights.

This script generates a sample business dataset, then uses groupby and pivot
operations to compute churn rate, revenue, and customer counts by segment.
It ranks segments, compares product performance, and prints business insights.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("..", "output", "groupby_aggregation")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_segmented_dataset(n_rows: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    customer_types = np.random.choice(
        ["enterprise", "SMB", "startup"],
        size=n_rows,
        p=[0.05, 0.40, 0.55],
    )
    products = np.random.choice(
        ["basic", "standard", "premium"],
        size=n_rows,
        p=[0.45, 0.35, 0.20],
    )
    revenue = np.round(
        np.where(
            customer_types == "enterprise",
            rng.normal(loc=120_000, scale=30_000, size=n_rows),
            np.where(
                customer_types == "SMB",
                rng.normal(loc=18_000, scale=8_000, size=n_rows),
                rng.normal(loc=6_000, scale=3_000, size=n_rows),
            ),
        ),
        2,
    )
    revenue = np.clip(revenue, 50.0, None)

    churn_probs = np.where(
        customer_types == "enterprise",
        0.01,
        np.where(customer_types == "SMB", 0.12, 0.08),
    )
    churn = rng.random(n_rows) < churn_probs

    customer_id = np.arange(1, n_rows + 1)
    order_count = rng.poisson(lam=3.0 + (revenue / 50_000), size=n_rows) + 1

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "customer_type": customer_types,
            "product": products,
            "revenue": revenue,
            "order_count": order_count,
            "churn": churn.astype(int),
        }
    )


def segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("customer_type").agg(
        total_revenue=("revenue", "sum"),
        average_revenue=("revenue", "mean"),
        customer_count=("customer_id", "count"),
        churned_customers=("churn", "sum"),
        churn_rate=("churn", "mean"),
    )
    grouped["revenue_share"] = grouped["total_revenue"] / grouped["total_revenue"].sum()
    grouped["churn_rate"] = grouped["churn_rate"].round(4)
    grouped["average_revenue"] = grouped["average_revenue"].round(2)
    grouped["revenue_share"] = grouped["revenue_share"].round(4)
    return grouped.sort_values(["churn_rate", "total_revenue"], ascending=[False, False])


def product_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.pivot_table(
        df,
        values="revenue",
        index="customer_type",
        columns="product",
        aggfunc="sum",
        fill_value=0,
    )
    return pivot


def rank_segments(df: pd.DataFrame) -> pd.DataFrame:
    metrics = segment_metrics(df).copy()
    metrics["churn_rank"] = metrics["churn_rate"].rank(method="dense", ascending=False).astype(int)
    metrics["revenue_rank"] = metrics["total_revenue"].rank(method="dense", ascending=False).astype(int)
    return metrics.sort_values(["churn_rank", "revenue_rank"])


def assign_segment_churn_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segment_churn_rate"] = df.groupby("customer_type")["churn"].transform("mean")
    return df


def plot_segment_metrics(metrics: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics["total_revenue"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Total Revenue by Customer Segment")
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel("Customer Segment")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "segment_revenue.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics["churn_rate"].plot(kind="bar", ax=ax, color="salmon")
    ax.set_title("Churn Rate by Customer Segment")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Customer Segment")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "segment_churn_rate.png"))
    plt.close(fig)


def analyze_segments(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    metrics = rank_segments(df)
    pivot = product_pivot(df)
    assigned = assign_segment_churn_rate(df)

    print("Segment metrics:")
    print(metrics.to_string())

    print("\nRevenue pivot table by customer_type and product:")
    print(pivot.round(2).to_string())

    print("\nExample rows with segment churn rate assigned:")
    print(assigned.head(5).to_string(index=False))

    plot_segment_metrics(metrics)
    pivot_path = os.path.join(OUTPUT_DIR, "segment_product_revenue.csv")
    metrics.to_csv(os.path.join(OUTPUT_DIR, "segment_metrics.csv"))
    pivot.to_csv(pivot_path)

    print("\nActionable insight examples:")
    for segment, row in metrics.iterrows():
        print(
            f"  - {segment.title()} segment has {row['churn_rate']:.2%} churn, "
            f"generates {row['revenue_share']:.2%} of revenue, and has "
            f"{row['customer_count']} customers."
        )
        if row["churn_rate"] > 0.10:
            print("    Suggestion: prioritize retention actions for this segment.")
        elif row["revenue_share"] > 0.5:
            print("    Suggestion: maintain experience for this high-revenue segment.")


def main() -> None:
    df = build_segmented_dataset()
    analyze_segments(df)


if __name__ == "__main__":
    main()
