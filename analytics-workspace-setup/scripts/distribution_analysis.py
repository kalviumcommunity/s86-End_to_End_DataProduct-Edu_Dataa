"""Analyze numeric distribution shapes and business implications.

This script computes skewness and kurtosis, plots histograms and KDE, and
compares high-value and low-value segments for a revenue-style distribution.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_DIR = os.path.join("..", "output", "distribution_analysis")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_distribution_metrics(series: pd.Series) -> dict:
    values = series.dropna().astype(float)
    return {
        "count": int(values.size),
        "mean": values.mean(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
    }


def interpret_distribution(metrics: dict) -> list:
    interpretation = []
    skewness = metrics["skewness"]
    kurtosis = metrics["kurtosis"]

    if abs(skewness) < 0.5:
        interpretation.append("Distribution is roughly symmetric.")
    elif skewness > 0:
        interpretation.append(
            "Positive skew: a few large values pull the mean to the right."
        )
        interpretation.append("Median is more representative than mean.")
    else:
        interpretation.append(
            "Negative skew: a few small values pull the mean to the left."
        )

    if kurtosis > 3:
        interpretation.append(
            "High kurtosis: heavy tails indicate extreme outliers are likely."
        )
    elif kurtosis < 1:
        interpretation.append(
            "Low kurtosis: distribution is flatter and less concentrated."
        )
    else:
        interpretation.append("Kurtosis is near normal, with moderate tail behavior.")

    return interpretation


def plot_histogram_and_kde(series: pd.Series, title: str, filename: str) -> None:
    values = series.dropna().astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(values, bins=50, alpha=0.6, edgecolor="black", density=True)
    values.plot(kind="density", ax=ax, label="KDE", color="red")

    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def compare_high_low_segments(df: pd.DataFrame, column: str) -> pd.DataFrame:
    quarter_25 = df[column].quantile(0.25)
    quarter_75 = df[column].quantile(0.75)

    low_value = df[df[column] <= quarter_25]
    high_value = df[df[column] >= quarter_75]

    metrics = {
        "segment": ["low_value", "high_value"],
        "count": [len(low_value), len(high_value)],
        "mean": [low_value[column].mean(), high_value[column].mean()],
        "median": [low_value[column].median(), high_value[column].median()],
        "skewness": [stats.skew(low_value[column].dropna()), stats.skew(high_value[column].dropna())],
        "kurtosis": [stats.kurtosis(low_value[column].dropna()), stats.kurtosis(high_value[column].dropna())],
    }

    return pd.DataFrame(metrics)


def analyze_revenue_distribution(df: pd.DataFrame, column: str = "revenue") -> None:
    ensure_output_dir(OUTPUT_DIR)

    metrics = compute_distribution_metrics(df[column])
    interpretation = interpret_distribution(metrics)
    segment_comparison = compare_high_low_segments(df, column)

    print("Distribution metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nInterpretation:")
    for line in interpretation:
        print(f"  - {line}")

    print("\nSegment comparison:")
    print(segment_comparison.to_string(index=False))

    plot_histogram_and_kde(
        df[column],
        title=f"{column.title()} Distribution with Histogram and KDE",
        filename=os.path.join(OUTPUT_DIR, "revenue_distribution.png"),
    )

    plot_histogram_and_kde(
        df[df[column] >= df[column].quantile(0.75)][column],
        title=f"High-Value {column.title()} Distribution",
        filename=os.path.join(OUTPUT_DIR, "revenue_high_value_distribution.png"),
    )

    plot_histogram_and_kde(
        df[df[column] <= df[column].quantile(0.25)][column],
        title=f"Low-Value {column.title()} Distribution",
        filename=os.path.join(OUTPUT_DIR, "revenue_low_value_distribution.png"),
    )


def build_sample_revenue_df(n_rows: int = 100_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    small_customers = rng.lognormal(mean=5.5, sigma=0.7, size=int(n_rows * 0.85))
    enterprise_customers = rng.lognormal(mean=10.5, sigma=0.5, size=int(n_rows * 0.15))
    revenue = np.concatenate([small_customers, enterprise_customers])
    revenue = np.clip(revenue, 10.0, None)

    return pd.DataFrame({"revenue": revenue})


if __name__ == "__main__":
    df_sample = build_sample_revenue_df()
    analyze_revenue_distribution(df_sample)
