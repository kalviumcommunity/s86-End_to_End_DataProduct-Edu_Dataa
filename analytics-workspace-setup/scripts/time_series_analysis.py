"""Demonstrate time-series trend analysis with rolling metrics and resampling.

This script builds a synthetic daily revenue dataset, computes moving averages,
resamples by week and month, calculates period-over-period change rates, and
plots trend visualizations alongside business interpretation guidance.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("..", "output", "time_series_analysis")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_daily_revenue_series(start_date: str = "2024-01-01", periods: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    trend = np.linspace(40_000, 60_000, periods)
    seasonality = 5_000 * np.sin(np.arange(periods) * 2 * np.pi / 30)
    noise = rng.normal(loc=0.0, scale=4_000, size=periods)
    revenue = np.clip(trend + seasonality + noise, 10_000, None)

    return pd.DataFrame({"date": dates, "revenue": revenue})


def compute_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["revenue_ma7"] = df["revenue"].rolling(window=7, min_periods=1).mean()
    df["revenue_ma30"] = df["revenue"].rolling(window=30, min_periods=1).mean()
    df["cumulative_revenue"] = df["revenue"].cumsum()
    df["weekly_revenue"] = df["revenue"].resample("W").sum()
    df["monthly_revenue"] = df["revenue"].resample("M").sum()
    df["weekly_pct_change"] = df["weekly_revenue"].pct_change() * 100
    df["monthly_pct_change"] = df["monthly_revenue"].pct_change() * 100

    return df


def plot_raw_vs_rolling(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["revenue"], label="Daily Revenue", alpha=0.3)
    ax.plot(df.index, df["revenue_ma7"], label="7-Day MA", linewidth=2)
    ax.plot(df.index, df["revenue_ma30"], label="30-Day MA", linewidth=2)
    ax.set_title("Daily Revenue with Rolling Averages")
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "revenue_raw_vs_rolling.png"))
    plt.close(fig)


def plot_cumulative_revenue(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["cumulative_revenue"], color="tab:green")
    ax.set_title("Cumulative Revenue Over Time")
    ax.set_ylabel("Cumulative Revenue")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_revenue.png"))
    plt.close(fig)


def plot_resampled_change(series: pd.Series, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(series.index.to_period(series.index.freqstr).to_timestamp(), series, color="tab:purple", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Percent Change")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


def identify_trends(df: pd.DataFrame) -> list:
    insights = []
    monthly = df["monthly_revenue"].dropna()

    if len(monthly) >= 4:
        recent_change = monthly.iloc[-1] - monthly.iloc[-4]
        if recent_change > 0:
            insights.append("Uptrend: revenue has risen over the last three months.")
        elif recent_change < 0:
            insights.append("Downtrend: revenue has declined over the last three months.")
        else:
            insights.append("Flat trend: revenue is stable over the last three months.")

    if df["revenue_ma7"].iloc[-1] > df["revenue_ma30"].iloc[-1]:
        insights.append("Short-term momentum is positive: 7-day average exceeds 30-day average.")
    else:
        insights.append("Short-term momentum is weak: 7-day average is below 30-day average.")

    return insights


def analyze_time_series(df: pd.DataFrame) -> None:
    df_features = compute_time_series_features(df)
    print("Latest period-over-period changes:")
    print(df_features[["weekly_pct_change", "monthly_pct_change"]].dropna().tail(5).round(2).to_string())

    plot_raw_vs_rolling(df_features)
    plot_cumulative_revenue(df_features)
    plot_resampled_change(
        df_features["weekly_pct_change"].dropna(),
        "Week-over-Week Revenue Change (%)",
        "weekly_pct_change.png",
    )
    plot_resampled_change(
        df_features["monthly_pct_change"].dropna(),
        "Month-over-Month Revenue Change (%)",
        "monthly_pct_change.png",
    )

    print("\nTrend interpretation:")
    for insight in identify_trends(df_features):
        print(f"  - {insight}")

    print("\nTime-series recommendation:")
    print("  - Use rolling averages to distinguish noise from true trend.")
    print("  - Compare current period to the prior period before acting.")
    print("  - Cumulative revenue shows total progress, not daily volatility.")


def main() -> None:
    df = build_daily_revenue_series()
    analyze_time_series(df)


if __name__ == "__main__":
    main()
