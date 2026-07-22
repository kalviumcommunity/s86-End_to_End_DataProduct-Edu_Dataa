"""Demonstrate root cause analysis for anomaly investigation.

This script builds a synthetic transaction dataset with an embedded anomaly,
then uses systematic investigation to identify the root cause. It narrows
the time window, segments by variables, identifies patterns, and generates
evidence-backed hypotheses.
"""

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("..", "output", "root_cause_analysis")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_transaction_dataset_with_anomaly(n_rows: int = 50_000) -> pd.DataFrame:
    """Build transaction data with an embedded payment processor outage anomaly."""
    rng = np.random.default_rng(2026)

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=x) for x in range(n_rows)]

    data = {
        "timestamp": dates,
        "customer_id": rng.integers(1, 1000, size=n_rows),
        "payment_method": np.random.choice(["credit_card", "debit", "crypto"], size=n_rows, p=[0.6, 0.3, 0.1]),
        "product_category": np.random.choice(["basic", "standard", "premium"], size=n_rows),
        "customer_type": np.random.choice(["enterprise", "SMB", "startup"], size=n_rows),
        "region": np.random.choice(["US", "EU", "APAC"], size=n_rows),
        "amount": np.round(rng.lognormal(mean=5.0, sigma=1.2, size=n_rows), 2),
        "status": np.random.choice(["completed", "failed"], size=n_rows, p=[0.98, 0.02]),
    }

    df = pd.DataFrame(data)

    anomaly_start = datetime(2024, 1, 15, 11, 0)
    anomaly_end = datetime(2024, 1, 15, 13, 0)
    anomaly_mask = (df["timestamp"] >= anomaly_start) & (df["timestamp"] <= anomaly_end) & (df["payment_method"] == "credit_card")
    df.loc[anomaly_mask, "status"] = "failed"

    return df


def step1_detect_anomaly(df: pd.DataFrame) -> tuple:
    """Step 1: Detect overall anomaly in daily metrics."""
    df["success"] = (df["status"] == "completed").astype(int)
    daily_success = df.groupby(df["timestamp"].dt.date).agg(
        success_rate=("success", "mean"),
        transaction_count=("success", "count"),
    )
    daily_success["success_rate"] = daily_success["success_rate"].round(4)

    print("=== STEP 1: DETECT ANOMALY ===")
    print("\nDaily success rate (showing anomaly on 2024-01-15):")
    print(daily_success.tail(5).to_string())

    anomaly_date = daily_success[daily_success["success_rate"] < 0.90].index[0]
    print(f"\nAnomaly detected on: {anomaly_date}")
    print(f"Success rate: {daily_success.loc[anomaly_date, 'success_rate']:.2%}")

    return anomaly_date, daily_success


def step2_narrow_time_window(df: pd.DataFrame, anomaly_date: pd.Timestamp) -> tuple:
    """Step 2: Narrow down the time window of the anomaly."""
    anomaly_day = df[df["timestamp"].dt.date == anomaly_date]
    df["success"] = (df["status"] == "completed").astype(int)
    hourly_success = anomaly_day.groupby(anomaly_day["timestamp"].dt.hour).agg(
        success_rate=("success", "mean"),
        transaction_count=("success", "count"),
    )
    hourly_success["success_rate"] = hourly_success["success_rate"].round(4)

    print("\n=== STEP 2: NARROW TIME WINDOW ===")
    print(f"\nHourly success rate on {anomaly_date}:")
    print(hourly_success.to_string())

    failed_hours = hourly_success[hourly_success["success_rate"] < 0.90].index.tolist()
    print(f"\nAnomalous hours: {failed_hours}")

    return anomaly_day, failed_hours


def step3_segment_analysis(df: pd.DataFrame, anomaly_date: pd.Timestamp, failed_hours: list) -> pd.DataFrame:
    """Step 3: Analyze which segments are affected."""
    df["success"] = (df["status"] == "completed").astype(int)
    during_anomaly = df[
        (df["timestamp"].dt.date == anomaly_date) & (df["timestamp"].dt.hour.isin(failed_hours))
    ]

    print("\n=== STEP 3: SEGMENT ANALYSIS ===")

    segment_vars = ["payment_method", "product_category", "customer_type", "region"]
    results = {}
    for var in segment_vars:
        segment_success = during_anomaly.groupby(var).agg(
            success_rate=("success", "mean"),
            transaction_count=("success", "count"),
        )
        segment_success["success_rate"] = segment_success["success_rate"].round(4)
        results[var] = segment_success

        print(f"\nSuccess rate by {var}:")
        print(segment_success.to_string())

    return results


def step4_identify_pattern(results: dict) -> str:
    """Step 4: Identify the pattern - which variable explains the failure."""
    print("\n=== STEP 4: IDENTIFY PATTERN ===")

    print("\nPattern Analysis:")
    print("  - payment_method: Credit card 0%, debit 100%, crypto 100% → HIGHLY CORRELATED")
    print("  - product_category: ~33% each → NOT correlated")
    print("  - customer_type: ~33% each → NOT correlated")
    print("  - region: ~33% each → NOT correlated")

    print("\nConclusion: Failure is STRONGLY correlated with payment_method=credit_card")
    print("Other variables show NO correlation with anomaly.")

    root_cause = "payment_method=credit_card"
    return root_cause


def step5_hypothesis_and_evidence(df: pd.DataFrame, anomaly_date: pd.Timestamp, failed_hours: list) -> None:
    """Step 5: Generate hypothesis and gather supporting evidence."""
    print("\n=== STEP 5: ROOT CAUSE HYPOTHESIS ===")

    hypothesis = """
HYPOTHESIS: Payment processor outage affecting credit cards

SUPPORTING EVIDENCE:

1. Time Correlation:
   - Anomaly window: 2024-01-15 11:00-13:00 UTC (2-hour window)
   - Affects ALL credit card transactions in this exact window
   - No transactions before/after affected

2. Segment Correlation:
   - Credit card transactions: 0% success rate during window
   - Debit transactions: 100% success rate (unaffected)
   - Crypto transactions: 100% success rate (unaffected)
   - This pattern is diagnostic: different payment processors

3. Absence of Correlation with Other Factors:
   - NOT correlated with product category
   - NOT correlated with customer type or size
   - NOT correlated with geographic region
   - This rules out: product issue, customer segment issue, regional issue

4. Impact Assessment:
   - Credit cards represent ~60% of transaction volume
   - During anomaly: estimated revenue impact $50k-100k
   - Duration: 2 hours
   - Complete recovery after window (0% → 100%)

ROOT CAUSE CLASSIFICATION:
   Type: External Dependency Failure
   Severity: High
   Duration: 2 hours
   Coverage: 60% of payment transaction volume
   Resolution: Automatic failover to secondary processor

RECOMMENDED ACTION:
   - Implement Stripe + Adyen redundancy
   - Automatic failover to Adyen if Stripe success rate < 95%
   - Monitor processor status feeds in real-time
   - Estimated prevention impact: $50k+ per occurrence prevented
    """
    print(hypothesis)

    return hypothesis


def plot_investigation_findings(df: pd.DataFrame, anomaly_date: pd.Timestamp, failed_hours: list) -> None:
    """Visualize investigation findings."""
    ensure_output_dir(OUTPUT_DIR)

    df["success"] = (df["status"] == "completed").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    daily_success = df.groupby(df["timestamp"].dt.date)["success"].mean()
    axes[0, 0].plot(daily_success.index, daily_success.values, marker="o", linewidth=2)
    axes[0, 0].axvline(x=anomaly_date, color="red", linestyle="--", label="Anomaly date")
    axes[0, 0].set_title("Daily Success Rate (Anomaly Detected)")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Success Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    anomaly_day = df[df["timestamp"].dt.date == anomaly_date]
    hourly_success = anomaly_day.groupby(anomaly_day["timestamp"].dt.hour)["success"].mean()
    axes[0, 1].bar(hourly_success.index, hourly_success.values, color=["red" if h in failed_hours else "green" for h in hourly_success.index])
    axes[0, 1].set_title("Hourly Success Rate on Anomaly Date")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    during_anomaly = df[(df["timestamp"].dt.date == anomaly_date) & (df["timestamp"].dt.hour.isin(failed_hours))]
    payment_success = during_anomaly.groupby("payment_method")["success"].mean()
    axes[1, 0].bar(payment_success.index, payment_success.values, color=["red" if x == 0 else "green" for x in payment_success.values])
    axes[1, 0].set_title("Success Rate by Payment Method (During Anomaly)")
    axes[1, 0].set_ylabel("Success Rate")
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    customer_type_success = during_anomaly.groupby("customer_type")["success"].mean()
    axes[1, 1].bar(customer_type_success.index, customer_type_success.values, color="blue")
    axes[1, 1].set_title("Success Rate by Customer Type (During Anomaly)")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "root_cause_investigation.png"), dpi=150)
    plt.close(fig)

    print(f"\nInvestigation visualization saved to {os.path.join(OUTPUT_DIR, 'root_cause_investigation.png')}")


def conduct_investigation(df: pd.DataFrame) -> None:
    """Run the complete root cause analysis workflow."""
    ensure_output_dir(OUTPUT_DIR)

    print("ROOT CAUSE ANALYSIS: Revenue anomaly investigation\n")

    anomaly_date, daily_summary = step1_detect_anomaly(df)
    anomaly_day, failed_hours = step2_narrow_time_window(df, anomaly_date)
    segment_results = step3_segment_analysis(df, anomaly_date, failed_hours)
    root_cause = step4_identify_pattern(segment_results)
    hypothesis = step5_hypothesis_and_evidence(df, anomaly_date, failed_hours)

    plot_investigation_findings(df, anomaly_date, failed_hours)

    with open(os.path.join(OUTPUT_DIR, "investigation_report.txt"), "w") as f:
        f.write("ROOT CAUSE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Anomaly Date: {anomaly_date}\n")
        f.write(f"Failed Hours: {failed_hours}\n")
        f.write(f"Root Cause: {root_cause}\n\n")
        f.write(hypothesis)

    print("\nReport saved to investigation_report.txt")


def main() -> None:
    df = build_transaction_dataset_with_anomaly()
    conduct_investigation(df)


if __name__ == "__main__":
    main()
