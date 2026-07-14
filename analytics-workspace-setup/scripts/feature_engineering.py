import json
import os

import pandas as pd


def _safe_ratio(numerator, denominator):
    """
    Compute a ratio while avoiding divide-by-zero errors.
    """
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator


def _safe_qcut(series, labels, fallback_value):
    """
    Apply qcut when possible, otherwise fall back to a single default bucket.
    """
    unique_values = series.dropna().nunique()
    if unique_values < 2:
        return pd.Series([fallback_value] * len(series), index=series.index)

    quantile_count = min(len(labels), unique_values)
    try:
        binned = pd.qcut(
            series.rank(method="first"),
            q=quantile_count,
            labels=labels[:quantile_count],
            duplicates="drop",
        )
        return binned.astype(str)
    except ValueError:
        return pd.Series([fallback_value] * len(series), index=series.index)


def build_customer_feature_table(df):
    """
    Create customer-level derived business columns.
    """
    required_columns = {"customer_id", "transaction_date", "amount"}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "Required columns for feature engineering were not available.",
            "required_columns": sorted(required_columns),
        }

    working_df = df.copy()
    working_df["transaction_date"] = pd.to_datetime(
        working_df["transaction_date"],
        errors="coerce",
    )
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")

    aggregated = (
        working_df.dropna(subset=["customer_id"]) 
        .groupby("customer_id", as_index=False)
        .agg(
            total_transactions=("customer_id", "size"),
            total_spent=("amount", "sum"),
            first_transaction_date=("transaction_date", "min"),
            last_transaction_date=("transaction_date", "max"),
        )
    )

    aggregated["days_as_customer"] = (
        aggregated["last_transaction_date"] - aggregated["first_transaction_date"]
    ).dt.days.fillna(0).astype(int) + 1

    aggregated["transactions_per_month"] = _safe_ratio(
        aggregated["total_transactions"] * 30,
        aggregated["days_as_customer"],
    )
    aggregated["avg_spend_per_transaction"] = _safe_ratio(
        aggregated["total_spent"],
        aggregated["total_transactions"],
    )
    aggregated["lifetime_value_per_month"] = _safe_ratio(
        aggregated["total_spent"] * 30,
        aggregated["days_as_customer"],
    )

    aggregated["engagement_tier"] = pd.cut(
        aggregated["transactions_per_month"],
        bins=[-1, 2, 10, float("inf")],
        labels=["low", "medium", "high"],
    )
    aggregated["spend_tier"] = _safe_qcut(
        aggregated["total_spent"],
        ["tier_1", "tier_2", "tier_3", "tier_4"],
        "tier_1",
    )

    aggregated["recency_days"] = (aggregated["last_transaction_date"].max() - aggregated["last_transaction_date"]).dt.days

    aggregated["recency_score"] = _safe_qcut(
        aggregated["recency_days"],
        ["5", "4", "3", "2", "1"],
        "3",
    ).astype(int)
    aggregated["frequency_score"] = _safe_qcut(
        aggregated["total_transactions"],
        ["1", "2", "3", "4", "5"],
        "3",
    ).astype(int)
    aggregated["monetary_score"] = _safe_qcut(
        aggregated["total_spent"],
        ["1", "2", "3", "4", "5"],
        "3",
    ).astype(int)
    aggregated["rfm_score"] = (
        aggregated["recency_score"]
        + aggregated["frequency_score"]
        + aggregated["monetary_score"]
    )

    feature_columns = [
        "customer_id",
        "total_transactions",
        "total_spent",
        "days_as_customer",
        "transactions_per_month",
        "avg_spend_per_transaction",
        "lifetime_value_per_month",
        "engagement_tier",
        "spend_tier",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
    ]

    summary = {
        "status": "created",
        "customer_count": int(len(aggregated)),
        "feature_columns": feature_columns[1:],
    }

    return aggregated[feature_columns], summary


def merge_feature_table(df, feature_table):
    """
    Merge engineered customer features back into the transaction table.
    """
    if feature_table.empty or "customer_id" not in df.columns:
        return df

    return df.merge(feature_table, on="customer_id", how="left", suffixes=("", "_feature"))


def save_feature_table(feature_table, report_path="output/derived_customer_features.csv"):
    """
    Persist the engineered feature table.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    feature_table.to_csv(report_path, index=False)


def save_feature_report(report, report_path="output/feature_engineering_report.json"):
    """
    Save the feature engineering summary report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)