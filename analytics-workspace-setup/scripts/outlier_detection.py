import json
import os

import pandas as pd


def detect_zscore_outliers(series, threshold=3.0):
    """
    Detect outliers using a Z-score threshold.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    standard_deviation = numeric_series.std(ddof=0)

    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(False, index=series.index)

    z_scores = (numeric_series - numeric_series.mean()).abs() / standard_deviation
    return z_scores > threshold


def detect_iqr_outliers(series, multiplier=1.5):
    """
    Detect outliers using the IQR rule.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    q1 = numeric_series.quantile(0.25)
    q3 = numeric_series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index), None, None

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    outliers = (numeric_series < lower_bound) | (numeric_series > upper_bound)

    return outliers, lower_bound, upper_bound


def cap_outliers(series, lower_bound, upper_bound):
    """
    Cap values within the provided bounds.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")

    if lower_bound is None or upper_bound is None:
        return numeric_series

    return numeric_series.clip(lower=lower_bound, upper=upper_bound)


def flag_outliers(df, column, flag_column, outlier_mask):
    """
    Add a binary outlier flag column.
    """
    df[flag_column] = outlier_mask.fillna(False).astype(int)
    return df


def remove_outliers(df, outlier_mask):
    """
    Remove rows that are marked as outliers.
    """
    return df.loc[~outlier_mask.fillna(False)].copy()


def apply_outlier_strategy(df, column, strategy="cap", detector="iqr"):
    """
    Detect outliers for a numeric column and apply a chosen handling strategy.
    """
    if column not in df.columns:
        return df, {
            "column": column,
            "detector": detector,
            "strategy": strategy,
            "outlier_count": 0,
            "lower_bound": None,
            "upper_bound": None,
        }

    if detector == "zscore":
        outlier_mask = detect_zscore_outliers(df[column])
        lower_bound = None
        upper_bound = None
    else:
        outlier_mask, lower_bound, upper_bound = detect_iqr_outliers(df[column])

    outlier_count = int(outlier_mask.fillna(False).sum())
    working_df = df.copy()

    if strategy == "flag":
        working_df = flag_outliers(
            working_df,
            column,
            f"{column}_is_outlier",
            outlier_mask,
        )
    elif strategy == "remove":
        working_df = remove_outliers(working_df, outlier_mask)
    else:
        if lower_bound is not None and upper_bound is not None:
            working_df[column] = cap_outliers(working_df[column], lower_bound, upper_bound)
        working_df = flag_outliers(
            working_df,
            column,
            f"{column}_is_outlier",
            outlier_mask,
        )

    report = {
        "column": column,
        "detector": detector,
        "strategy": strategy,
        "outlier_count": outlier_count,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    return working_df, report


def save_outlier_report(report, report_path="output/outlier_detection_report.json"):
    """
    Save the outlier detection report to disk.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)