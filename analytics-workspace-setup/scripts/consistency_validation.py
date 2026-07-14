import json
import os

import pandas as pd


def _ensure_datetime(series):
    """
    Convert a series to datetime when needed.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    return pd.to_datetime(series, errors="coerce")


def validate_not_null(df, column):
    """
    Validate that a column has no missing values.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)

    return df[column].notna()


def validate_numeric_range(df, column, minimum=None, maximum=None):
    """
    Validate that numeric values stay within an expected range.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)

    numeric_series = pd.to_numeric(df[column], errors="coerce")
    valid = numeric_series.notna()

    if minimum is not None:
        valid = valid & (numeric_series >= minimum)

    if maximum is not None:
        valid = valid & (numeric_series <= maximum)

    return valid


def validate_datetime_range(df, column, minimum=None, maximum=None):
    """
    Validate that datetime values stay within the requested range.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)

    datetime_series = _ensure_datetime(df[column])
    valid = datetime_series.notna()

    if minimum is not None:
        valid = valid & (datetime_series >= minimum)

    if maximum is not None:
        valid = valid & (datetime_series <= maximum)

    return valid


def validate_regex_format(df, column, pattern):
    """
    Validate text values against a regex pattern.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)

    return df[column].astype(str).str.match(pattern, na=False)


def validate_date_order(df, start_column, end_column):
    """
    Validate that an end date is not earlier than the start date.
    """
    if start_column not in df.columns or end_column not in df.columns:
        return pd.Series(True, index=df.index)

    start_dates = _ensure_datetime(df[start_column])
    end_dates = _ensure_datetime(df[end_column])

    return start_dates.notna() & end_dates.notna() & (end_dates >= start_dates)


def run_validation_rules(df):
    """
    Run data consistency checks and isolate failing rows.
    """
    rule_results = {}

    rule_results["customer_id_required"] = validate_not_null(df, "customer_id")

    if "amount" in df.columns:
        rule_results["amount_non_negative"] = validate_numeric_range(
            df,
            "amount",
            minimum=0,
        )

    if "transaction_date" in df.columns:
        min_date = pd.Timestamp("1920-01-01")
        max_date = pd.Timestamp.now()
        rule_results["transaction_date_range"] = validate_datetime_range(
            df,
            "transaction_date",
            minimum=min_date,
            maximum=max_date,
        )

    if "email" in df.columns:
        rule_results["email_format"] = validate_regex_format(
            df,
            "email",
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        )

    if "start_date" in df.columns and "end_date" in df.columns:
        rule_results["date_order"] = validate_date_order(df, "start_date", "end_date")

    if not rule_results:
        empty_report = {
            "total_records": int(len(df)),
            "passed_records": int(len(df)),
            "failed_records": 0,
            "rules": [],
        }
        return df.copy(), df.iloc[0:0].copy(), empty_report

    rule_frame = pd.DataFrame(rule_results, index=df.index)
    passes_all_rules = rule_frame.all(axis=1)

    cleaned_df = df.loc[passes_all_rules].copy()
    failed_df = df.loc[~passes_all_rules].copy()
    failure_counts = (~rule_frame).sum().astype(int)

    report = {
        "total_records": int(len(df)),
        "passed_records": int(passes_all_rules.sum()),
        "failed_records": int((~passes_all_rules).sum()),
        "rules": [
            {
                "rule": rule_name,
                "passed_records": int(rule_series.sum()),
                "failed_records": int((~rule_series).sum()),
            }
            for rule_name, rule_series in rule_results.items()
        ],
        "failure_summary": failure_counts.to_dict(),
    }

    return cleaned_df, failed_df, report


def save_validation_report(report, report_path="output/validation_report.json"):
    """
    Save the validation report to disk.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)


def save_validation_failures(df, report_path="output/validation_failures.csv"):
    """
    Persist failing validation rows for audit purposes.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)