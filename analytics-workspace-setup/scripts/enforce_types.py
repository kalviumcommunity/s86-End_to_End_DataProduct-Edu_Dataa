import json
import os

import pandas as pd


def convert_dates(df, column, date_format="%Y-%m-%d"):
    """
    Convert a string column to datetime.
    """
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            format=date_format,
            errors="coerce",
        )

    return df


def convert_currency(df, column):
    """
    Convert currency strings to numeric values.
    """
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def convert_boolean(df, column):
    """
    Convert common boolean-like values to booleans.
    """
    if column in df.columns:
        mapping = {
            1: True,
            0: False,
            "yes": True,
            "no": False,
            "Yes": True,
            "No": False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
            True: True,
            False: False,
        }

        normalized_values = df[column].astype(str).str.strip().str.lower()
        df[column] = normalized_values.map(mapping)

    return df


def compare_dtypes(before_df, after_df):
    """
    Compare column data types before and after enforcement.
    """
    report = {}

    for column in before_df.columns:

        report[column] = {
            "before": str(before_df[column].dtype),
            "after": str(after_df[column].dtype),
        }

    return report


def save_dtype_report(report):
    """
    Save the type conversion report to disk.
    """

    report_path = "output/type_conversion_report.json"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)