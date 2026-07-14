import json
import os


def analyze_missing(df):
    """
    Analyze missing values in each column.
    """
    report = {}

    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = round((null_count / len(df)) * 100, 2)

        report[col] = {
            "null_count": int(null_count),
            "null_percentage": null_percentage,
        }

    return report


def fill_numeric_median(df, column):
    """
    Fill missing numeric values using the median.
    """
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())

    return df


def fill_categorical_mode(df, column):
    """
    Fill missing categorical values using the mode.
    """
    if column in df.columns:
        mode_value = df[column].mode()

        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])

    return df


def forward_fill(df, column):
    """
    Forward fill missing values.
    """
    if column in df.columns:
        df[column] = df[column].ffill()

    return df


def drop_missing_ids(df, id_column):
    """
    Remove rows where the primary identifier is missing.
    """
    if id_column in df.columns:
        return df.dropna(subset=[id_column])

    return df


def generate_imputation_report(before, after):
    """
    Compare missing values before and after imputation.
    """
    report = {}

    for column in before.keys():
        report[column] = {
            "before": before[column]["null_count"],
            "after": after[column]["null_count"],
        }

    return report


def save_imputation_report(report, report_path="output/imputation_report.json"):
    """
    Save the imputation report to disk.
    """

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)