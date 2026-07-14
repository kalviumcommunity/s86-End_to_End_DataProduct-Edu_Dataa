import json
import os

import pandas as pd


def convert_to_datetime(df, column, date_format=None):
    """
    Convert a string column to pandas datetime.
    """
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            format=date_format,
            errors="coerce",
        )

    return df


def add_datetime_features(df, column, prefix=None):
    """
    Extract common datetime features from a timestamp column.
    """
    if column not in df.columns:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df = convert_to_datetime(df, column)

    feature_prefix = prefix or column

    df[f"{feature_prefix}_day_of_week"] = df[column].dt.day_name()
    df[f"{feature_prefix}_day_of_week_num"] = df[column].dt.dayofweek
    df[f"{feature_prefix}_hour"] = df[column].dt.hour
    df[f"{feature_prefix}_week_num"] = df[column].dt.isocalendar().week.astype("Int64")
    df[f"{feature_prefix}_month"] = df[column].dt.month
    df[f"{feature_prefix}_quarter"] = df[column].dt.quarter

    today = pd.Timestamp.now(tz=df[column].dt.tz)
    df[f"{feature_prefix}_days_since_event"] = (today - df[column]).dt.days

    return df


def resample_time_series(df, datetime_column, value_column, frequency="W", aggregation="sum"):
    """
    Aggregate a value column by a datetime frequency.
    """
    if datetime_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame(columns=[value_column])

    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df = convert_to_datetime(df, datetime_column)

    time_indexed_df = df.dropna(subset=[datetime_column]).set_index(datetime_column)

    if time_indexed_df.empty:
        return pd.DataFrame(columns=[value_column])

    resampler = getattr(time_indexed_df[value_column].resample(frequency), aggregation, None)
    if resampler is None:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return resampler().to_frame(name=value_column)


def save_datetime_report(report, report_path="output/datetime_transformation_report.json"):
    """
    Persist a datetime transformation summary to disk.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)