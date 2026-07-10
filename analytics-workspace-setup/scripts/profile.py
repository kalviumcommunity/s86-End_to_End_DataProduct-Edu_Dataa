import json
import os


def profile_nulls_and_duplicates(df):
    """
    Profile null values and duplicate records.
    """
    profile = {}

    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = round((null_count / len(df)) * 100, 2)

        profile[col] = {
            "nulls": int(null_count),
            "null_percentage": null_pct,
        }

    profile["exact_duplicates"] = int(df.duplicated().sum())

    return profile


def profile_numerical(df):
    """
    Generate summary statistics for numeric columns.
    """
    stats = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:

        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": round(df[col].mean(), 2),
            "median": df[col].median(),
        }

    return stats


def identify_quality_issues(df, null_threshold=30, duplicate_threshold=5):
    """
    Detect columns with potential quality problems.
    """

    issues = []

    for col in df.columns:

        null_pct = (df[col].isnull().sum() / len(df)) * 100

        if null_pct > null_threshold:
            issues.append(
                {
                    "column": col,
                    "issue": "High null percentage",
                    "value": f"{null_pct:.2f}%",
                }
            )

    duplicate_pct = (df.duplicated().sum() / len(df)) * 100

    if duplicate_pct > duplicate_threshold:

        issues.append(
            {
                "issue": "High duplicate percentage",
                "value": f"{duplicate_pct:.2f}%",
            }
        )

    return issues


def save_profile(profile, stats, issues, report_path="output/profiling_report.json"):
    """
    Save the profiling report to disk.
    """

    report = {
        "nulls_and_duplicates": profile,
        "numerical_statistics": stats,
        "quality_issues": issues,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)