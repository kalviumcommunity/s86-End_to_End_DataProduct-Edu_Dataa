import json
import os


def analyze_duplicates(df, key_columns=None):
    """
    Analyze exact and key-based duplicate records.
    """
    report = {
        "exact_duplicates": int(df.duplicated().sum()),
    }

    if key_columns:
        available_keys = [column for column in key_columns if column in df.columns]

        if available_keys:
            report["key_columns"] = available_keys
            report["key_duplicates"] = int(
                df.duplicated(subset=available_keys).sum()
            )

    return report


def deduplicate_records(df, key_columns=None, keep="first"):
    """
    Remove duplicate rows using the configured strategy.
    """
    if key_columns:
        available_keys = [column for column in key_columns if column in df.columns]

        if available_keys:
            deduplicated_df = df.drop_duplicates(subset=available_keys, keep=keep)
        else:
            deduplicated_df = df.drop_duplicates(keep=keep)
    else:
        deduplicated_df = df.drop_duplicates(keep=keep)

    removed_rows = df.loc[~df.index.isin(deduplicated_df.index)].copy()
    return deduplicated_df, removed_rows


def compare_before_after(before_df, after_df):
    """
    Compare row counts before and after deduplication.
    """
    rows_before = len(before_df)
    rows_after = len(after_df)
    rows_removed = rows_before - rows_after

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "removal_pct": round((rows_removed / rows_before) * 100, 2) if rows_before else 0,
    }


def save_duplicate_audit(removed_df, report_path="output/removed_duplicates_audit.csv"):
    """
    Save removed duplicate rows for audit purposes.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    removed_df.to_csv(report_path, index=False)


def save_deduplication_report(report, report_path="output/deduplication_report.json"):
    """
    Save the deduplication summary report to disk.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)