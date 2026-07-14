import json
import os

import pandas as pd


def validate_join(left_df, right_df, key_columns, join_type="left"):
    """
    Merge two dataframes and capture join validation metrics.
    """
    merged_df = pd.merge(
        left_df,
        right_df,
        on=key_columns,
        how=join_type,
        indicator=True,
        suffixes=("_left", ".right"),
    )

    unmatched_left = merged_df.loc[merged_df["_merge"] == "left_only"].copy()
    unmatched_right = merged_df.loc[merged_df["_merge"] == "right_only"].copy()

    matched_rows = int((merged_df["_merge"] == "both").sum())
    report = {
        "join_type": join_type,
        "key_columns": key_columns,
        "left_rows": int(len(left_df)),
        "right_rows": int(len(right_df)),
        "merged_rows": int(len(merged_df)),
        "row_change_from_left": int(len(merged_df) - len(left_df)),
        "matched_rows": matched_rows,
        "unmatched_left_rows": int(len(unmatched_left)),
        "unmatched_right_rows": int(len(unmatched_right)),
        "expected_behavior": (
            "left join keeps all left rows and may expand when the right side has repeated keys."
            if join_type == "left"
            else "join behavior depends on the selected join type and key cardinality."
        ),
    }

    merged_df = merged_df.drop(columns=["_merge"])

    return merged_df, report, unmatched_left, unmatched_right


def save_join_validation_report(report, report_path="output/join_validation_report.json"):
    """
    Save join validation metadata to disk.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)


def save_unmatched_records(unmatched_left, unmatched_right, left_path="output/unmatched_left_records.csv", right_path="output/unmatched_right_records.csv"):
    """
    Persist unmatched rows for investigation.
    """
    os.makedirs(os.path.dirname(left_path), exist_ok=True)
    unmatched_left.to_csv(left_path, index=False)
    unmatched_right.to_csv(right_path, index=False)