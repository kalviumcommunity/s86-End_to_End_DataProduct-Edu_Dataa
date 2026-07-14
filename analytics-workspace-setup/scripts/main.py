import os

from .ingest import document_ingestion, ingest_data
from .enforce_types import (
    compare_dtypes,
    convert_boolean,
    convert_currency,
    convert_dates,
    save_dtype_report,
)
from .missing_values import (
    analyze_missing,
    drop_missing_ids,
    fill_categorical_mode,
    fill_numeric_median,
    forward_fill,
    generate_imputation_report,
    save_imputation_report,
)
from .duplicates import (
    analyze_duplicates,
    compare_before_after,
    deduplicate_records,
    save_deduplication_report,
    save_duplicate_audit,
)
from .datetime_transform import (
    add_datetime_features,
    convert_to_datetime,
    resample_time_series,
    save_datetime_report,
)
from .outlier_detection import apply_outlier_strategy, save_outlier_report
from .consistency_validation import (
    run_validation_rules,
    save_validation_failures,
    save_validation_report,
)
from .join_validation import (
    save_join_validation_report,
    save_unmatched_records,
    validate_join,
)
from .feature_engineering import (
    build_customer_feature_table,
    merge_feature_table,
    save_feature_report,
    save_feature_table,
)
from .output import output_results
from .profile import (
    identify_quality_issues,
    profile_nulls_and_duplicates,
    profile_numerical,
    save_profile,
)
from .process import process_data
from .validate import (
    build_validation_report,
    dataset_statistics,
    detect_encoding,
    validate_file_exists,
    validate_file_format,
    validate_schema,
    write_validation_report,
)


INPUT_FILE = "data/raw/sample.csv"
OUTPUT_FILE = "data/processed/cleaned_sample.csv"
REPORT_FILE = "output/intake_report.json"
TYPE_REPORT_FILE = "output/type_conversion_report.json"
IMPUTATION_REPORT_FILE = "output/imputation_report.json"
DEDUPLICATION_REPORT_FILE = "output/deduplication_report.json"
REMOVED_DUPLICATES_AUDIT_FILE = "output/removed_duplicates_audit.csv"
DATETIME_REPORT_FILE = "output/datetime_transformation_report.json"
WEEKLY_SUMMARY_FILE = "output/weekly_transaction_summary.csv"
OUTLIER_REPORT_FILE = "output/outlier_detection_report.json"
VALIDATION_REPORT_FILE = "output/validation_report.json"
VALIDATION_FAILURES_FILE = "output/validation_failures.csv"
JOIN_VALIDATION_REPORT_FILE = "output/join_validation_report.json"
UNMATCHED_LEFT_FILE = "output/unmatched_left_records.csv"
UNMATCHED_RIGHT_FILE = "output/unmatched_right_records.csv"
JOIN_SOURCE_FILE = "data/raw/reference.csv"
JOIN_KEY_COLUMNS = ["customer_id"]
JOIN_TYPE = "left"
FEATURE_TABLE_FILE = "output/derived_customer_features.csv"
FEATURE_REPORT_FILE = "output/feature_engineering_report.json"
REQUIRED_COLUMNS = []
DATE_COLUMNS = ["transaction_date"]
CURRENCY_COLUMNS = ["amount"]
BOOLEAN_COLUMNS = ["is_active"]
NUMERIC_IMPUTE_COLUMNS = ["amount"]
CATEGORICAL_IMPUTE_COLUMNS = ["segment"]
TIME_SERIES_COLUMNS = ["transaction_date"]
ID_COLUMNS = ["customer_id"]
DEDUPLICATION_KEY_COLUMNS = ["customer_id", "transaction_date"]
DEDUPLICATION_KEEP = "first"


def main():
    """
    Run the ingest-process-output workflow.

    Returns:
        None.
    """
    exists_ok, exists_message = validate_file_exists(INPUT_FILE)
    if not exists_ok:
        raise ValueError(exists_message)

    format_ok, format_message = validate_file_format(INPUT_FILE)
    if not format_ok:
        raise ValueError(format_message)

    df = ingest_data(INPUT_FILE)
    document_ingestion(df, INPUT_FILE)

    schema_ok, schema_message = validate_schema(df, REQUIRED_COLUMNS)
    if not schema_ok:
        raise ValueError(schema_message)

    profile = profile_nulls_and_duplicates(df)
    stats = profile_numerical(df)
    issues = identify_quality_issues(df)
    save_profile(profile, stats, issues)

    before_type_df = df.copy()

    for column in DATE_COLUMNS:
        df = convert_dates(df, column)
        df = convert_to_datetime(df, column)
        df = add_datetime_features(df, column)

    for column in CURRENCY_COLUMNS:
        df = convert_currency(df, column)

    for column in BOOLEAN_COLUMNS:
        df = convert_boolean(df, column)

    type_report = compare_dtypes(before_type_df, df)
    save_dtype_report(type_report)

    before_missing_report = analyze_missing(df)

    for column in NUMERIC_IMPUTE_COLUMNS:
        df = fill_numeric_median(df, column)

    for column in CATEGORICAL_IMPUTE_COLUMNS:
        df = fill_categorical_mode(df, column)

    for column in TIME_SERIES_COLUMNS:
        df = forward_fill(df, column)

    for column in ID_COLUMNS:
        df = drop_missing_ids(df, column)

    after_missing_report = analyze_missing(df)
    imputation_report = generate_imputation_report(
        before_missing_report,
        after_missing_report,
    )
    save_imputation_report(imputation_report, IMPUTATION_REPORT_FILE)

    before_deduplication_df = df.copy()
    duplicate_analysis = analyze_duplicates(df, DEDUPLICATION_KEY_COLUMNS)
    df, removed_duplicates = deduplicate_records(
        df,
        key_columns=DEDUPLICATION_KEY_COLUMNS,
        keep=DEDUPLICATION_KEEP,
    )
    deduplication_comparison = compare_before_after(
        before_deduplication_df,
        df,
    )
    save_duplicate_audit(removed_duplicates, REMOVED_DUPLICATES_AUDIT_FILE)
    save_deduplication_report(
        {
            "duplicate_analysis": duplicate_analysis,
            "deduplication_comparison": deduplication_comparison,
            "key_columns": DEDUPLICATION_KEY_COLUMNS,
            "keep_strategy": DEDUPLICATION_KEEP,
        },
        DEDUPLICATION_REPORT_FILE,
    )

    df, outlier_report = apply_outlier_strategy(
        df,
        "amount",
        strategy="cap",
        detector="iqr",
    )
    save_outlier_report(
        {
            "cleaning_rules": [
                {
                    "column": "amount",
                    "detector": "iqr",
                    "strategy": "cap",
                    "reason": "Preserve rows while limiting extreme values that can distort summary statistics.",
                }
            ],
            "steps": [outlier_report],
        },
        OUTLIER_REPORT_FILE,
    )

    df, validation_failures, validation_report = run_validation_rules(df)
    save_validation_failures(validation_failures, VALIDATION_FAILURES_FILE)
    save_validation_report(validation_report, VALIDATION_REPORT_FILE)

    if os.path.exists(JOIN_SOURCE_FILE):
        secondary_df = ingest_data(JOIN_SOURCE_FILE)
        merged_df, join_report, unmatched_left, unmatched_right = validate_join(
            df,
            secondary_df,
            JOIN_KEY_COLUMNS,
            join_type=JOIN_TYPE,
        )
        save_unmatched_records(
            unmatched_left,
            unmatched_right,
            UNMATCHED_LEFT_FILE,
            UNMATCHED_RIGHT_FILE,
        )
        save_join_validation_report(join_report, JOIN_VALIDATION_REPORT_FILE)
        df = merged_df
    else:
        save_join_validation_report(
            {
                "join_type": JOIN_TYPE,
                "key_columns": JOIN_KEY_COLUMNS,
                "source_file": JOIN_SOURCE_FILE,
                "status": "skipped",
                "reason": "Secondary source file not present in the workspace.",
            },
            JOIN_VALIDATION_REPORT_FILE,
        )

    feature_table, feature_report = build_customer_feature_table(df)
    save_feature_table(feature_table, FEATURE_TABLE_FILE)
    save_feature_report(feature_report, FEATURE_REPORT_FILE)
    df = merge_feature_table(df, feature_table)

    weekly_summary = resample_time_series(
        df,
        "transaction_date",
        "amount",
        frequency="W",
    )
    weekly_summary.to_csv(WEEKLY_SUMMARY_FILE, index=True)
    save_datetime_report(
        {
            "datetime_columns": DATE_COLUMNS,
            "feature_columns": [
                "transaction_date_day_of_week",
                "transaction_date_day_of_week_num",
                "transaction_date_hour",
                "transaction_date_week_num",
                "transaction_date_month",
                "transaction_date_quarter",
                "transaction_date_days_since_event",
            ],
            "weekly_summary_rows": int(len(weekly_summary)),
            "weekly_summary_file": WEEKLY_SUMMARY_FILE,
        },
        DATETIME_REPORT_FILE,
    )

    encoding_result = detect_encoding(INPUT_FILE)
    statistics = dataset_statistics(INPUT_FILE, df)

    report = build_validation_report(
        INPUT_FILE,
        {
            "file_exists": exists_message,
            "format": format_message,
            "schema": schema_message,
            "encoding": encoding_result,
        },
        statistics,
    )
    write_validation_report(report, REPORT_FILE)

    processed_df = process_data(df)

    output_results(processed_df, OUTPUT_FILE)

    print("Workflow completed.")


if __name__ == "__main__":
    main()