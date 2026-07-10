from ingest import document_ingestion, ingest_data
from enforce_types import (
    compare_dtypes,
    convert_boolean,
    convert_currency,
    convert_dates,
    save_dtype_report,
)
from missing_values import (
    analyze_missing,
    drop_missing_ids,
    fill_categorical_mode,
    fill_numeric_median,
    forward_fill,
    generate_imputation_report,
    save_imputation_report,
)
from output import output_results
from profile import (
    identify_quality_issues,
    profile_nulls_and_duplicates,
    profile_numerical,
    save_profile,
)
from process import process_data
from validate import (
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
REQUIRED_COLUMNS = []
DATE_COLUMNS = ["transaction_date"]
CURRENCY_COLUMNS = ["amount"]
BOOLEAN_COLUMNS = ["is_active"]
NUMERIC_IMPUTE_COLUMNS = ["amount"]
CATEGORICAL_IMPUTE_COLUMNS = ["segment"]
TIME_SERIES_COLUMNS = ["transaction_date"]
ID_COLUMNS = ["customer_id"]


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