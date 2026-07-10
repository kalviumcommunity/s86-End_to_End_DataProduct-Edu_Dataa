from ingest import document_ingestion, ingest_data
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
REQUIRED_COLUMNS = []


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