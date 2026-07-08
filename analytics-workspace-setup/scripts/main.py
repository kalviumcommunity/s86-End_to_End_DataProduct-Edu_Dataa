from ingest import ingest_data
from process import process_data
from output import output_results


INPUT_FILE = "data/raw/sample.csv"
OUTPUT_FILE = "data/processed/cleaned_sample.csv"


def main():
    """
    Run the ingest-process-output workflow.

    Returns:
        None.
    """
    df = ingest_data(INPUT_FILE)

    processed_df = process_data(df)

    output_results(processed_df, OUTPUT_FILE)

    print("Workflow completed.")


if __name__ == "__main__":
    main()