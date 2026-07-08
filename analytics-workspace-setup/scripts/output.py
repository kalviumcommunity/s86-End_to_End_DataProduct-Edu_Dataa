def output_results(df, filepath):
    """
    Save processed data to a CSV file.

    Args:
        df: A pandas DataFrame to save.
        filepath: Destination path for the output CSV file.

    Returns:
        None.
    """
    df.to_csv(filepath, index=False)