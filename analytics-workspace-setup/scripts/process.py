def process_data(df):
    """
    Remove duplicate rows from the input DataFrame.

    Args:
        df: A pandas DataFrame to preprocess.

    Returns:
        A pandas DataFrame with duplicate rows removed.
    """
    df = df.drop_duplicates()

    return df