import pandas as pd


def ingest_data(filepath):
    """
    Read data from a CSV file.

    Args:
        filepath: Path to the CSV file to read.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(filepath)