import pandas as pd


def ingest_csv(filepath, delimiter=",", encoding="utf-8"):
    """
    Load a CSV file with explicit delimiter and encoding.
    """

    return pd.read_csv(
        filepath,
        delimiter=delimiter,
        encoding=encoding,
    )


def ingest_csv_with_fallback(filepath):
    """
    Load a CSV file by trying common encodings.
    """

    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode file.")


def ingest_json(filepath, nested=False):
    """
    Load a JSON file and optionally flatten nested records.
    """

    df = pd.read_json(filepath)

    if nested:
        df = pd.json_normalize(df.to_dict(orient="records"))

    return df


def ingest_excel(filepath, sheet_name=0):
    """
    Load an Excel worksheet into a DataFrame.
    """

    return pd.read_excel(
        filepath,
        sheet_name=sheet_name,
    )


def document_ingestion(df, source):
    """
    Print a lightweight ingestion audit summary.
    """

    print("=" * 40)
    print(f"Source : {source}")
    print(f"Rows   : {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nData Types")
    print(df.dtypes)

    print("\nPreview")
    print(df.head(3))


def ingest_data(filepath):
    """
    Load tabular data by dispatching to the appropriate loader.
    """

    extension = filepath.split(".")[-1].lower()

    if extension == "csv":
        return ingest_csv_with_fallback(filepath)

    if extension == "json":
        return ingest_json(filepath)

    if extension in {"xlsx", "xls"}:
        return ingest_excel(filepath)

    raise ValueError(f"Unsupported file type: {extension}")