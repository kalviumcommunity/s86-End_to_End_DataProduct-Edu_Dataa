import json
import os
from datetime import datetime

import chardet


def validate_file_exists(filepath):
    """Check if the file exists and is not empty."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    if os.path.getsize(filepath) == 0:
        return False, "File is empty."

    return True, "File exists."


def validate_file_format(filepath):
    """Check if the file format is supported."""
    allowed = ["csv", "json", "xlsx"]

    extension = filepath.split(".")[-1].lower()

    if extension not in allowed:
        return False, f"Unsupported format: {extension}"

    return True, f"Valid format: {extension}"


def validate_schema(df, required_columns):
    """Check whether required columns exist."""

    missing = set(required_columns) - set(df.columns)

    if missing:
        return False, f"Missing columns: {missing}"

    return True, "Schema is valid."


def detect_encoding(filepath):
    """Detect file encoding."""

    with open(filepath, "rb") as file:
        result = chardet.detect(file.read(10000))

    return result


def dataset_statistics(filepath, df):
    """Capture dataset statistics."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "file_size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
    }


def build_validation_report(filepath, validation_results, statistics):
    """Build a JSON-ready validation report payload."""

    return {
        "timestamp": datetime.now().isoformat(),
        "file": filepath,
        "checks": validation_results,
        "statistics": statistics,
    }


def write_validation_report(report, report_path):
    """Write the validation report to disk."""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)