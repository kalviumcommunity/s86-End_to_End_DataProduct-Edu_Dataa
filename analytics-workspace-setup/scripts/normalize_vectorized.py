"""Demonstrate NumPy-based normalization versus Python loop normalization.

This script creates a sample DataFrame with a large numeric column, then
compares a simple Python loop normalization to a vectorized NumPy approach.
The final normalized values are added back to the DataFrame as a new column.
"""

import time

import numpy as np
import pandas as pd


def normalize_with_loop(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column].to_list()
    min_value = min(values)
    max_value = max(values)
    normalized = []
    for x in values:
        normalized.append((x - min_value) / (max_value - min_value))
    return pd.Series(normalized, index=df.index)


def normalize_with_numpy(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].values.astype(float)
    return (values - values.min()) / (values.max() - values.min())


def benchmark_normalization(n_rows: int = 100_000) -> None:
    df = pd.DataFrame({"revenue": np.random.default_rng(42).uniform(100.0, 10_000.0, size=n_rows)})

    start = time.time()
    loop_normalized = normalize_with_loop(df, "revenue")
    loop_time = time.time() - start

    start = time.time()
    vectorized_normalized = normalize_with_numpy(df, "revenue")
    vectorized_time = time.time() - start

    df["revenue_normalized_loop"] = loop_normalized
    df["revenue_normalized_vectorized"] = vectorized_normalized

    print(f"Rows: {n_rows:,}")
    print(f"Loop normalization time: {loop_time:.3f}s")
    print(f"Vectorized NumPy time: {vectorized_time:.3f}s")
    print(f"Speedup: {loop_time / vectorized_time:.0f}x")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    benchmark_normalization(n_rows=100_000)
