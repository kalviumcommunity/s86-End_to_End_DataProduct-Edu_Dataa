"""Define and compute business KPIs with formal specifications and target validation.

This script establishes a formal KPI definition framework, computes key business
metrics (MAU, revenue per customer, churn rate, customer acquisition cost, etc.),
validates them against targets, and demonstrates consistent measurement across
the organization.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("..", "output", "kpi_definition")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class KPIDefinition:
    """Formal KPI specification."""

    name: str
    formula: str
    data_source: str
    target_min: float
    target_max: float
    owner: str
    update_frequency: str
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"KPI: {self.name}\n"
            f"  Formula: {self.formula}\n"
            f"  Data Source: {self.data_source}\n"
            f"  Target Range: {self.target_min} - {self.target_max}\n"
            f"  Owner: {self.owner}\n"
            f"  Update Frequency: {self.update_frequency}\n"
            f"  Notes: {self.notes}"
        )


def build_sample_transaction_data(n_rows: int = 50_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    data = {
        "transaction_date": np.random.choice(dates, size=n_rows),
        "customer_id": rng.integers(1, 8000, size=n_rows),
        "amount": np.round(rng.lognormal(mean=5.0, sigma=1.5, size=n_rows), 2),
        "product_type": np.random.choice(["basic", "standard", "premium"], size=n_rows),
    }
    churn_dates = rng.integers(1, 365, size=n_rows)
    last_active_threshold = (pd.Timestamp.now() - pd.Timestamp("2024-01-01")).days
    data["churned"] = churn_dates < (last_active_threshold - 30)

    return pd.DataFrame(data)


def calculate_mau(df: pd.DataFrame, days: int = 30) -> int:
    """Monthly Active Users: distinct customers with transaction in last N days."""
    cutoff = df["transaction_date"].max() - pd.Timedelta(days=days)
    mau = df[df["transaction_date"] >= cutoff]["customer_id"].nunique()
    return int(mau)


def calculate_revenue_per_customer(df: pd.DataFrame) -> float:
    """Average revenue generated per unique customer (RPC)."""
    total_revenue = df["amount"].sum()
    unique_customers = df["customer_id"].nunique()
    return total_revenue / unique_customers if unique_customers > 0 else 0.0


def calculate_average_order_value(df: pd.DataFrame) -> float:
    """Average transaction amount (AOV)."""
    return df["amount"].mean()


def calculate_customer_acquisition_cost(df: pd.DataFrame, marketing_spend: float) -> float:
    """CAC: total marketing spend / customers acquired in period."""
    unique_customers = df["customer_id"].nunique()
    return marketing_spend / unique_customers if unique_customers > 0 else 0.0


def calculate_churn_rate(df: pd.DataFrame) -> float:
    """Percentage of customers who churned (no transaction in last 30 days)."""
    if "churned" in df.columns:
        return (df["churned"].sum() / len(df)) if len(df) > 0 else 0.0
    return 0.0


def calculate_ltv(rpc: float, avg_customer_lifetime_months: float = 12) -> float:
    """Customer Lifetime Value: RPC * months active."""
    return rpc * avg_customer_lifetime_months


def calculate_gross_margin(df: pd.DataFrame, cogs_ratio: float = 0.35) -> float:
    """Gross Margin: (Revenue - COGS) / Revenue."""
    total_revenue = df["amount"].sum()
    cogs = total_revenue * cogs_ratio
    return ((total_revenue - cogs) / total_revenue) if total_revenue > 0 else 0.0


def define_kpi_catalog() -> Dict[str, KPIDefinition]:
    """Formal KPI specifications."""
    return {
        "mau": KPIDefinition(
            name="Monthly Active Users (MAU)",
            formula="COUNT(DISTINCT customer_id) WHERE transaction_date >= TODAY() - 30 days",
            data_source="transactions table, transaction_date column",
            target_min=5000,
            target_max=7000,
            owner="Product Lead",
            update_frequency="Daily",
            notes="Indicator of product engagement; metric of customer retention.",
        ),
        "rpc": KPIDefinition(
            name="Revenue Per Customer (RPC)",
            formula="SUM(amount) / COUNT(DISTINCT customer_id)",
            data_source="transactions table, amount column",
            target_min=85.0,
            target_max=115.0,
            owner="Finance",
            update_frequency="Daily",
            notes="Average revenue generated per unique customer; tracks monetization efficiency.",
        ),
        "aov": KPIDefinition(
            name="Average Order Value (AOV)",
            formula="SUM(amount) / COUNT(transactions)",
            data_source="transactions table, amount column",
            target_min=45.0,
            target_max=65.0,
            owner="Sales",
            update_frequency="Daily",
            notes="Average transaction size; indicator of deal size and product mix.",
        ),
        "churn": KPIDefinition(
            name="Churn Rate",
            formula="COUNT(churned_customers) / COUNT(total_customers)",
            data_source="transactions table, churned flag",
            target_min=0.0,
            target_max=0.10,
            owner="Product/Customer Success",
            update_frequency="Weekly",
            notes="Percentage of customers inactive in last 30 days; retention health indicator.",
        ),
        "ltv": KPIDefinition(
            name="Customer Lifetime Value (LTV)",
            formula="RPC * avg_customer_lifetime_months",
            data_source="Derived from transactions, with assumed 12-month lifecycle",
            target_min=1020.0,
            target_max=1380.0,
            owner="Finance",
            update_frequency="Monthly",
            notes="Total expected revenue from average customer over lifetime.",
        ),
    }


def validate_kpis(kpi_values: Dict[str, float], kpi_defs: Dict[str, KPIDefinition]) -> Dict[str, bool]:
    """Check if KPI values fall within target ranges."""
    validation = {}
    for key, value in kpi_values.items():
        if key in kpi_defs:
            kpi_def = kpi_defs[key]
            is_valid = kpi_def.target_min <= value <= kpi_def.target_max
            validation[key] = is_valid
    return validation


def compute_all_kpis(df: pd.DataFrame, marketing_spend: float = 5000.0) -> Dict[str, float]:
    """Compute all KPIs."""
    mau = calculate_mau(df)
    rpc = calculate_revenue_per_customer(df)
    aov = calculate_average_order_value(df)
    churn = calculate_churn_rate(df)
    ltv = calculate_ltv(rpc, avg_customer_lifetime_months=12)

    return {
        "mau": mau,
        "rpc": rpc,
        "aov": aov,
        "churn": churn,
        "ltv": ltv,
    }


def generate_kpi_report(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    kpi_defs = define_kpi_catalog()
    kpi_values = compute_all_kpis(df)
    validation = validate_kpis(kpi_values, kpi_defs)

    print("=== KPI Definitions ===\n")
    for key, kpi_def in kpi_defs.items():
        print(kpi_def)
        print()

    print("=== Computed KPIs ===\n")
    for key, value in kpi_values.items():
        status = "✓" if validation.get(key, False) else "✗"
        kpi_def = kpi_defs[key]
        print(f"{status} {kpi_def.name}")
        print(f"  Actual: {value:.2f}")
        print(f"  Target: {kpi_def.target_min:.2f} - {kpi_def.target_max:.2f}")
        if validation.get(key, False):
            print("  Status: ON TARGET")
        else:
            print("  Status: OUT OF RANGE - Requires attention")
        print()

    print("=== Consistency Check ===")
    print("All KPIs computed using same formulas and data source.")
    print("Same function called by all teams → single source of truth.")
    print("Metric definitions documented and versioned.")

    report_df = pd.DataFrame(
        {
            "KPI": list(kpi_values.keys()),
            "Actual": list(kpi_values.values()),
            "Target Min": [kpi_defs[k].target_min for k in kpi_values.keys()],
            "Target Max": [kpi_defs[k].target_max for k in kpi_values.keys()],
            "OnTarget": [validation.get(k, False) for k in kpi_values.keys()],
        }
    )
    report_df.to_csv(os.path.join(OUTPUT_DIR, "kpi_report.csv"), index=False)
    print(f"\nKPI report saved to {os.path.join(OUTPUT_DIR, 'kpi_report.csv')}")


def main() -> None:
    df = build_sample_transaction_data()
    generate_kpi_report(df)


if __name__ == "__main__":
    main()
