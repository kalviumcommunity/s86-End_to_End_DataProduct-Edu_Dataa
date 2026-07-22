"""Analyze user funnels to identify bottlenecks and drop-off points.

This script builds a synthetic funnel from signup through first purchase,
computes drop-off rates at each stage, identifies the biggest leak, and
visualizes the funnel with annotations. It demonstrates how to quantify
friction and prioritize optimization efforts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("..", "output", "funnel_analysis")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_funnel_dataset(n_users: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    data = {
        "user_id": np.arange(1, n_users + 1),
        "signup_completed": np.ones(n_users, dtype=int),
    }

    data["email_entered"] = (rng.random(n_users) < 0.80).astype(int)
    email_entered_users = np.where(data["email_entered"] == 1)[0]

    data["password_created"] = np.zeros(n_users, dtype=int)
    data["password_created"][email_entered_users[rng.random(len(email_entered_users)) < 0.75]] = 1
    password_created_users = np.where(data["password_created"] == 1)[0]

    data["email_verified"] = np.zeros(n_users, dtype=int)
    data["email_verified"][password_created_users[rng.random(len(password_created_users)) < 0.83]] = 1
    email_verified_users = np.where(data["email_verified"] == 1)[0]

    data["payment_added"] = np.zeros(n_users, dtype=int)
    data["payment_added"][email_verified_users[rng.random(len(email_verified_users)) < 0.80]] = 1
    payment_added_users = np.where(data["payment_added"] == 1)[0]

    data["first_purchase"] = np.zeros(n_users, dtype=int)
    data["first_purchase"][payment_added_users[rng.random(len(payment_added_users)) < 0.50]] = 1

    return pd.DataFrame(data)


def compute_funnel_stages(df: pd.DataFrame) -> dict:
    stages = {
        "Sign Up": (df["signup_completed"] == 1).sum(),
        "Email Entered": (df["email_entered"] == 1).sum(),
        "Password Created": (df["password_created"] == 1).sum(),
        "Email Verified": (df["email_verified"] == 1).sum(),
        "Payment Added": (df["payment_added"] == 1).sum(),
        "First Purchase": (df["first_purchase"] == 1).sum(),
    }
    return stages


def compute_dropoff_analysis(stages: dict) -> pd.DataFrame:
    stage_list = list(stages.values())
    stage_names = list(stages.keys())

    dropoff_rows = []
    for i in range(len(stage_list) - 1):
        current_count = stage_list[i]
        next_count = stage_list[i + 1]
        lost = current_count - next_count
        drop_rate = (lost / current_count * 100) if current_count > 0 else 0
        completion_rate = 100 - drop_rate

        dropoff_rows.append(
            {
                "from_stage": stage_names[i],
                "to_stage": stage_names[i + 1],
                "users_at_start": current_count,
                "users_after": next_count,
                "users_lost": lost,
                "drop_rate_%": round(drop_rate, 2),
                "completion_rate_%": round(completion_rate, 2),
            }
        )

    return pd.DataFrame(dropoff_rows)


def identify_biggest_leak(dropoff_df: pd.DataFrame) -> tuple:
    biggest_leak_idx = dropoff_df["users_lost"].idxmax()
    biggest_leak_row = dropoff_df.loc[biggest_leak_idx]
    return biggest_leak_idx, biggest_leak_row


def plot_funnel_chart(stages: dict, dropoff_df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    stage_names = list(stages.keys())
    stage_counts = list(stages.values())

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(stage_names)), stage_counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=45, ha="right")
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("User Funnel: Sign-Up to First Purchase", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(stage_counts) * 1.15)

    for i, (stage, count) in enumerate(zip(stage_names, stage_counts)):
        ax.text(i, count + 100, str(count), ha="center", va="bottom", fontweight="bold", fontsize=10)
        if i < len(stage_counts) - 1:
            drop_pct = dropoff_df.iloc[i]["drop_rate_%"]
            ax.text(i + 0.5, (stage_counts[i] + stage_counts[i + 1]) / 2, f"{drop_pct:.1f}% drop", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "funnel_chart.png"), dpi=150)
    plt.close(fig)


def plot_dropoff_analysis(dropoff_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(dropoff_df))
    ax.barh(x_pos, dropoff_df["users_lost"], color="#ef4444", alpha=0.8, edgecolor="black")
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f"{row['from_stage']} → {row['to_stage']}" for _, row in dropoff_df.iterrows()])
    ax.set_xlabel("Users Lost", fontsize=12)
    ax.set_title("Drop-Off Analysis: Where Users Leave the Funnel", fontsize=14, fontweight="bold")

    for i, (idx, row) in enumerate(dropoff_df.iterrows()):
        ax.text(row["users_lost"] + 20, i, f"{row['users_lost']} ({row['drop_rate_%']:.1f}%)", va="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dropoff_analysis.png"), dpi=150)
    plt.close(fig)


def analyze_funnel(df: pd.DataFrame) -> None:
    ensure_output_dir(OUTPUT_DIR)

    stages = compute_funnel_stages(df)
    dropoff_df = compute_dropoff_analysis(stages)
    biggest_leak_idx, biggest_leak = identify_biggest_leak(dropoff_df)

    print("=== Funnel Summary ===")
    for stage, count in stages.items():
        print(f"{stage}: {count:,} users")

    print("\n=== Drop-Off Analysis ===")
    print(dropoff_df.to_string(index=False))

    print("\n=== Biggest Bottleneck ===")
    print(f"Stage: {biggest_leak['from_stage']} → {biggest_leak['to_stage']}")
    print(f"Users lost: {biggest_leak['users_lost']:,}")
    print(f"Drop rate: {biggest_leak['drop_rate_%']:.1f}%")

    print("\n=== Business Impact ===")
    first_purchase_count = stages["First Purchase"]
    print(f"Current first purchases: {first_purchase_count:,}")
    improvement_potential = biggest_leak["users_lost"]
    print(f"If we fix the biggest leak, potential new first purchases: {improvement_potential:,}")
    improvement_pct = (improvement_potential / first_purchase_count * 100) if first_purchase_count > 0 else 0
    print(f"Potential uplift: {improvement_pct:.1f}%")

    plot_funnel_chart(stages, dropoff_df)
    plot_dropoff_analysis(dropoff_df)

    csv_path = os.path.join(OUTPUT_DIR, "funnel_analysis.csv")
    dropoff_df.to_csv(csv_path, index=False)
    print(f"\nFunnel analysis saved to {csv_path}")


def main() -> None:
    df = build_funnel_dataset()
    analyze_funnel(df)


if __name__ == "__main__":
    main()
