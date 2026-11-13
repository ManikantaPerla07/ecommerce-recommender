"""
Exploratory Data Analysis (EDA) and Data Quality Profiling Module.
Performs comprehensive data checks:
- Matrix sparsity calculation
- User activity and item popularity distributions
- Action funnel breakdown (views -> carts -> purchases -> ratings)
- Price and rating distributions
- Missing value, duplicate, and referential integrity audits
- Leakage risk assessment
- Generates JSON summary and terminal report
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    PRODUCTS_FILE, USERS_FILE, INTERACTIONS_FILE, REPORTS_DIR
)


def run_eda(save_report: bool = True) -> dict:
    """Run full EDA and return dictionary of findings."""
    print("=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS & DATA QUALITY AUDIT")
    print("=" * 60)

    # 1. Load data
    df_products = pd.read_csv(PRODUCTS_FILE)
    df_users = pd.read_csv(USERS_FILE)
    df_interactions = pd.read_csv(INTERACTIONS_FILE)

    num_products = len(df_products)
    num_users = len(df_users)
    num_interactions = len(df_interactions)

    # 2. Referential integrity & Missing Value Checks
    missing_prods = df_products.isnull().sum().to_dict()
    missing_users = df_users.isnull().sum().to_dict()
    missing_inter = df_interactions.isnull().sum().to_dict()

    # Note: 'rating' is expected to be null for views/carts/purchases, verify this
    rating_null_expected = df_interactions[df_interactions["event_type"] != "rating"]["rating"].isnull().all()
    rating_filled_for_ratings = df_interactions[df_interactions["event_type"] == "rating"]["rating"].notnull().all()

    # Foreign key checks
    invalid_user_fk = len(df_interactions[~df_interactions["user_id"].isin(df_users["user_id"])])
    invalid_prod_fk = len(df_interactions[~df_interactions["product_id"].isin(df_products["product_id"])])

    # Duplicate check
    duplicate_interactions = df_interactions.duplicated(subset=["user_id", "product_id", "timestamp"]).sum()

    # 3. Interaction Matrix Sparsity
    unique_user_interactions = df_interactions["user_id"].nunique()
    unique_item_interactions = df_interactions["product_id"].nunique()
    total_possible_cells = unique_user_interactions * unique_item_interactions
    unique_pairs = df_interactions[["user_id", "product_id"]].drop_duplicates().shape[0]
    sparsity = (1.0 - (unique_pairs / total_possible_cells)) * 100.0

    # 4. Action Funnel
    event_counts = df_interactions["event_type"].value_counts().to_dict()
    event_pcts = {k: round(v / num_interactions * 100, 2) for k, v in event_counts.items()}

    # 5. User Activity Profile
    user_counts = df_interactions.groupby("user_id").size()
    user_activity_stats = {
        "min": int(user_counts.min()),
        "p25": float(user_counts.quantile(0.25)),
        "median": float(user_counts.median()),
        "mean": round(float(user_counts.mean()), 2),
        "p75": float(user_counts.quantile(0.75)),
        "p95": float(user_counts.quantile(0.95)),
        "max": int(user_counts.max()),
        "users_with_under_3_events": int((user_counts <= 3).sum())
    }

    # 6. Item Popularity & Pareto Principle
    item_counts = df_interactions.groupby("product_id").size().sort_values(ascending=False)
    top_20_pct_items = int(np.ceil(0.20 * len(item_counts)))
    interactions_top_20_pct = item_counts.iloc[:top_20_pct_items].sum()
    pareto_share_pct = round((interactions_top_20_pct / num_interactions) * 100, 2)

    item_popularity_stats = {
        "min": int(item_counts.min()),
        "p25": float(item_counts.quantile(0.25)),
        "median": float(item_counts.median()),
        "mean": round(float(item_counts.mean()), 2),
        "p75": float(item_counts.quantile(0.75)),
        "max": int(item_counts.max()),
        "pareto_share_top_20pct_items": f"{pareto_share_pct}% of total interaction volume"
    }

    # 7. Category Distribution
    cat_distribution = df_products["category"].value_counts().to_dict()
    price_stats = {
        "min": float(df_products["price"].min()),
        "mean": round(float(df_products["price"].mean()), 2),
        "median": float(df_products["price"].median()),
        "max": float(df_products["price"].max()),
        "std": round(float(df_products["price"].std()), 2)
    }

    # 8. Leakage Risk Analysis
    df_interactions["dt"] = pd.to_datetime(df_interactions["timestamp"])
    is_chronological = df_interactions["dt"].is_monotonic_increasing
    time_min = str(df_interactions["dt"].min())
    time_max = str(df_interactions["dt"].max())

    eda_summary = {
        "overview": {
            "num_products": num_products,
            "num_users": num_users,
            "num_interactions": num_interactions,
            "matrix_sparsity_pct": round(sparsity, 2),
            "time_window": f"{time_min} to {time_max}"
        },
        "data_quality": {
            "invalid_user_foreign_keys": invalid_user_fk,
            "invalid_product_foreign_keys": invalid_prod_fk,
            "duplicate_interactions": int(duplicate_interactions),
            "missing_values_products": missing_prods,
            "missing_values_users": missing_users,
            "missing_values_interactions_excl_ratings": {k: v for k, v in missing_inter.items() if k != "rating"},
            "rating_field_integrity": bool(rating_null_expected and rating_filled_for_ratings)
        },
        "funnel_distribution": {
            "counts": event_counts,
            "percentages": event_pcts
        },
        "user_activity": user_activity_stats,
        "item_popularity": item_popularity_stats,
        "catalog_summary": {
            "categories": cat_distribution,
            "price_stats": price_stats
        },
        "leakage_assessment": {
            "is_strictly_chronological": bool(is_chronological),
            "split_recommendation": "Chronological split (train on earliest 70%, val on next 15%, test on final 15%). Strictly avoids future interaction leakage."
        }
    }

    # Print clean summary
    print(f"Total Products:          {num_products}")
    print(f"Total Users:             {num_users}")
    print(f"Total Interactions:      {num_interactions}")
    print(f"User-Item Sparsity:      {sparsity:.2f}%")
    print("-" * 60)
    print("Action Funnel:")
    for event, count in event_counts.items():
        print(f"  - {event.capitalize():<10}: {count:>6} ({event_pcts[event]}%)")
    print("-" * 60)
    print(f"User Activity:           Median {user_activity_stats['median']} events/user (Max: {user_activity_stats['max']})")
    print(f"Cold-Start Users (<=3):  {user_activity_stats['users_with_under_3_events']} users")
    print(f"Item Popularity:         Top 20% items account for {pareto_share_pct}% of interactions (Power Law)")
    print(f"Price Range:             ${price_stats['min']:.2f} - ${price_stats['max']:.2f} (Median: ${price_stats['median']:.2f})")
    print("-" * 60)
    print("Data Integrity Checks:")
    print(f"  - Foreign Key Errors:  Users: {invalid_user_fk}, Products: {invalid_prod_fk}")
    print(f"  - Duplicates:          {duplicate_interactions}")
    print(f"  - Chronological Order: {'VALID (Monotonic)' if is_chronological else 'INVALID'}")
    print("=" * 60)

    if save_report:
        report_file = REPORTS_DIR / "eda_summary.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=2)
        print(f"[OK] EDA summary saved to {report_file}")

    return eda_summary


if __name__ == "__main__":
    run_eda()
