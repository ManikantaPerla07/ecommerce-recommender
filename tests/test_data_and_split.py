"""
Unit tests for data generation, referential integrity, and chronological leakage prevention.
Demonstrates Acceptance Test 1 & Acceptance Test 5.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.config import (
    PRODUCTS_FILE, USERS_FILE, INTERACTIONS_FILE,
    TRAIN_INTERACTIONS_FILE, VAL_INTERACTIONS_FILE, TEST_INTERACTIONS_FILE
)


def test_datasets_exist_and_meet_size_requirements():
    """Verify datasets exist and meet the minimum record threshold."""
    assert PRODUCTS_FILE.exists(), f"Missing {PRODUCTS_FILE}"
    assert USERS_FILE.exists(), f"Missing {USERS_FILE}"
    assert INTERACTIONS_FILE.exists(), f"Missing {INTERACTIONS_FILE}"

    df_products = pd.read_csv(PRODUCTS_FILE)
    df_users = pd.read_csv(USERS_FILE)
    df_interactions = pd.read_csv(INTERACTIONS_FILE)

    assert len(df_products) >= 200, "Catalog should have at least 200 products"
    assert len(df_users) >= 1000, "Users should have at least 1,000 records"
    assert len(df_interactions) >= 1000, "Interactions must exceed 1,000 records"


def test_referential_integrity():
    """Ensure no orphaned foreign keys in interactions."""
    df_products = pd.read_csv(PRODUCTS_FILE)
    df_users = pd.read_csv(USERS_FILE)
    df_interactions = pd.read_csv(INTERACTIONS_FILE)

    valid_users = set(df_users["user_id"])
    valid_products = set(df_products["product_id"])

    orphaned_users = df_interactions[~df_interactions["user_id"].isin(valid_users)]
    orphaned_prods = df_interactions[~df_interactions["product_id"].isin(valid_products)]

    assert len(orphaned_users) == 0, f"Found {len(orphaned_users)} orphaned user interactions"
    assert len(orphaned_prods) == 0, f"Found {len(orphaned_prods)} orphaned product interactions"


def test_zero_temporal_leakage_in_splits():
    """
    CRITICAL ACCEPTANCE TEST:
    Assert strictly that Train timestamps < Val timestamps < Test timestamps.
    Prevents future interaction leakage.
    """
    df_train = pd.read_csv(TRAIN_INTERACTIONS_FILE)
    df_val = pd.read_csv(VAL_INTERACTIONS_FILE)
    df_test = pd.read_csv(TEST_INTERACTIONS_FILE)

    train_max_time = pd.to_datetime(df_train["timestamp"]).max()
    val_min_time = pd.to_datetime(df_val["timestamp"]).min()
    val_max_time = pd.to_datetime(df_val["timestamp"]).max()
    test_min_time = pd.to_datetime(df_test["timestamp"]).min()

    assert train_max_time < val_min_time, (
        f"Temporal leakage detected: Train max ({train_max_time}) >= Val min ({val_min_time})"
    )
    assert val_max_time < test_min_time, (
        f"Temporal leakage detected: Val max ({val_max_time}) >= Test min ({test_min_time})"
    )
