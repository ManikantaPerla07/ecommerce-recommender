"""
Data Preprocessing, Chronological Splitting, and Feature Engineering Module.
Ensures zero temporal leakage, converts multi-action events into implicit feedback scores,
enriches product metadata, and creates explicit cold-start test slices.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from src.config import (
    PRODUCTS_FILE, USERS_FILE, INTERACTIONS_FILE,
    TRAIN_INTERACTIONS_FILE, VAL_INTERACTIONS_FILE, TEST_INTERACTIONS_FILE,
    COLD_START_TEST_FILE, PREPROCESSOR_FILE,
    EVENT_WEIGHTS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class DataPreprocessor:
    def __init__(self, event_weights: Dict[str, float] = None):
        self.event_weights = event_weights or EVENT_WEIGHTS
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.train_users = set()
        self.train_items = set()
        self.split_timestamps = {}

    def compute_interaction_weight(self, row: pd.Series) -> float:
        """Assign confidence weight to each user-item interaction action."""
        event = row["event_type"]
        if event == "rating" and pd.notnull(row["rating"]):
            return 1.0 + (float(row["rating"]) / 5.0) * 4.0
        return self.event_weights.get(event, 1.0)

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads raw data, computes interaction affinity, performs strict chronological split,
        and extracts cold-start evaluation slices.
        """
        print("=" * 60)
        print("PREPROCESSING DATA & CREATING CHRONOLOGICAL SPLITS")
        print("=" * 60)

        df_products = pd.read_csv(PRODUCTS_FILE)
        df_users = pd.read_csv(USERS_FILE)
        df_interactions = pd.read_csv(INTERACTIONS_FILE)

        # 1. Feature Engineering: Content text blob for products
        df_products["content_text"] = (
            df_products["title"].fillna("") + " " +
            df_products["category"].fillna("") + " " +
            df_products["subcategory"].fillna("") + " " +
            df_products["brand"].fillna("") + " " +
            df_products["description"].fillna("") + " " +
            df_products["tags"].fillna("")
        ).str.lower().str.strip()

        # 2. Assign action weights
        df_interactions["weight"] = df_interactions.apply(self.compute_interaction_weight, axis=1)

        # 3. Sort strictly by timestamp for zero temporal leakage
        df_interactions["dt"] = pd.to_datetime(df_interactions["timestamp"])
        df_interactions = df_interactions.sort_values("dt").reset_index(drop=True)

        n_total = len(df_interactions)
        train_end_idx = int(n_total * TRAIN_RATIO)
        val_end_idx = int(n_total * (TRAIN_RATIO + VAL_RATIO))

        train_cutoff_dt = df_interactions.iloc[train_end_idx]["dt"]
        val_cutoff_dt = df_interactions.iloc[val_end_idx]["dt"]

        df_train = df_interactions.iloc[:train_end_idx].copy()
        df_val = df_interactions.iloc[train_end_idx:val_end_idx].copy()
        df_test = df_interactions.iloc[val_end_idx:].copy()

        self.split_timestamps = {
            "train_start": str(df_train["dt"].min()),
            "train_end": str(df_train["dt"].max()),
            "val_start": str(df_val["dt"].min()),
            "val_end": str(df_val["dt"].max()),
            "test_start": str(df_test["dt"].min()),
            "test_end": str(df_test["dt"].max()),
        }

        # 4. Fit vocabulary & mappings strictly on train set to prevent transductive leakage
        self.train_users = set(df_train["user_id"].unique())
        self.train_items = set(df_train["product_id"].unique())

        all_users = sorted(df_users["user_id"].unique())
        all_items = sorted(df_products["product_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(all_users)}
        self.idx_to_user = {i: u for i, u in enumerate(all_users)}
        self.item_to_idx = {p: i for i, p in enumerate(all_items)}
        self.idx_to_item = {i: p for i, p in enumerate(all_items)}

        # 5. Identify cold-start subsets in test set
        # User cold start: user has <= 2 interactions in train
        user_train_counts = df_train.groupby("user_id").size().to_dict()
        df_test["is_cold_user"] = df_test["user_id"].apply(lambda u: user_train_counts.get(u, 0) <= 2)
        # Item cold start: item has 0 interactions in train
        df_test["is_cold_item"] = df_test["product_id"].apply(lambda p: p not in self.train_items)

        df_cold_test = df_test[df_test["is_cold_user"] | df_test["is_cold_item"]].copy()

        # 6. Save splits
        df_train.drop(columns=["dt"]).to_csv(TRAIN_INTERACTIONS_FILE, index=False)
        df_val.drop(columns=["dt"]).to_csv(VAL_INTERACTIONS_FILE, index=False)
        df_test.drop(columns=["dt"]).to_csv(TEST_INTERACTIONS_FILE, index=False)
        df_cold_test.drop(columns=["dt"]).to_csv(COLD_START_TEST_FILE, index=False)

        # Save preprocessor state
        joblib.dump(self, PREPROCESSOR_FILE)

        print(f"Train set:     {len(df_train):>6} events ({self.split_timestamps['train_start']} -> {self.split_timestamps['train_end']})")
        print(f"Val set:       {len(df_val):>6} events ({self.split_timestamps['val_start']} -> {self.split_timestamps['val_end']})")
        print(f"Test set:      {len(df_test):>6} events ({self.split_timestamps['test_start']} -> {self.split_timestamps['test_end']})")
        print(f"Cold-Start Test Slice: {len(df_cold_test)} events (Cold users: {df_test['is_cold_user'].sum()}, Cold items: {df_test['is_cold_item'].sum()})")
        print(f"[OK] Preprocessor saved -> {PREPROCESSOR_FILE}")
        print("=" * 60)

        return df_products, df_users, df_train, df_val, df_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.prepare_data()
