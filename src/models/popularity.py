"""
Popularity Baseline Recommender.
Ranks products by total interaction weight and event frequencies in the training set.
Serves as the benchmark baseline against which all personalized models are compared,
and functions as the ultimate fallback for cold-start scenarios.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self):
        super().__init__(name="PopularityBaseline")
        self.item_scores = {}
        self.ranked_items = []
        self.category_ranked_items = {}
        self.products_df = None
        self.user_history = {}

    def fit(self, df_train: pd.DataFrame, df_products: pd.DataFrame, df_users: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        self.products_df = df_products.set_index("product_id")

        # Record user training history to allow excluding past interacted items
        self.user_history = df_train.groupby("user_id")["product_id"].apply(set).to_dict()

        # Compute weighted popularity per product
        item_grouped = df_train.groupby("product_id")["weight"].sum().to_dict()

        # Ensure all catalog items are present (even zero interaction ones with base prior)
        all_prods = df_products["product_id"].tolist()
        for p in all_prods:
            self.item_scores[p] = float(item_grouped.get(p, 0.0))

        # Normalize scores to [0, 1]
        max_score = max(self.item_scores.values()) if self.item_scores else 1.0
        if max_score > 0:
            self.normalized_scores = {p: s / max_score for p, s in self.item_scores.items()}
        else:
            self.normalized_scores = {p: 0.0 for p in self.item_scores}

        # Global ranked list
        self.ranked_items = sorted(all_prods, key=lambda p: self.item_scores[p], reverse=True)

        # Category-specific ranked lists
        for cat in df_products["category"].unique():
            cat_prods = df_products[df_products["category"] == cat]["product_id"].tolist()
            self.category_ranked_items[cat] = sorted(cat_prods, key=lambda p: self.item_scores.get(p, 0.0), reverse=True)

        self.is_fitted = True
        return self

    def predict_score(self, user_id: str, product_id: str) -> float:
        return self.normalized_scores.get(product_id, 0.0)

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend().")

        seen = self.user_history.get(user_id, set()) if exclude_interacted else set()

        if category_filter and category_filter in self.category_ranked_items:
            candidate_list = self.category_ranked_items[category_filter]
        else:
            candidate_list = self.ranked_items

        results = []
        for p_id in candidate_list:
            if p_id in seen:
                continue

            prod_info = self.products_df.loc[p_id] if p_id in self.products_df.index else None
            title = prod_info["title"] if prod_info is not None else p_id
            category = prod_info["category"] if prod_info is not None else "Unknown"
            price = float(prod_info["price"]) if prod_info is not None else 0.0

            results.append({
                "product_id": p_id,
                "score": round(self.normalized_scores.get(p_id, 0.0), 4),
                "title": title,
                "category": category,
                "price": price,
                "model": self.name,
                "signal": "High overall catalog engagement"
            })

            if len(results) >= top_k:
                break

        return results

    def explain(self, user_id: str, product_id: str) -> Dict[str, Any]:
        rank = self.ranked_items.index(product_id) + 1 if product_id in self.ranked_items else -1
        score = self.predict_score(user_id, product_id)
        prod_info = self.products_df.loc[product_id] if product_id in self.products_df.index else None
        title = prod_info["title"] if prod_info is not None else product_id

        return {
            "model": self.name,
            "user_id": user_id,
            "product_id": product_id,
            "title": title,
            "score": round(score, 4),
            "global_rank": rank,
            "reason": f"Ranked #{rank} across all catalog items based on high historical purchase and view volume.",
            "is_personalized": False,
            "feature_signals": {
                "popularity_score": round(score, 4),
                "raw_interaction_weight": round(self.item_scores.get(product_id, 0.0), 2)
            }
        }
