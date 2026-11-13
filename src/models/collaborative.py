"""
Collaborative Filtering Recommender using Latent Factor Matrix Factorization.
Decomposes implicit interaction matrix R = P * Q^T using TruncatedSVD.
Captures latent co-interaction patterns, user taste profiles, and item affinity.
Provides item-item latent similarity for explainability.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from src.models.base import BaseRecommender
from src.config import CF_LATENT_FACTORS, CF_RANDOM_STATE


class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(self, n_factors: int = CF_LATENT_FACTORS, random_state: int = CF_RANDOM_STATE):
        super().__init__(name="CollaborativeFiltering (SVD)")
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = None

        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_factors = None  # shape: (n_users, n_factors)
        self.item_factors = None  # shape: (n_items, n_factors)
        self.item_norm_factors = None

        self.products_df = None
        self.user_history = {}
        self.item_interactions = {}

    def fit(self, df_train: pd.DataFrame, df_products: pd.DataFrame, df_users: Optional[pd.DataFrame] = None) -> "CollaborativeFilteringRecommender":
        self.products_df = df_products.set_index("product_id")

        # Map unique users and items from training set
        unique_users = sorted(df_train["user_id"].unique())
        all_catalog_items = sorted(df_products["product_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.item_to_idx = {p: i for i, p in enumerate(all_catalog_items)}
        self.idx_to_item = {i: p for i, p in enumerate(all_catalog_items)}

        # Record user history
        self.user_history = df_train.groupby("user_id")["product_id"].apply(set).to_dict()
        self.item_interactions = df_train.groupby("product_id")["weight"].sum().to_dict()

        # Build sparse implicit interaction matrix
        # Aggregate multiple interactions between same user and item: log1p(sum(weight))
        aggregated = df_train.groupby(["user_id", "product_id"])["weight"].sum().reset_index()
        aggregated["score"] = np.log1p(aggregated["weight"])

        row_indices = [self.user_to_idx[u] for u in aggregated["user_id"]]
        col_indices = [self.item_to_idx[p] for p in aggregated["product_id"]]
        data_values = aggregated["score"].values

        n_u = len(unique_users)
        n_i = len(all_catalog_items)
        sparse_matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(n_u, n_i))

        # Matrix Factorization via TruncatedSVD
        actual_factors = min(self.n_factors, min(n_u, n_i) - 1)
        self.svd = TruncatedSVD(n_components=actual_factors, random_state=self.random_state)

        # U * Sigma
        user_embeddings = self.svd.fit_transform(sparse_matrix)
        # V^T
        item_components = self.svd.components_

        self.user_factors = user_embeddings
        self.item_factors = item_components.T  # shape: (n_items, actual_factors)

        # Normalized item factors for cosine similarity lookups
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.item_norm_factors = self.item_factors / norms

        self.is_fitted = True
        return self

    def predict_score(self, user_id: str, product_id: str) -> float:
        if not self.is_fitted:
            return 0.0
        if user_id not in self.user_to_idx or product_id not in self.item_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        p_idx = self.item_to_idx[product_id]

        dot_product = float(np.dot(self.user_factors[u_idx], self.item_factors[p_idx]))
        # Sigmoid or non-linear scaling into [0, 1]
        return float(1.0 / (1.0 + np.exp(-dot_product)))

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend().")

        # Cold-start user handling: User not present in training
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        user_vec = self.user_factors[u_idx]

        # Compute dot product across all items in catalog
        raw_scores = np.dot(self.item_factors, user_vec)

        # Min-max scale user's predicted scores
        s_min, s_max = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - s_min) / (s_max - s_min + 1e-9)

        seen = self.user_history.get(user_id, set()) if exclude_interacted else set()

        # Sort candidate indices by score descending
        sorted_indices = np.argsort(-norm_scores)

        results = []
        for idx in sorted_indices:
            p_id = self.idx_to_item[idx]
            if p_id in seen:
                continue

            prod_info = self.products_df.loc[p_id] if p_id in self.products_df.index else None
            cat = prod_info["category"] if prod_info is not None else "Unknown"

            if category_filter and cat != category_filter:
                continue

            title = prod_info["title"] if prod_info is not None else p_id
            price = float(prod_info["price"]) if prod_info is not None else 0.0

            results.append({
                "product_id": p_id,
                "score": round(float(norm_scores[idx]), 4),
                "title": title,
                "category": cat,
                "price": price,
                "model": self.name,
                "signal": "Collaborative taste match based on latent co-interaction factors"
            })

            if len(results) >= top_k:
                break

        return results

    def explain(self, user_id: str, product_id: str) -> Dict[str, Any]:
        """Explain recommendation using most similar past item the user interacted with."""
        if not self.is_fitted or user_id not in self.user_to_idx or product_id not in self.item_to_idx:
            return {
                "model": self.name,
                "user_id": user_id,
                "product_id": product_id,
                "score": 0.0,
                "reason": "Cold-start entity (user or item not present in collaborative training matrix).",
                "is_cold": True
            }

        p_idx = self.item_to_idx[product_id]
        user_seen = self.user_history.get(user_id, set())
        target_vec = self.item_norm_factors[p_idx]

        # Find which item in user's history has highest cosine similarity to target item
        most_similar_item = None
        max_sim = -1.0
        for past_p in user_seen:
            if past_p in self.item_to_idx:
                past_idx = self.item_to_idx[past_p]
                sim = float(np.dot(target_vec, self.item_norm_factors[past_idx]))
                if sim > max_sim:
                    max_sim = sim
                    most_similar_item = past_p

        score = self.predict_score(user_id, product_id)
        prod_title = self.products_df.loc[product_id]["title"] if product_id in self.products_df.index else product_id

        if most_similar_item and most_similar_item in self.products_df.index:
            past_title = self.products_df.loc[most_similar_item]["title"]
            reason = f"Users with similar taste who interacted with '{past_title}' also engaged strongly with '{prod_title}' (Latent affinity: {max_sim:.2f})."
        else:
            reason = f"Strong alignment with user's latent preference profile (Score: {score:.2f})."

        return {
            "model": self.name,
            "user_id": user_id,
            "product_id": product_id,
            "title": prod_title,
            "score": round(score, 4),
            "reason": reason,
            "latent_match": {
                "top_similar_past_item": most_similar_item,
                "similarity": round(max_sim, 4) if max_sim >= 0 else None
            }
        }
