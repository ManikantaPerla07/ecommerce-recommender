"""
Content-Based Product Recommender using TF-IDF and Semantic Text Profiles.
Builds TF-IDF representations of product metadata (titles, categories, descriptions, tags)
and constructs weighted user interest vectors from past interactions.
Ranks candidate items via cosine similarity and extracts matching feature keywords for explainability.
Naturally resolves Item Cold-Start.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base import BaseRecommender
from src.config import TFIDF_MAX_FEATURES


class ContentBasedRecommender(BaseRecommender):
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        super().__init__(name="ContentBased (TF-IDF)")
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.item_tfidf = None  # shape: (n_items, max_features)
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.feature_names = []

        self.products_df = None
        self.user_profiles = {}  # user_id -> normalized dense vector
        self.user_history = {}

    def fit(self, df_train: pd.DataFrame, df_products: pd.DataFrame, df_users: Optional[pd.DataFrame] = None) -> "ContentBasedRecommender":
        self.products_df = df_products.set_index("product_id")

        all_items = sorted(df_products["product_id"].unique())
        self.item_to_idx = {p: i for i, p in enumerate(all_items)}
        self.idx_to_item = {i: p for i, p in enumerate(all_items)}

        # Prepare rich content string per item
        content_strings = []
        for p in all_items:
            row = self.products_df.loc[p]
            text = f"{row['title']} {row['category']} {row['subcategory']} {row['brand']} {row['description']} {row['tags']}".lower()
            content_strings.append(text)

        # Fit TF-IDF matrix
        self.item_tfidf = self.vectorizer.fit_transform(content_strings).toarray()
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Record user history
        self.user_history = df_train.groupby("user_id")["product_id"].apply(set).to_dict()

        # Build user profile vectors: weighted average of item vectors
        user_grouped = df_train.groupby("user_id")
        for u_id, group in user_grouped:
            profile = np.zeros(self.item_tfidf.shape[1], dtype=np.float32)
            for _, row in group.iterrows():
                p_id = row["product_id"]
                if p_id in self.item_to_idx:
                    weight = float(row["weight"])
                    profile += weight * self.item_tfidf[self.item_to_idx[p_id]]

            norm = np.linalg.norm(profile)
            if norm > 0:
                self.user_profiles[u_id] = profile / norm

        # Handle user cold start with declared category preferences from users.csv if provided
        if df_users is not None:
            user_meta = df_users.set_index("user_id")
            for u_id in df_users["user_id"]:
                if u_id not in self.user_profiles and u_id in user_meta.index:
                    pref_cat = user_meta.loc[u_id]["primary_category"]
                    # Build pseudo-profile from category text
                    pseudo_vec = self.vectorizer.transform([pref_cat.lower()]).toarray()[0]
                    p_norm = np.linalg.norm(pseudo_vec)
                    if p_norm > 0:
                        self.user_profiles[u_id] = pseudo_vec / p_norm

        self.is_fitted = True
        return self

    def predict_score(self, user_id: str, product_id: str) -> float:
        if not self.is_fitted:
            return 0.0
        if user_id not in self.user_profiles or product_id not in self.item_to_idx:
            return 0.0

        u_vec = self.user_profiles[user_id]
        p_vec = self.item_tfidf[self.item_to_idx[product_id]]
        sim = float(np.dot(u_vec, p_vec))
        return max(0.0, sim)

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend().")

        if user_id not in self.user_profiles:
            return []

        u_vec = self.user_profiles[user_id]
        # Cosine similarity against all items: item_tfidf is already L2-normalized rows
        sim_scores = np.dot(self.item_tfidf, u_vec)

        seen = self.user_history.get(user_id, set()) if exclude_interacted else set()
        sorted_indices = np.argsort(-sim_scores)

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
            score = float(sim_scores[idx])

            results.append({
                "product_id": p_id,
                "score": round(score, 4),
                "title": title,
                "category": cat,
                "price": price,
                "model": self.name,
                "signal": "High keyword and category semantic alignment with your interaction profile"
            })

            if len(results) >= top_k:
                break

        return results

    def explain(self, user_id: str, product_id: str) -> Dict[str, Any]:
        """Explain recommendation by extracting top matching TF-IDF keywords."""
        if not self.is_fitted or user_id not in self.user_profiles or product_id not in self.item_to_idx:
            return {
                "model": self.name,
                "user_id": user_id,
                "product_id": product_id,
                "score": 0.0,
                "reason": "Cold-start entity without content overlap.",
                "is_cold": True
            }

        u_vec = self.user_profiles[user_id]
        p_idx = self.item_to_idx[product_id]
        p_vec = self.item_tfidf[p_idx]

        # Element-wise contribution of features to the dot product
        overlap = u_vec * p_vec
        top_indices = np.argsort(-overlap)[:5]
        top_features = [self.feature_names[i] for i in top_indices if overlap[i] > 0]

        prod_title = self.products_df.loc[product_id]["title"] if product_id in self.products_df.index else product_id
        score = float(np.dot(u_vec, p_vec))

        if top_features:
            features_str = ", ".join(f"'{f}'" for f in top_features)
            reason = f"Matches your demonstrated interests in features: {features_str} (Content similarity: {score:.2f})."
        else:
            reason = f"General category similarity with your past browsing history."

        return {
            "model": self.name,
            "user_id": user_id,
            "product_id": product_id,
            "title": prod_title,
            "score": round(score, 4),
            "reason": reason,
            "matched_keywords": top_features,
            "feature_signals": {
                "top_keywords": top_features,
                "cosine_similarity": round(score, 4)
            }
        }
