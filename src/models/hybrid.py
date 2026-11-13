"""
Hybrid Recommendation Engine.
Harmonizes Collaborative Filtering, Content-Based TF-IDF, and Popularity Baseline.
Includes dynamic cold-start fallback switches, multi-signal explainability,
and confidence calibration for decision-support interfaces.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.content_based import ContentBasedRecommender
from src.config import HYBRID_DEFAULT_WEIGHTS


class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        popularity_model: Optional[PopularityRecommender] = None,
        cf_model: Optional[CollaborativeFilteringRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(name="HybridRecommender")
        self.pop_model = popularity_model or PopularityRecommender()
        self.cf_model = cf_model or CollaborativeFilteringRecommender()
        self.content_model = content_model or ContentBasedRecommender()
        self.weights = weights or HYBRID_DEFAULT_WEIGHTS.copy()

        self.products_df = None
        self.user_history = {}
        self.train_users = set()
        self.train_items = set()

    def fit(self, df_train: pd.DataFrame, df_products: pd.DataFrame, df_users: Optional[pd.DataFrame] = None) -> "HybridRecommender":
        self.products_df = df_products.set_index("product_id")
        self.user_history = df_train.groupby("user_id")["product_id"].apply(set).to_dict()
        self.train_users = set(df_train["user_id"].unique())
        self.train_items = set(df_train["product_id"].unique())

        # Fit all constituent sub-models
        if not self.pop_model.is_fitted:
            self.pop_model.fit(df_train, df_products, df_users)
        if not self.cf_model.is_fitted:
            self.cf_model.fit(df_train, df_products, df_users)
        if not self.content_model.is_fitted:
            self.content_model.fit(df_train, df_products, df_users)

        self.is_fitted = True
        return self

    def _get_dynamic_weights(self, user_id: str) -> Dict[str, float]:
        """Adjust ensemble weights dynamically if user is in cold-start regime."""
        n_inter = len(self.user_history.get(user_id, set()))
        if user_id not in self.train_users or n_inter == 0:
            # Completely new cold-start user
            return {"collaborative": 0.0, "content": 0.40, "popularity": 0.60}
        elif n_inter <= 2:
            # Low-activity user: dampen CF, boost content & popularity
            return {"collaborative": 0.20, "content": 0.45, "popularity": 0.35}
        else:
            # Full personalized regime
            return self.weights

    def predict_score(self, user_id: str, product_id: str) -> float:
        if not self.is_fitted:
            return 0.0

        weights = self._get_dynamic_weights(user_id)
        cf_score = self.cf_model.predict_score(user_id, product_id)
        content_score = self.content_model.predict_score(user_id, product_id)
        pop_score = self.pop_model.predict_score(user_id, product_id)

        hybrid_score = (
            weights["collaborative"] * cf_score +
            weights["content"] * content_score +
            weights["popularity"] * pop_score
        )
        return float(hybrid_score)

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("HybridRecommender must be fitted before recommend().")

        weights = self._get_dynamic_weights(user_id)
        is_cold_user = (weights["collaborative"] == 0.0)

        seen = self.user_history.get(user_id, set()) if exclude_interacted else set()
        all_prods = self.products_df.index.tolist()

        candidate_scores = []
        for p_id in all_prods:
            if p_id in seen:
                continue

            row = self.products_df.loc[p_id]
            cat = row["category"]
            if category_filter and cat != category_filter:
                continue

            # Gather sub-scores
            cf_s = self.cf_model.predict_score(user_id, p_id)
            ct_s = self.content_model.predict_score(user_id, p_id)
            pop_s = self.pop_model.predict_score(user_id, p_id)

            score = (weights["collaborative"] * cf_s +
                     weights["content"] * ct_s +
                     weights["popularity"] * pop_s)

            candidate_scores.append({
                "product_id": p_id,
                "score": round(float(score), 4),
                "title": row["title"],
                "category": cat,
                "price": float(row["price"]),
                "model": self.name,
                "is_cold_fallback": is_cold_user,
                "sub_scores": {
                    "cf": round(float(cf_s), 3),
                    "content": round(float(ct_s), 3),
                    "popularity": round(float(pop_s), 3)
                }
            })

        # Sort descending by hybrid score
        candidate_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = candidate_scores[:top_k]

        # Add readable decision-support signals
        for item in top_results:
            sub = item["sub_scores"]
            if is_cold_user:
                item["signal"] = "Cold-start fallback: Balanced by category profile & trending popularity"
            elif sub["cf"] >= sub["content"] and sub["cf"] >= sub["popularity"]:
                item["signal"] = "Personalized taste match (Collaborative Filtering primary driver)"
            elif sub["content"] >= sub["cf"]:
                item["signal"] = "Semantic interest match (Product features align with your viewing habits)"
            else:
                item["signal"] = "Trending catalog recommendation with personalized blend"

        return top_results

    def explain(self, user_id: str, product_id: str) -> Dict[str, Any]:
        """Synthesize explainability signals across all sub-models for human decision support."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        prod_title = self.products_df.loc[product_id]["title"] if product_id in self.products_df.index else product_id
        category = self.products_df.loc[product_id]["category"] if product_id in self.products_df.index else "Unknown"

        weights = self._get_dynamic_weights(user_id)
        cf_exp = self.cf_model.explain(user_id, product_id)
        content_exp = self.content_model.explain(user_id, product_id)
        pop_exp = self.pop_model.explain(user_id, product_id)

        cf_s = self.cf_model.predict_score(user_id, product_id)
        ct_s = self.content_model.predict_score(user_id, product_id)
        pop_s = self.pop_model.predict_score(user_id, product_id)
        overall = self.predict_score(user_id, product_id)

        is_cold = (weights["collaborative"] == 0.0)

        # Human-interpretable rationale synthesis
        rationale_parts = []
        if is_cold:
            rationale_parts.append("Cold-start user profile: Recommendation is weighted toward trending popularity and category discovery.")
        else:
            if cf_s > 0.4:
                rationale_parts.append(f"Collaborative signal ({cf_s:.2f}): {cf_exp['reason']}")
            if ct_s > 0.3:
                rationale_parts.append(f"Content signal ({ct_s:.2f}): {content_exp['reason']}")
            if pop_s > 0.5:
                rationale_parts.append(f"Popularity signal ({pop_s:.2f}): High customer engagement across the platform.")

        if not rationale_parts:
            rationale_parts.append(f"Combined cross-model discovery recommendation with hybrid confidence score of {overall:.2f}.")

        return {
            "model": self.name,
            "user_id": user_id,
            "product_id": product_id,
            "title": prod_title,
            "category": category,
            "hybrid_score": round(overall, 4),
            "is_cold_start_fallback": is_cold,
            "weights_used": weights,
            "sub_scores": {
                "collaborative_filtering": round(cf_s, 4),
                "content_based": round(ct_s, 4),
                "popularity_baseline": round(pop_s, 4)
            },
            "sub_explanations": {
                "collaborative": cf_exp,
                "content": content_exp,
                "popularity": pop_exp
            },
            "synthesis_summary": " ".join(rationale_parts),
            "decision_support_limitations": (
                "Decision-Support Notice: This score reflects historical implicit feedback and catalog similarity. "
                "It does not verify real-time user intent, budget constraints, or external market trends. "
                "Human review is advised before using in automated high-stakes promotions."
            )
        }
