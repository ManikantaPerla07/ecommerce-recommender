"""
Abstract Base Class for all Recommendation Models.
Defines unified interface for model training, recommendation generation,
scoring, and explanation.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseRecommender(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, df_products: pd.DataFrame, df_users: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """Fit model parameters on training interactions and metadata."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Top-K recommendations for a target user.
        Returns list of dicts: [{"product_id": ..., "score": ..., "title": ..., "category": ...}]
        """
        pass

    @abstractmethod
    def predict_score(self, user_id: str, product_id: str) -> float:
        """Predict preference score for a specific user-product pair."""
        pass

    def explain(self, user_id: str, product_id: str) -> Dict[str, Any]:
        """Explain recommendation signal / feature drivers for decision support."""
        return {
            "model": self.name,
            "user_id": user_id,
            "product_id": product_id,
            "score": self.predict_score(user_id, product_id),
            "reason": f"Recommended by {self.name} based on historical patterns."
        }
