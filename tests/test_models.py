"""
Unit tests for model artifact loading, recommendation outputs, cold-start fallback, and explainability.
Demonstrates Acceptance Test 2, 3 & 4.
"""
import pytest
import joblib
from pathlib import Path
from src.config import (
    POPULARITY_MODEL_FILE, COLLABORATIVE_MODEL_FILE,
    CONTENT_MODEL_FILE, HYBRID_MODEL_FILE
)


@pytest.fixture(scope="module")
def models():
    assert POPULARITY_MODEL_FILE.exists(), "Popularity model artifact not found"
    assert COLLABORATIVE_MODEL_FILE.exists(), "CF model artifact not found"
    assert CONTENT_MODEL_FILE.exists(), "Content model artifact not found"
    assert HYBRID_MODEL_FILE.exists(), "Hybrid model artifact not found"

    return {
        "popularity": joblib.load(POPULARITY_MODEL_FILE),
        "collaborative": joblib.load(COLLABORATIVE_MODEL_FILE),
        "content": joblib.load(CONTENT_MODEL_FILE),
        "hybrid": joblib.load(HYBRID_MODEL_FILE),
    }


def test_models_are_fitted(models):
    for name, model in models.items():
        assert model.is_fitted is True, f"Model {name} should be marked as fitted"


def test_popularity_recommender(models):
    recs = models["popularity"].recommend(user_id="USER_0001", top_k=5)
    assert len(recs) == 5
    assert all("score" in r and "product_id" in r for r in recs)
    # Check descending scores
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_collaborative_recommender_warm_user(models):
    # Warm user who has interactions in training
    recs = models["collaborative"].recommend(user_id="USER_0001", top_k=5)
    assert len(recs) == 5
    assert all("score" in r for r in recs)


def test_content_based_recommender(models):
    recs = models["content"].recommend(user_id="USER_0001", top_k=5)
    assert len(recs) == 5
    assert all("score" in r for r in recs)


def test_hybrid_recommender_warm_user(models):
    recs = models["hybrid"].recommend(user_id="USER_0001", top_k=5)
    assert len(recs) == 5
    assert all("score" in r and "sub_scores" in r for r in recs)


def test_hybrid_recommender_cold_start_fallback(models):
    """
    CRITICAL TEST:
    Verify that an unseen user ID triggers cold-start fallback in Hybrid model,
    returning valid recommendations without throwing an unhandled exception.
    """
    cold_user = "USER_TOTALLY_NEW_COLD_START"
    recs = models["hybrid"].recommend(user_id=cold_user, top_k=5)
    assert len(recs) == 5
    assert all("score" in r for r in recs)
    assert any(r.get("is_cold_fallback") for r in recs)


def test_hybrid_explainability(models):
    """Verify that explain() returns meaningful signals, weights, and decision support disclaimers."""
    exp = models["hybrid"].explain(user_id="USER_0001", product_id="PROD_0001")
    assert "hybrid_score" in exp
    assert "weights_used" in exp
    assert "sub_scores" in exp
    assert "decision_support_limitations" in exp
    assert len(exp["synthesis_summary"]) > 10
