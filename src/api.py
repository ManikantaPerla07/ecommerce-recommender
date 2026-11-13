"""
FastAPI Inference REST API for E-Commerce Recommendation Engine.
Loads trained model artifacts from disk and serves real-time predictions,
detailed explainability payloads, catalog search, and offline evaluation metrics.
"""
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    PRODUCTS_FILE, USERS_FILE, TRAIN_INTERACTIONS_FILE,
    POPULARITY_MODEL_FILE, COLLABORATIVE_MODEL_FILE, CONTENT_MODEL_FILE,
    HYBRID_MODEL_FILE, METRICS_EVAL_FILE
)

# In-memory storage for loaded artifacts
_MODELS = {}
_CATALOG_DF = None
_USERS_DF = None
_EVAL_METRICS = None


def load_artifacts():
    global _MODELS, _CATALOG_DF, _USERS_DF, _EVAL_METRICS
    if not _MODELS:
        print("[API] Loading model artifacts from disk...")
        _MODELS["popularity"] = joblib.load(POPULARITY_MODEL_FILE)
        _MODELS["collaborative"] = joblib.load(COLLABORATIVE_MODEL_FILE)
        _MODELS["content"] = joblib.load(CONTENT_MODEL_FILE)
        _MODELS["hybrid"] = joblib.load(HYBRID_MODEL_FILE)

        _CATALOG_DF = pd.read_csv(PRODUCTS_FILE)
        _USERS_DF = pd.read_csv(USERS_FILE)

        if METRICS_EVAL_FILE.exists():
            import json
            with open(METRICS_EVAL_FILE, "r", encoding="utf-8") as f:
                _EVAL_METRICS = json.load(f)
        print("[API] All artifacts successfully loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="E-Commerce Product Recommendation Engine API",
    description="Decision-support ML service delivering personalized product recommendations with explainability and cold-start fallback.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Schemas
class RecommendRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "USER_0012"}, description="Unique user identifier")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    category_filter: Optional[str] = Field(None, json_schema_extra={"example": "Electronics"}, description="Optional filter by catalog category")
    model_type: Optional[str] = Field("hybrid", json_schema_extra={"example": "hybrid"}, description="Model choice: 'hybrid', 'collaborative', 'content', or 'popularity'")
    exclude_interacted: bool = Field(True, description="Whether to suppress items the user interacted with in training")


class ProductRecommendation(BaseModel):
    product_id: str
    score: float
    title: str
    category: str
    price: float
    model: str
    signal: Optional[str] = None
    is_cold_fallback: Optional[bool] = False


class RecommendResponse(BaseModel):
    user_id: str
    model_type: str
    top_k: int
    recommendations: List[ProductRecommendation]
    total_returned: int


class ExplainRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "USER_0012"})
    product_id: str = Field(..., json_schema_extra={"example": "PROD_0045"})
    model_type: Optional[str] = Field("hybrid", json_schema_extra={"example": "hybrid"})


@app.get("/health", summary="Service Health & Model Status")
def health_check():
    """Returns service health, loaded models, and catalog dimensions."""
    load_artifacts()
    return {
        "status": "healthy",
        "service": "E-Commerce Product Recommendation Engine",
        "loaded_models": list(_MODELS.keys()),
        "catalog_size": len(_CATALOG_DF) if _CATALOG_DF is not None else 0,
        "total_users": len(_USERS_DF) if _USERS_DF is not None else 0,
        "decision_support_role": "Advisory / Discovery Support (Human Oversight Recommended)"
    }


@app.get("/catalog", summary="Search and Browse Catalog Items")
def get_catalog(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search keyword in title/tags"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Retrieve catalog items with optional filtering and search."""
    load_artifacts()
    df = _CATALOG_DF.copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if search:
        s = search.lower()
        df = df[df["title"].str.lower().str.contains(s) | df["tags"].str.lower().str.contains(s)]

    total_matched = len(df)
    items = df.iloc[offset:offset + limit].to_dict(orient="records")
    return {
        "total_matched": total_matched,
        "offset": offset,
        "limit": limit,
        "items": items
    }


@app.get("/users/sample", summary="Sample Users for Testing")
def get_sample_users():
    """Returns sample warm and cold-start users for testing inference."""
    load_artifacts()
    cold = _USERS_DF[_USERS_DF["activity_tier"] == "casual"].head(5).to_dict(orient="records")
    warm = _USERS_DF[_USERS_DF["activity_tier"] == "power"].head(5).to_dict(orient="records")
    return {
        "warm_users": warm,
        "cold_start_users": cold
    }


@app.post("/recommend", response_model=RecommendResponse, summary="Generate Top-N Product Recommendations")
def recommend_products(req: RecommendRequest):
    """
    Generates personalized recommendations for a user using the selected model.
    Falls back gracefully if the user is in cold-start regime.
    """
    load_artifacts()
    model_key = req.model_type.lower()
    if model_key not in _MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type '{req.model_type}'. Choose from {list(_MODELS.keys())}"
        )

    model = _MODELS[model_key]

    try:
        raw_recs = model.recommend(
            user_id=req.user_id,
            top_k=req.top_k,
            exclude_interacted=req.exclude_interacted,
            category_filter=req.category_filter
        )

        if not raw_recs and model_key == "collaborative":
            raw_recs = _MODELS["popularity"].recommend(
                user_id=req.user_id,
                top_k=req.top_k,
                exclude_interacted=req.exclude_interacted,
                category_filter=req.category_filter
            )
            for r in raw_recs:
                r["signal"] = "Cold-start fallback to popularity (User not in CF training matrix)"
                r["is_cold_fallback"] = True

        recs = [ProductRecommendation(**r) for r in raw_recs]

        return RecommendResponse(
            user_id=req.user_id,
            model_type=model.name,
            top_k=req.top_k,
            recommendations=recs,
            total_returned=len(recs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/explain", summary="Explain Recommendation and Inspect Feature Signals")
def explain_recommendation(req: ExplainRequest):
    """
    Returns an explainable breakdown of a user-product recommendation:
    constituent sub-scores, keyword matches, latent co-interaction drivers,
    and decision-support limitations.
    """
    load_artifacts()
    model_key = req.model_type.lower()
    if model_key not in _MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Choose from {list(_MODELS.keys())}")

    model = _MODELS[model_key]
    explanation = model.explain(user_id=req.user_id, product_id=req.product_id)
    return explanation


@app.get("/metrics", summary="Offline Test Evaluation Benchmark Metrics")
def get_evaluation_metrics():
    """Returns the held-out chronological test benchmark metrics for all models."""
    load_artifacts()
    if _EVAL_METRICS is None:
        raise HTTPException(status_code=404, detail="Evaluation metrics file not found. Run src/evaluate.py first.")
    return _EVAL_METRICS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
