"""
Model Training and Validation Pipeline.
Fits all candidate models on the chronological training set,
tunes ensemble hyperparameters on the validation set,
and serializes artifacts into the artifacts/ directory.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    PRODUCTS_FILE, USERS_FILE, TRAIN_INTERACTIONS_FILE, VAL_INTERACTIONS_FILE,
    POPULARITY_MODEL_FILE, COLLABORATIVE_MODEL_FILE, CONTENT_MODEL_FILE,
    HYBRID_MODEL_FILE, CF_LATENT_FACTORS, TFIDF_MAX_FEATURES
)
from src.models import (
    PopularityRecommender,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    HybridRecommender
)
from src.metrics import evaluate_recommendations_batch


def run_training():
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df_products = pd.read_csv(PRODUCTS_FILE)
    df_users = pd.read_csv(USERS_FILE)
    df_train = pd.read_csv(TRAIN_INTERACTIONS_FILE)
    df_val = pd.read_csv(VAL_INTERACTIONS_FILE)

    print(f"Loaded {len(df_train)} training interactions and {len(df_val)} validation interactions.")

    # 2. Train Popularity Baseline
    print("\n[1/4] Training Popularity Baseline Recommender...")
    popularity_model = PopularityRecommender()
    popularity_model.fit(df_train, df_products, df_users)
    joblib.dump(popularity_model, POPULARITY_MODEL_FILE)
    print(f"      Saved -> {POPULARITY_MODEL_FILE}")

    # 3. Train Collaborative Filtering (SVD)
    print("\n[2/4] Training Collaborative Filtering Recommender (SVD)...")
    cf_model = CollaborativeFilteringRecommender(n_factors=CF_LATENT_FACTORS)
    cf_model.fit(df_train, df_products, df_users)
    joblib.dump(cf_model, COLLABORATIVE_MODEL_FILE)
    print(f"      Saved -> {COLLABORATIVE_MODEL_FILE}")

    # 4. Train Content-Based Recommender (TF-IDF)
    print("\n[3/4] Training Content-Based Recommender (TF-IDF)...")
    content_model = ContentBasedRecommender(max_features=TFIDF_MAX_FEATURES)
    content_model.fit(df_train, df_products, df_users)
    joblib.dump(content_model, CONTENT_MODEL_FILE)
    print(f"      Saved -> {CONTENT_MODEL_FILE}")

    # 5. Tune and Train Hybrid Recommender on Validation set
    print("\n[4/4] Optimizing Hybrid Recommender Weights on Validation Set...")
    catalog_items = set(df_products["product_id"])
    item_train_counts = df_train.groupby("product_id").size().to_dict()
    total_train_events = len(df_train)

    # Ground truth interactions in validation set (only relevant interactions like purchases/carts/views)
    val_ground_truth = df_val.groupby("user_id")["product_id"].apply(set).to_dict()

    # Candidate weight combinations (CF, Content, Popularity)
    candidate_weights = [
        {"collaborative": 0.50, "content": 0.35, "popularity": 0.15},
        {"collaborative": 0.60, "content": 0.25, "popularity": 0.15},
        {"collaborative": 0.40, "content": 0.40, "popularity": 0.20},
        {"collaborative": 0.30, "content": 0.50, "popularity": 0.20}
    ]

    best_ndcg = -1.0
    best_weights = candidate_weights[0]

    # Quick validation grid evaluation over a representative sample of validation users
    val_users_sample = list(val_ground_truth.keys())[:150]
    sub_gt = {u: val_ground_truth[u] for u in val_users_sample}

    for weights in candidate_weights:
        hybrid_temp = HybridRecommender(
            popularity_model=popularity_model,
            cf_model=cf_model,
            content_model=content_model,
            weights=weights
        )
        hybrid_temp.fit(df_train, df_products, df_users)

        recs_map = {}
        for u in val_users_sample:
            recs = [r["product_id"] for r in hybrid_temp.recommend(u, top_k=10, exclude_interacted=True)]
            recs_map[u] = recs

        val_metrics = evaluate_recommendations_batch(
            recommendations_map=recs_map,
            ground_truth_map=sub_gt,
            catalog_items=catalog_items,
            item_train_counts=item_train_counts,
            total_train_events=total_train_events,
            k_list=[10]
        )
        ndcg_val = val_metrics["K=10"]["NDCG@K"]
        print(f"      Candidate weights {weights} -> Val NDCG@10: {ndcg_val:.4f}")

        if ndcg_val > best_ndcg:
            best_ndcg = ndcg_val
            best_weights = weights

    print(f"      Best weights chosen: {best_weights} (Val NDCG@10: {best_ndcg:.4f})")

    # Fit final hybrid model with optimal weights
    final_hybrid = HybridRecommender(
        popularity_model=popularity_model,
        cf_model=cf_model,
        content_model=content_model,
        weights=best_weights
    )
    final_hybrid.fit(df_train, df_products, df_users)
    joblib.dump(final_hybrid, HYBRID_MODEL_FILE)
    print(f"      Saved -> {HYBRID_MODEL_FILE}")

    print("=" * 60)
    print("TRAINING PIPELINE COMPLETE: ALL ARTIFACTS SERIALIZED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
