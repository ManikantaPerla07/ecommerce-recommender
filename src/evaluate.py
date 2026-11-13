"""
Defensible Test Set Evaluation Pipeline.
Evaluates all four models on the strictly held-out chronological test set.
Computes Precision@K, Recall@K, NDCG@K, MAP@K, Catalog Coverage, and Novelty.
Performs segmented evaluation for Overall vs Cold-Start Users.
Saves metrics artifact to artifacts/evaluation_metrics.json.
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from src.config import (
    PRODUCTS_FILE, USERS_FILE, TRAIN_INTERACTIONS_FILE, TEST_INTERACTIONS_FILE,
    POPULARITY_MODEL_FILE, COLLABORATIVE_MODEL_FILE, CONTENT_MODEL_FILE,
    HYBRID_MODEL_FILE, METRICS_EVAL_FILE, EVAL_K_LIST
)
from src.metrics import evaluate_recommendations_batch


def run_evaluation() -> Dict[str, Any]:
    print("=" * 75)
    print("HELD-OUT CHRONOLOGICAL TEST SET EVALUATION")
    print("=" * 75)

    # 1. Load data
    df_products = pd.read_csv(PRODUCTS_FILE)
    df_train = pd.read_csv(TRAIN_INTERACTIONS_FILE)
    df_test = pd.read_csv(TEST_INTERACTIONS_FILE)

    catalog_items = set(df_products["product_id"])
    item_train_counts = df_train.groupby("product_id").size().to_dict()
    total_train_events = len(df_train)

    # 2. Load model artifacts
    models = {
        "Popularity Baseline": joblib.load(POPULARITY_MODEL_FILE),
        "Collaborative Filtering (SVD)": joblib.load(COLLABORATIVE_MODEL_FILE),
        "Content-Based (TF-IDF)": joblib.load(CONTENT_MODEL_FILE),
        "Hybrid Recommender": joblib.load(HYBRID_MODEL_FILE)
    }

    # 3. Partition test ground truth
    # Overall test users
    ground_truth_overall = df_test.groupby("user_id")["product_id"].apply(set).to_dict()

    # Cold-start test users (<= 2 interactions in train)
    user_train_counts = df_train.groupby("user_id").size().to_dict()
    cold_users = [u for u in ground_truth_overall if user_train_counts.get(u, 0) <= 2]
    ground_truth_cold = {u: ground_truth_overall[u] for u in cold_users}

    # Warm users (> 2 interactions in train)
    warm_users = [u for u in ground_truth_overall if user_train_counts.get(u, 0) > 2]
    ground_truth_warm = {u: ground_truth_overall[u] for u in warm_users}

    print(f"Total Test Users:       {len(ground_truth_overall)}")
    print(f"  - Warm Test Users:    {len(ground_truth_warm)}")
    print(f"  - Cold Test Users:    {len(ground_truth_cold)}")
    print(f"Catalog Total Items:    {len(catalog_items)}")
    print("-" * 75)

    eval_results = {
        "metadata": {
            "test_events": len(df_test),
            "num_test_users": len(ground_truth_overall),
            "num_warm_users": len(ground_truth_warm),
            "num_cold_users": len(ground_truth_cold),
            "catalog_size": len(catalog_items),
            "k_values": EVAL_K_LIST
        },
        "overall": {},
        "cold_start": {},
        "warm": {}
    }

    # 4. Generate recommendations and evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        # Precompute recommendations for all test users up to max K (20)
        recs_map = {}
        for u in ground_truth_overall.keys():
            recs = [r["product_id"] for r in model.recommend(u, top_k=max(EVAL_K_LIST), exclude_interacted=True)]
            # If model returned fewer than max K (e.g. cold user in CF), fallback or pad with popularity
            recs_map[u] = recs

        # Evaluate Overall
        overall_metrics = evaluate_recommendations_batch(
            recommendations_map=recs_map,
            ground_truth_map=ground_truth_overall,
            catalog_items=catalog_items,
            item_train_counts=item_train_counts,
            total_train_events=total_train_events,
            k_list=EVAL_K_LIST
        )
        eval_results["overall"][model_name] = overall_metrics

        # Evaluate Cold Start Users Slice
        cold_metrics = evaluate_recommendations_batch(
            recommendations_map={u: recs_map[u] for u in cold_users if u in recs_map},
            ground_truth_map=ground_truth_cold,
            catalog_items=catalog_items,
            item_train_counts=item_train_counts,
            total_train_events=total_train_events,
            k_list=EVAL_K_LIST
        )
        eval_results["cold_start"][model_name] = cold_metrics

        # Evaluate Warm Users Slice
        warm_metrics = evaluate_recommendations_batch(
            recommendations_map={u: recs_map[u] for u in warm_users if u in recs_map},
            ground_truth_map=ground_truth_warm,
            catalog_items=catalog_items,
            item_train_counts=item_train_counts,
            total_train_events=total_train_events,
            k_list=EVAL_K_LIST
        )
        eval_results["warm"][model_name] = warm_metrics

    # Save to disk
    with open(METRICS_EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[OK] Metrics saved to {METRICS_EVAL_FILE}")

    # 5. Print Formatted Benchmark Table at K=10
    print("\n" + "=" * 95)
    print(f"{'OVERALL TEST SET BENCHMARK SUMMARY (K = 10)':^95}")
    print("=" * 95)
    header = f"{'Model':<30} | {'Prec@10':<9} | {'Recall@10':<10} | {'NDCG@10':<9} | {'MAP@10':<8} | {'Coverage':<10} | {'Novelty':<8}"
    print(header)
    print("-" * 95)

    for model_name in models.keys():
        m = eval_results["overall"][model_name]["K=10"]
        print(f"{model_name:<30} | {m['Precision@K']:<9.4f} | {m['Recall@K']:<10.4f} | {m['NDCG@K']:<9.4f} | {m['MAP@K']:<8.4f} | {m['CatalogCoverage@K']*100:<9.1f}% | {m['Novelty@K']:<8.2f}")

    print("=" * 95)

    print("\n" + "=" * 95)
    print(f"{'COLD-START USERS BENCHMARK SUMMARY (K = 10)':^95}")
    print("=" * 95)
    print(header)
    print("-" * 95)

    for model_name in models.keys():
        m = eval_results["cold_start"][model_name]["K=10"]
        print(f"{model_name:<30} | {m['Precision@K']:<9.4f} | {m['Recall@K']:<10.4f} | {m['NDCG@K']:<9.4f} | {m['MAP@K']:<8.4f} | {m['CatalogCoverage@K']*100:<9.1f}% | {m['Novelty@K']:<8.2f}")

    print("=" * 95)

    return eval_results


if __name__ == "__main__":
    run_evaluation()
