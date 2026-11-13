"""
Ranking and Recommendation Evaluation Metrics Module.
Implements standard industry metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- Catalog Coverage
- Novelty (Self-Information)
"""
import numpy as np
from typing import List, Dict, Set, Union


def precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Compute Precision@K for a single user."""
    if k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return float(hits) / float(k)


def recall_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Compute Recall@K for a single user."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return float(hits) / float(len(ground_truth))


def ndcg_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K) for a single user."""
    if not ground_truth or k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 2)  # idx+2 because rank index is 0-based -> log2(rank + 1)

    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(ground_truth))))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def map_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Compute Average Precision at K (AP@K) for a single user."""
    if not ground_truth or k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    score = 0.0
    num_hits = 0.0
    for idx, item in enumerate(rec_k):
        if item in ground_truth:
            num_hits += 1.0
            score += num_hits / float(idx + 1)
    ideal_hits = min(k, len(ground_truth))
    return float(score / ideal_hits) if ideal_hits > 0 else 0.0


def catalog_coverage(all_recommendations: List[List[str]], catalog_items: Set[str], k: int) -> float:
    """Compute Catalog Coverage@K: fraction of catalog items recommended at least once."""
    if not catalog_items:
        return 0.0
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs[:k])
    return float(len(recommended_items & catalog_items)) / float(len(catalog_items))


def novelty_at_k(all_recommendations: List[List[str]], item_train_counts: Dict[str, int], total_train_events: int, k: int) -> float:
    """
    Compute Novelty@K (Mean Self-Information):
    Novelty = -log2(P(item)). Higher means surfacing less mainstream, more diverse/novel items.
    """
    if total_train_events <= 0 or not all_recommendations:
        return 0.0
    novelties = []
    for recs in all_recommendations:
        user_novelties = []
        for item in recs[:k]:
            count = item_train_counts.get(item, 1)  # laplace smoothing
            prob = count / float(total_train_events)
            user_novelties.append(-np.log2(prob))
        if user_novelties:
            novelties.append(np.mean(user_novelties))
    return float(np.mean(novelties)) if novelties else 0.0


def evaluate_recommendations_batch(
    recommendations_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, Set[str]],
    catalog_items: Set[str],
    item_train_counts: Dict[str, int],
    total_train_events: int,
    k_list: List[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a batch of user recommendations against ground truth across multiple K values.
    Returns nested dictionary: {f"K={k}": {metric_name: value}}
    """
    if k_list is None:
        k_list = [5, 10, 20]

    eval_users = [u for u in ground_truth_map.keys() if len(ground_truth_map[u]) > 0 and u in recommendations_map]
    results = {}

    for k in k_list:
        precisions = []
        recalls = []
        ndcgs = []
        maps = []
        user_recs_list = []

        for u in eval_users:
            recs = recommendations_map[u]
            truth = ground_truth_map[u]
            user_recs_list.append(recs)

            precisions.append(precision_at_k(recs, truth, k))
            recalls.append(recall_at_k(recs, truth, k))
            ndcgs.append(ndcg_at_k(recs, truth, k))
            maps.append(map_at_k(recs, truth, k))

        cov = catalog_coverage(user_recs_list, catalog_items, k)
        nov = novelty_at_k(user_recs_list, item_train_counts, total_train_events, k)

        results[f"K={k}"] = {
            "Precision@K": round(float(np.mean(precisions)) if precisions else 0.0, 4),
            "Recall@K": round(float(np.mean(recalls)) if recalls else 0.0, 4),
            "NDCG@K": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
            "MAP@K": round(float(np.mean(maps)) if maps else 0.0, 4),
            "CatalogCoverage@K": round(float(cov), 4),
            "Novelty@K": round(float(nov), 4),
            "EvaluatedUsers": len(eval_users)
        }

    return results
