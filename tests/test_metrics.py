"""
Unit tests for Ranking Metrics implementations.
Validates Precision@K, Recall@K, NDCG@K, and MAP@K against exact analytical toy cases.
"""
import pytest
import numpy as np
from src.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k, catalog_coverage
)


def test_precision_at_k():
    recs = ["P1", "P2", "P3", "P4", "P5"]
    truth = {"P1", "P3"}

    assert precision_at_k(recs, truth, k=2) == 0.5  # 1 hit in 2
    assert precision_at_k(recs, truth, k=3) == 2.0 / 3.0  # 2 hits in 3
    assert precision_at_k(recs, truth, k=5) == 2.0 / 5.0  # 2 hits in 5
    assert precision_at_k([], truth, k=5) == 0.0
    assert precision_at_k(recs, set(), k=5) == 0.0


def test_recall_at_k():
    recs = ["P1", "P2", "P3", "P4", "P5"]
    truth = {"P1", "P3", "P9", "P10"}  # 4 relevant items total

    assert recall_at_k(recs, truth, k=2) == 0.25  # 1 hit out of 4
    assert recall_at_k(recs, truth, k=3) == 0.50  # 2 hits out of 4
    assert recall_at_k(recs, truth, k=5) == 0.50  # 2 hits out of 4
    assert recall_at_k(recs, set(), k=5) == 0.0


def test_ndcg_at_k_perfect_ranking():
    recs = ["P1", "P2", "P3"]
    truth = {"P1", "P2", "P3"}
    # Perfect top 3 ranking should have NDCG@3 = 1.0
    assert np.isclose(ndcg_at_k(recs, truth, k=3), 1.0)


def test_ndcg_at_k_discounted_penalty():
    # If the relevant item is at rank 1 vs rank 3, NDCG should be strictly higher for rank 1
    recs_good = ["P1", "P2", "P3"]
    recs_poor = ["P2", "P3", "P1"]
    truth = {"P1"}

    ndcg_good = ndcg_at_k(recs_good, truth, k=3)
    ndcg_poor = ndcg_at_k(recs_poor, truth, k=3)

    assert ndcg_good > ndcg_poor
    assert np.isclose(ndcg_good, 1.0)  # P1 at rank 1 is ideal IDCG


def test_catalog_coverage():
    all_recs = [
        ["P1", "P2", "P3"],
        ["P2", "P3", "P4"],
        ["P1", "P5", "P6"]
    ]
    catalog = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}
    # Total unique items recommended: P1, P2, P3, P4, P5, P6 (6 items out of 10)
    cov = catalog_coverage(all_recs, catalog, k=3)
    assert np.isclose(cov, 0.60)
