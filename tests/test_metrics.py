"""
Unit Tests for Ranking Evaluation Metrics
=========================================
Tests HR@K, NDCG@K, MRR@K, Precision@K, Recall@K against analytical edge cases.
"""

import os
import sys
import math

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Safe console encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    RankingEvaluator
)

def test_metrics_accuracy():
    print("Testing Ranking Evaluation Metrics...")

    # Case 1: Ground truth item at rank 1 (top position)
    recs = [101, 102, 103, 104, 105]
    gt = 101

    assert hit_rate_at_k(recs, gt, k=5) == 1.0
    assert ndcg_at_k(recs, gt, k=5) == 1.0  # 1 / log2(1 + 1) = 1.0
    assert mrr_at_k(recs, gt, k=5) == 1.0
    assert precision_at_k(recs, gt, k=5) == 1.0 / 5.0
    assert recall_at_k(recs, gt, k=5) == 1.0
    print("✅ Case 1 (Rank 1 Hit) Passed.")

    # Case 2: Ground truth item at rank 2
    gt2 = 102
    assert hit_rate_at_k(recs, gt2, k=5) == 1.0
    assert math.isclose(ndcg_at_k(recs, gt2, k=5), 1.0 / math.log2(2 + 1), rel_tol=1e-5)
    assert math.isclose(mrr_at_k(recs, gt2, k=5), 0.5)
    print("✅ Case 2 (Rank 2 Hit) Passed.")

    # Case 3: Ground truth item missing (Miss)
    gt_miss = 999
    assert hit_rate_at_k(recs, gt_miss, k=5) == 0.0
    assert ndcg_at_k(recs, gt_miss, k=5) == 0.0
    assert mrr_at_k(recs, gt_miss, k=5) == 0.0
    assert precision_at_k(recs, gt_miss, k=5) == 0.0
    assert recall_at_k(recs, gt_miss, k=5) == 0.0
    print("✅ Case 3 (Complete Miss) Passed.")

    # Case 4: Batch Evaluator Verification
    evaluator = RankingEvaluator(k_list=[5, 10])
    recs_dict = {
        0: [101, 102, 103, 104, 105], # user 0: rank 1 hit for 101
        1: [201, 202, 203, 204, 205], # user 1: rank 2 hit for 202
        2: [301, 302, 303, 304, 305], # user 2: miss for 999
    }
    gt_dict = {0: 101, 1: 202, 2: 999}

    res = evaluator.evaluate_recommendations(recs_dict, gt_dict)
    # Expected HR@5 = (1.0 + 1.0 + 0.0) / 3 = 2/3 = 0.6667
    assert math.isclose(res["HR@5"], 2.0 / 3.0, rel_tol=1e-4)
    # Expected MRR@5 = (1.0 + 0.5 + 0.0) / 3 = 1.5 / 3 = 0.5
    assert math.isclose(res["MRR@5"], 0.5, rel_tol=1e-4)
    print("✅ Case 4 (Batch Evaluator) Passed.")

    # Case 5: Edge cases & deduplication safeguards
    dup_recs = [101, 101, 102, 103, 104]
    gt_multi = [101, 102]
    assert ndcg_at_k(dup_recs, gt_multi, k=5) <= 1.0
    assert precision_at_k(dup_recs, gt_multi, k=5) == 2.0 / 5.0
    assert hit_rate_at_k([], 101, k=5) == 0.0
    assert ndcg_at_k([], 101, k=5) == 0.0
    assert mrr_at_k([], 101, k=5) == 0.0
    print("✅ Case 5 (Edge cases & Deduplication) Passed.")

    print("\n🎉 ALL METRIC TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_metrics_accuracy()
