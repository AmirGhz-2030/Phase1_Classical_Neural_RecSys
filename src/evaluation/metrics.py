"""
Top-K Ranking Evaluation Engine for Recommender Systems
======================================================
Implements standardized academic ranking metrics:
- Hit Rate (HR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR@K / MRR)
- Precision@K
- Recall@K

Supports:
1. Single-user item list evaluation (Leave-One-Out / Multi-Item).
2. Batch evaluation across all users with multi-K support (e.g. K=[5, 10, 20]).
3. Formatted benchmark summary generation for logging and Streamlit reporting.

Author: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import math
from typing import Dict, List, Set, Union, Optional
import numpy as np
import pandas as pd


def hit_rate_at_k(
    recommendations: List[int],
    ground_truth: Union[int, List[int], Set[int]],
    k: int = 10
) -> float:
    """
    Compute Hit Rate @ K (HR@K).
    Returns 1.0 if any ground-truth item is in top-K recommendations, else 0.0.
    """
    if k <= 0 or not recommendations:
        return 0.0
    top_k = recommendations[:k]
    if isinstance(ground_truth, (int, np.integer)):
        return 1.0 if ground_truth in top_k else 0.0
    gt_set = set(ground_truth)
    return 1.0 if any(item in gt_set for item in top_k) else 0.0


def ndcg_at_k(
    recommendations: List[int],
    ground_truth: Union[int, List[int], Set[int]],
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K (NDCG@K).
    For single positive ground truth: 1.0 / log2(rank + 1) where rank is 1-indexed.
    For multiple items: DCG@K / IDCG@K with item deduplication safeguards.
    """
    if k <= 0 or not recommendations:
        return 0.0
    top_k = recommendations[:k]
    if isinstance(ground_truth, (int, np.integer)):
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1  # 1-indexed rank
            return 1.0 / math.log2(rank + 1)
        return 0.0

    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    dcg = 0.0
    counted_items = set()
    for i, item in enumerate(top_k):
        if item in gt_set and item not in counted_items:
            counted_items.add(item)
            rank = i + 1
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(gt_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return min(1.0, dcg / idcg)


def mrr_at_k(
    recommendations: List[int],
    ground_truth: Union[int, List[int], Set[int]],
    k: int = 10
) -> float:
    """
    Compute Mean Reciprocal Rank @ K (MRR@K).
    Returns 1.0 / rank of the first relevant item in top-K, else 0.0.
    """
    if k <= 0 or not recommendations:
        return 0.0
    top_k = recommendations[:k]
    if isinstance(ground_truth, (int, np.integer)):
        gt_set = {ground_truth}
    else:
        gt_set = set(ground_truth)

    for i, item in enumerate(top_k):
        if item in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    recommendations: List[int],
    ground_truth: Union[int, List[int], Set[int]],
    k: int = 10
) -> float:
    """
    Compute Precision @ K: (Number of distinct relevant items in top-K) / K.
    """
    if k <= 0 or not recommendations:
        return 0.0
    top_k = recommendations[:k]
    if isinstance(ground_truth, (int, np.integer)):
        gt_set = {ground_truth}
    else:
        gt_set = set(ground_truth)

    distinct_hits = len(set(top_k) & gt_set)
    return min(1.0, distinct_hits / float(k))


def recall_at_k(
    recommendations: List[int],
    ground_truth: Union[int, List[int], Set[int]],
    k: int = 10
) -> float:
    """
    Compute Recall @ K: (Number of distinct relevant items in top-K) / (Total relevant items).
    """
    if k <= 0 or not recommendations:
        return 0.0
    if isinstance(ground_truth, (int, np.integer)):
        gt_set = {ground_truth}
    else:
        gt_set = set(ground_truth)

    if not gt_set:
        return 0.0

    top_k = recommendations[:k]
    distinct_hits = len(set(top_k) & gt_set)
    return min(1.0, distinct_hits / float(len(gt_set)))


class RankingEvaluator:
    """
    Batch Evaluation Manager for Recommender Systems.
    Evaluates model predictions against test ground-truth interactions.
    """

    def __init__(self, k_list: Optional[List[int]] = None):
        self.k_list = k_list or [5, 10, 20]

    def evaluate_recommendations(
        self,
        recommendations_dict: Dict[int, List[int]],
        ground_truth_dict: Dict[int, Union[int, List[int], Set[int]]]
    ) -> Dict[str, float]:
        """
        Evaluate a complete dictionary of {user_id: [ranked_item_ids]}.

        Args:
            recommendations_dict: Mapping user_idx to list of ranked recommended item_idxs.
            ground_truth_dict: Mapping user_idx to target item_idx (or list/set of targets).

        Returns:
            Dictionary of averaged metric values (e.g. {'HR@5': 0.12, 'NDCG@10': 0.08, ...}).
        """
        eval_users = [u for u in ground_truth_dict if u in recommendations_dict]
        n_users = len(eval_users)

        if n_users == 0:
            return {f"{m}@{k}": 0.0 for k in self.k_list for m in ["HR", "NDCG", "MRR", "Precision", "Recall"]}

        metrics_sum: Dict[str, float] = {}

        # Initialize metric accumulators
        for k in self.k_list:
            metrics_sum[f"HR@{k}"] = 0.0
            metrics_sum[f"NDCG@{k}"] = 0.0
            metrics_sum[f"MRR@{k}"] = 0.0
            metrics_sum[f"Precision@{k}"] = 0.0
            metrics_sum[f"Recall@{k}"] = 0.0

        for u in eval_users:
            recs = recommendations_dict[u]
            gt = ground_truth_dict[u]

            for k in self.k_list:
                metrics_sum[f"HR@{k}"] += hit_rate_at_k(recs, gt, k=k)
                metrics_sum[f"NDCG@{k}"] += ndcg_at_k(recs, gt, k=k)
                metrics_sum[f"MRR@{k}"] += mrr_at_k(recs, gt, k=k)
                metrics_sum[f"Precision@{k}"] += precision_at_k(recs, gt, k=k)
                metrics_sum[f"Recall@{k}"] += recall_at_k(recs, gt, k=k)

        # Compute averages
        results = {metric: val / float(n_users) for metric, val in metrics_sum.items()}
        return results

    def create_results_dataframe(self, all_models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Convert multiple model metric dictionaries into a clean comparison DataFrame.
        """
        df = pd.DataFrame(all_models_results).T
        return df
