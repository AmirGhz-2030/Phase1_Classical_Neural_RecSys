"""
Sampled 100-Item Ranking Benchmark (NCF He et al., 2017 Protocol)
=================================================================
Evaluates models using the 100-item candidate pool protocol:
- For each test user: 1 positive target item + 99 randomly sampled unobserved negative items.
- Ranks only among these 100 candidates.
- Computes HR@5, HR@10, NDCG@10, MRR@10.

Author: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from src.models.classical import (
    PopularityRecommender,
    ItemKNNRecommender,
    MatrixFactorizationRecommender
)
from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k

PROCESSED_DIR = r"G:\Phase1_Classical_Neural_RecSys\data\processed"


def generate_sampled_100_candidates(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_items: int,
    n_negatives: int = 99,
    seed: int = 42
) -> dict:
    """
    Builds candidate list of [target_item, neg_1, neg_2, ..., neg_99] for each test user.
    """
    np.random.seed(seed)
    train_history = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_dict = test_df.set_index("user_idx")["item_idx"].to_dict()

    user_candidates = {}
    for u, target in test_dict.items():
        seen = train_history.get(u, set()) | {target}
        negs = []
        while len(negs) < n_negatives:
            cand = np.random.randint(0, n_items)
            if cand not in seen and cand not in negs:
                negs.append(cand)
        # 100 items total: target item + 99 negatives
        user_candidates[u] = [target] + negs

    return user_candidates, test_dict


def evaluate_sampled_100(
    model_name: str,
    score_fn,
    user_candidates: dict,
    test_dict: dict,
    eval_users: list,
    k_list: list = [5, 10, 20]
) -> dict:
    """
    Scores and ranks the 100 candidate items for each user.
    """
    metrics = {f"HR@{k}": 0.0 for k in k_list}
    metrics["NDCG@10"] = 0.0
    metrics["MRR@10"] = 0.0

    n_users = len(eval_users)
    for u in eval_users:
        target = test_dict[u]
        candidates = user_candidates[u]  # 100 items
        scores = score_fn(u, candidates)

        # Rank candidates by predicted scores descending
        ranked_indices = np.argsort(-np.array(scores))
        ranked_items = [candidates[i] for i in ranked_indices]

        for k in k_list:
            metrics[f"HR@{k}"] += hit_rate_at_k(ranked_items, target, k=k)
        metrics["NDCG@10"] += ndcg_at_k(ranked_items, target, k=10)
        metrics["MRR@10"] += mrr_at_k(ranked_items, target, k=10)

    # Average
    return {m: val / float(n_users) for m, val in metrics.items()}


def run_sampled_benchmark(sample_eval_users: int = 5000):
    print("=" * 70)
    print("🎯 RUNNING SAMPLED 100-ITEM RANKING BENCHMARK (NCF 2017 PROTOCOL)")
    print("=" * 70)

    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    n_users = meta["stats"]["n_users"]
    n_items = meta["stats"]["n_items"]

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    print(f"Generating 100 candidates (1 Positive + 99 Negatives) for {sample_eval_users:,} users...")
    user_candidates, test_dict = generate_sampled_100_candidates(test_df, train_df, n_items=n_items)

    np.random.seed(42)
    all_test_users = list(test_dict.keys())
    eval_users = list(np.random.choice(all_test_users, size=min(sample_eval_users, len(all_test_users)), replace=False))

    results = {}

    # 1. Popularity
    print("\n--- Evaluating Popularity ---")
    pop_model = PopularityRecommender()
    pop_model.fit(train_df, n_users=n_users, n_items=n_items)
    pop_counts = train_df["item_idx"].value_counts().to_dict()

    def score_pop(u, cands):
        return [pop_counts.get(c, 0) for c in cands]

    results["Popularity"] = evaluate_sampled_100("Popularity", score_pop, user_candidates, test_dict, eval_users)
    print(f"Popularity: HR@10 = {results['Popularity']['HR@10']:.4f} ({results['Popularity']['HR@10']*100:.2f}%) | NDCG@10 = {results['Popularity']['NDCG@10']:.4f}")

    # 2. Matrix Factorization (SVD)
    print("\n--- Evaluating Matrix Factorization (SVD) ---")
    svd_model = MatrixFactorizationRecommender(n_factors=64)
    svd_model.fit(train_df, n_users=n_users, n_items=n_items)

    def score_svd(u, cands):
        u_vec = svd_model.user_factors[u]
        cand_factors = svd_model.item_factors[cands]
        return cand_factors @ u_vec

    results["Matrix Factorization (SVD)"] = evaluate_sampled_100("SVD", score_svd, user_candidates, test_dict, eval_users)
    print(f"SVD:        HR@10 = {results['Matrix Factorization (SVD)']['HR@10']:.4f} ({results['Matrix Factorization (SVD)']['HR@10']*100:.2f}%) | NDCG@10 = {results['Matrix Factorization (SVD)']['NDCG@10']:.4f}")

    # 3. Item-KNN
    print("\n--- Evaluating Item-KNN ---")
    knn_model = ItemKNNRecommender()
    knn_model.fit(train_df, n_users=n_users, n_items=n_items)

    def score_knn(u, cands):
        seen = knn_model.user_history.get(u, set())
        if not seen:
            return [0.0] * len(cands)
        seen_items = np.array(list(seen), dtype=np.int32)
        user_items_mat = knn_model.normalized_item_matrix[seen_items]
        user_rep = user_items_mat.sum(axis=0)
        cand_mat = knn_model.normalized_item_matrix[cands]
        return (cand_mat @ user_rep.T).A1

    results["Item-KNN (CF)"] = evaluate_sampled_100("Item-KNN", score_knn, user_candidates, test_dict, eval_users)
    print(f"Item-KNN:   HR@10 = {results['Item-KNN (CF)']['HR@10']:.4f} ({results['Item-KNN (CF)']['HR@10']*100:.2f}%) | NDCG@10 = {results['Item-KNN (CF)']['NDCG@10']:.4f}")

    # Final Table
    print("\n" + "=" * 70)
    print("🏆 FINAL COMPARISON: SAMPLED 100-ITEM RANKING (NCF PROTOCOL)")
    print("=" * 70)
    df_res = pd.DataFrame(results).T
    print(df_res[["HR@5", "HR@10", "HR@20", "NDCG@10", "MRR@10"]].to_string())

    out_csv = os.path.join(PROCESSED_DIR, "sampled100_benchmark_results.csv")
    df_res.to_csv(out_csv)
    print(f"\nSaved sampled benchmark to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_sampled_benchmark(sample_eval_users=5000)
