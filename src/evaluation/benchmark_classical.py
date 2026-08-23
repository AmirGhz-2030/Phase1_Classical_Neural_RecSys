"""
Benchmark Evaluation Script for Classical Baselines
===================================================
Trains Popularity, Item-KNN, and Matrix Factorization (SVD) on MovieLens 20M Train split
and evaluates ranking performance (HR@5, HR@10, HR@20, NDCG@10, MRR) on the Test split.
"""

import os
import sys
import pickle
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Safe console encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models.classical import (
    PopularityRecommender,
    ItemKNNRecommender,
    MatrixFactorizationRecommender
)
from src.evaluation.metrics import RankingEvaluator

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def run_classical_benchmark(sample_eval_users: int = 5000):
    print("=" * 70)
    print("🚀 RUNNING CLASSICAL RECOMMENDATION BENCHMARK (PHASE 1)")
    print("=" * 70)

    # 1. Load Data
    print("Loading processed datasets...")
    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    n_users = meta["stats"]["n_users"]
    n_items = meta["stats"]["n_items"]

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(f"Loaded Train: {len(train_df):,} | Test: {len(test_df):,} | Users: {n_users:,} | Items: {n_items:,}")

    # Build ground truth mapping for test set
    test_gt = test_df.set_index("user_idx")["item_idx"].to_dict()

    # Sample evaluation users for fast and statistically significant benchmarking
    np.random.seed(42)
    eval_users = list(np.random.choice(list(test_gt.keys()), size=min(sample_eval_users, len(test_gt)), replace=False))
    eval_gt = {u: test_gt[u] for u in eval_users}
    print(f"Evaluating on {len(eval_users):,} randomly sampled active users (Seed=42).")

    evaluator = RankingEvaluator(k_list=[5, 10, 20])
    results_summary = {}

    # Define models
    models = [
        PopularityRecommender(),
        MatrixFactorizationRecommender(n_factors=64),
        ItemKNNRecommender()
    ]

    for model in models:
        print("\n" + "-" * 50)
        print(f"📌 Training & Evaluating: {model.name}")
        t0 = time.time()
        model.fit(train_df, n_users=n_users, n_items=n_items)
        fit_time = time.time() - t0
        print(f"   -> Fitted in {fit_time:.2f} seconds.")

        # Generate recommendations
        t_rec = time.time()
        print(f"   -> Generating Top-20 recommendations for {len(eval_users):,} test users...")
        recs = model.recommend_batch(eval_users, top_k=20, filter_history=True)
        rec_time = time.time() - t_rec
        print(f"   -> Inferred in {rec_time:.2f} seconds ({len(eval_users)/rec_time:.1f} users/sec).")

        # Evaluate
        metrics = evaluator.evaluate_recommendations(recs, eval_gt)
        metrics["Fit_Time_s"] = round(fit_time, 2)
        metrics["Inf_Time_s"] = round(rec_time, 2)
        results_summary[model.name] = metrics

        print(f"   📊 Results: HR@10 = {metrics['HR@10']:.4f} | NDCG@10 = {metrics['NDCG@10']:.4f} | MRR@10 = {metrics['MRR@10']:.4f}")

    # Output Final Comparison Table
    print("\n" + "=" * 70)
    print("🏆 FINAL BENCHMARK COMPARISON TABLE (PHASE 1 CLASSICAL MODELS)")
    print("=" * 70)
    comparison_df = pd.DataFrame(results_summary).T
    display_cols = ["HR@5", "HR@10", "HR@20", "NDCG@10", "MRR@10", "Fit_Time_s"]
    print(comparison_df[display_cols].to_string())

    # Save benchmark table to processed dir
    out_csv = os.path.join(PROCESSED_DIR, "classical_benchmark_results.csv")
    comparison_df.to_csv(out_csv)
    print(f"\nSaved benchmark results to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    run_classical_benchmark(sample_eval_users=5000)
