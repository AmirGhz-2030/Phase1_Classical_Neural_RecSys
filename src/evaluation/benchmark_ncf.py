"""
Benchmark Script for NCF (Neural Collaborative Filtering)
=========================================================
Trains NeuMF on MovieLens 20M and evaluates on the Test set.
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from src.models.ncf import NCFRecommender
from src.evaluation.metrics import RankingEvaluator

PROCESSED_DIR = r"G:\Phase1_Classical_Neural_RecSys\data\processed"

def run_ncf_benchmark(sample_eval_users: int = 5000, sample_ratio: float = 0.05, num_epochs: int = 3):
    print("=" * 70)
    print("🧠 RUNNING NEURAL COLLABORATIVE FILTERING (NCF) BENCHMARK")
    print("=" * 70)

    # 1. Load Data
    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    n_users = meta["stats"]["n_users"]
    n_items = meta["stats"]["n_items"]

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    test_gt = test_df.set_index("user_idx")["item_idx"].to_dict()

    np.random.seed(42)
    eval_users = list(np.random.choice(list(test_gt.keys()), size=min(sample_eval_users, len(test_gt)), replace=False))
    eval_gt = {u: test_gt[u] for u in eval_users}

    evaluator = RankingEvaluator(k_list=[5, 10, 20])

    # 2. Train NCF
    ncf = NCFRecommender(
        latent_dim_gmf=32,
        latent_dim_mlp=32,
        mlp_layers=[64, 32, 16],
        lr=0.002,
        batch_size=2048,
        num_epochs=num_epochs,
        num_negatives=4
    )

    t0 = time.time()
    ncf.fit(train_df, n_users=n_users, n_items=n_items, sample_ratio=sample_ratio)
    fit_time = time.time() - t0
    print(f"   -> NCF Trained in {fit_time:.2f} seconds.")

    # 3. Evaluate NCF
    t_rec = time.time()
    print(f"   -> Generating Top-20 recommendations for {len(eval_users):,} test users...")
    recs = ncf.recommend_batch(eval_users, top_k=20, filter_history=True, batch_size=128)
    rec_time = time.time() - t_rec
    print(f"   -> Inferred in {rec_time:.2f} seconds.")

    metrics = evaluator.evaluate_recommendations(recs, eval_gt)
    metrics["Fit_Time_s"] = round(fit_time, 2)
    metrics["Inf_Time_s"] = round(rec_time, 2)

    print(f"\n📊 NCF Results:")
    print(f"   - HR@5:    {metrics['HR@5']:.4f}")
    print(f"   - HR@10:   {metrics['HR@10']:.4f}")
    print(f"   - HR@20:   {metrics['HR@20']:.4f}")
    print(f"   - NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"   - MRR@10:  {metrics['MRR@10']:.4f}")

    # Combine with Classical Results
    classical_csv = os.path.join(PROCESSED_DIR, "classical_benchmark_results.csv")
    if os.path.exists(classical_csv):
        df_classical = pd.read_csv(classical_csv, index_col=0)
        df_ncf = pd.DataFrame([metrics], index=["NCF (NeuMF)"])
        combined_df = pd.concat([df_classical, df_ncf])
        combined_csv = os.path.join(PROCESSED_DIR, "full_phase1_benchmark_results.csv")
        combined_df.to_csv(combined_csv)

        print("\n" + "=" * 70)
        print("🏆 COMBINED PHASE 1 BENCHMARK LEADERBOARD")
        print("=" * 70)
        display_cols = ["HR@5", "HR@10", "HR@20", "NDCG@10", "MRR@10", "Fit_Time_s"]
        print(combined_df[display_cols].to_string())

if __name__ == "__main__":
    run_ncf_benchmark(sample_eval_users=5000, sample_ratio=0.03, num_epochs=3)
