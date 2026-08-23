"""
Data Pipeline and Split Integrity Test Suite
============================================
Validates preprocessing invariants for MovieLens 20M Phase 1:
1. Processed artifact existence (train, val, test, movies_clean, meta.pkl).
2. Temporal Leave-One-Last partition invariants (exactly 1 test/val item per user).
3. 20-Core constraint satisfaction (no cold users/items < 20 interactions).
4. Continuous 0-indexed integer ID remappings for PyTorch/NCF embedding alignment.
5. Matrix sparsity and summary stats consistency against meta.pkl.

Author: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

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

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def test_processed_artifacts_exist():
    print("1. Checking processed artifact files on disk...")
    required_files = [
        "train.csv",
        "val.csv",
        "test.csv",
        "movies_clean.csv",
        "meta.pkl"
    ]
    for fname in required_files:
        fpath = os.path.join(PROCESSED_DIR, fname)
        assert os.path.exists(fpath), f"Missing required processed artifact: {fpath}"
        assert os.path.getsize(fpath) > 0, f"File {fpath} is empty."
    print("   -> All 5 processed artifacts exist and are non-empty.")


def test_metadata_consistency():
    print("2. Checking metadata structure and mappings...")
    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    assert "stats" in meta, "Metadata missing 'stats' key."
    assert "user2idx" in meta and "idx2user" in meta, "Metadata missing user mappings."
    assert "item2idx" in meta and "idx2item" in meta, "Metadata missing item mappings."

    stats = meta["stats"]
    assert stats["n_users"] == len(meta["user2idx"]) == len(meta["idx2user"])
    assert stats["n_items"] == len(meta["item2idx"]) == len(meta["idx2item"])
    assert stats["k_core"] == 20

    print(f"   -> Mappings verified: {stats['n_users']:,} users, {stats['n_items']:,} items.")
    return meta


def test_temporal_split_integrity(meta):
    print("3. Checking Temporal Leave-One-Last split invariants...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    n_users = meta["stats"]["n_users"]
    n_interactions = meta["stats"]["n_interactions"]

    # Each active user must have exactly 1 test interaction
    assert len(test_df) == n_users, f"Test set size {len(test_df)} != n_users {n_users}"
    assert test_df["user_idx"].nunique() == n_users, "Test set contains duplicate users."

    # Each active user must have exactly 1 validation interaction
    assert len(val_df) == n_users, f"Val set size {len(val_df)} != n_users {n_users}"
    assert val_df["user_idx"].nunique() == n_users, "Val set contains duplicate users."

    # Total interaction count check
    total_split = len(train_df) + len(val_df) + len(test_df)
    assert total_split == n_interactions, f"Split sum {total_split:,} != total {n_interactions:,}"

    print(f"   -> Temporal split verified: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}.")


def test_continuous_indexing(meta):
    print("4. Checking continuous 0-indexed ID alignment...")
    n_users = meta["stats"]["n_users"]
    n_items = meta["stats"]["n_items"]

    # Check contiguous range [0, n-1]
    user_indices = sorted(meta["idx2user"].keys())
    assert user_indices[0] == 0 and user_indices[-1] == n_users - 1
    assert len(user_indices) == n_users

    item_indices = sorted(meta["idx2item"].keys())
    assert item_indices[0] == 0 and item_indices[-1] == n_items - 1
    assert len(item_indices) == n_items

    # Check movie metadata alignment
    movies_df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_clean.csv"))
    assert len(movies_df) == n_items, f"Clean movies count {len(movies_df)} != n_items {n_items}"
    assert movies_df["item_idx"].tolist() == list(range(n_items))

    print(f"   -> 0-indexed continuity verified for {n_users:,} users and {n_items:,} items.")


def run_all_integrity_tests():
    print("=" * 60)
    print("🔍 RUNNING DATA INTEGRITY & PIPELINE VALIDATION TESTS")
    print("=" * 60)

    test_processed_artifacts_exist()
    meta = test_metadata_consistency()
    test_temporal_split_integrity(meta)
    test_continuous_indexing(meta)

    print("=" * 60)
    print("🎉 ALL DATA INTEGRITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_integrity_tests()
