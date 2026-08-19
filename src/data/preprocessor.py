"""
MovieLens 20M Preprocessing Module for Phase 1
==============================================
Standard preprocessing pipeline for Classical & Neural Recommender Systems:
1. Ingest raw ratings and movies metadata.
2. Apply K-Core iterative filtering (dense matrix, removing cold users/items).
3. Continuous 0-indexed ID remapping for embedding layers (PyTorch/NCF).
4. Temporal Leave-One-Last train/validation/test splitting.
5. Save processed artifacts and metadata.

Author: Amirhossein Ghasemzadeh
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import os
import pickle
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase1Preprocessor")


class Phase1DataPreprocessor:
    def __init__(
        self,
        raw_dir: str = r"G:\Phase1_Classical_Neural_RecSys\data\raw",
        processed_dir: str = r"G:\Phase1_Classical_Neural_RecSys\data\processed",
        k_core: int = 20
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.k_core = k_core

        os.makedirs(self.processed_dir, exist_ok=True)

        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.stats: Dict[str, any] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading raw datasets from: {self.raw_dir}")
        ratings_path = os.path.join(self.raw_dir, "rating.csv")
        movies_path = os.path.join(self.raw_dir, "movie.csv")

        logger.info("Reading rating.csv...")
        ratings_df = pd.read_csv(
            ratings_path,
            dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
            parse_dates=["timestamp"]
        )
        logger.info(f"Loaded {len(ratings_df):,} raw ratings.")

        logger.info("Reading movie.csv...")
        movies_df = pd.read_csv(
            movies_path,
            dtype={"movieId": np.int32, "title": str, "genres": str}
        )
        logger.info(f"Loaded {len(movies_df):,} raw movies.")
        return ratings_df, movies_df

    def apply_k_core(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Applying {self.k_core}-Core iterative filtering...")
        initial_len = len(ratings_df)
        df = ratings_df.copy()

        iteration = 0
        while True:
            iteration += 1
            n_users_before = df["userId"].nunique()
            n_items_before = df["movieId"].nunique()

            # Active users
            user_counts = df["userId"].value_counts()
            active_users = user_counts[user_counts >= self.k_core].index
            df = df[df["userId"].isin(active_users)]

            # Popular items
            item_counts = df["movieId"].value_counts()
            popular_items = item_counts[item_counts >= self.k_core].index
            df = df[df["movieId"].isin(popular_items)]

            n_users_after = df["userId"].nunique()
            n_items_after = df["movieId"].nunique()

            logger.info(
                f"Round {iteration}: Users {n_users_before:,} -> {n_users_after:,} | "
                f"Items {n_items_before:,} -> {n_items_after:,} | Interactions: {len(df):,}"
            )

            if n_users_before == n_users_after and n_items_before == n_items_after:
                break

        retained_pct = (len(df) / initial_len) * 100.0
        logger.info(
            f"K-Core complete in {iteration} rounds. "
            f"Retained {len(df):,} / {initial_len:,} interactions ({retained_pct:.2f}%)."
        )
        return df

    def create_mappings(self, df: pd.DataFrame):
        logger.info("Creating continuous 0-indexed ID mappings...")
        unique_users = sorted(df["userId"].unique())
        unique_items = sorted(df["movieId"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}

        self.item2idx = {mid: idx for idx, mid in enumerate(unique_items)}
        self.idx2item = {idx: mid for mid, idx in self.item2idx.items()}

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        n_interactions = len(df)
        sparsity = (1.0 - (n_interactions / (n_users * n_items))) * 100.0

        self.stats = {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
            "sparsity_pct": sparsity,
            "k_core": self.k_core
        }
        logger.info(
            f"Mapped {n_users:,} Users and {n_items:,} Items. Matrix Sparsity: {sparsity:.4f}%."
        )

    def split_temporal(
        self, ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Performing Temporal Leave-One-Last train/val/test split...")
        df = ratings_df.copy()
        df["user_idx"] = df["userId"].map(self.user2idx)
        df["item_idx"] = df["movieId"].map(self.item2idx)

        # Sort temporally per user
        df = df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

        # Rank: 1 is the latest interaction
        df["rank_latest"] = df.groupby("user_idx")["timestamp"].rank(
            method="first", ascending=False
        ).astype(int)

        test_df = df[df["rank_latest"] == 1].drop(columns=["rank_latest"]).copy()
        val_df = df[df["rank_latest"] == 2].drop(columns=["rank_latest"]).copy()
        train_df = df[df["rank_latest"] > 2].drop(columns=["rank_latest"]).copy()

        logger.info(
            f"Split Complete -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )
        return train_df, val_df, test_df

    def process_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Aligning movie metadata with mapped indices...")
        valid_movie_ids = set(self.item2idx.keys())
        filtered_movies = movies_df[movies_df["movieId"].isin(valid_movie_ids)].copy()
        filtered_movies["item_idx"] = filtered_movies["movieId"].map(self.item2idx)
        filtered_movies = filtered_movies.sort_values("item_idx").reset_index(drop=True)
        return filtered_movies

    def run(self) -> Dict[str, any]:
        logger.info("=" * 60)
        logger.info("PHASE 1 PREPROCESSING PIPELINE STARTED")
        logger.info("=" * 60)

        ratings_raw, movies_raw = self.load_data()
        ratings_filtered = self.apply_k_core(ratings_raw)
        self.create_mappings(ratings_filtered)
        movies_clean = self.process_movies(movies_raw)
        train_df, val_df, test_df = self.split_temporal(ratings_filtered)

        logger.info(f"Saving processed artifacts to: {self.processed_dir}")
        train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        movies_clean.to_csv(os.path.join(self.processed_dir, "movies_clean.csv"), index=False)

        meta = {
            "stats": self.stats,
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item
        }
        with open(os.path.join(self.processed_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        logger.info("=" * 60)
        logger.info("PHASE 1 PREPROCESSING COMPLETE!")
        logger.info(f"Summary: {self.stats}")
        logger.info("=" * 60)
        return self.stats


if __name__ == "__main__":
    preprocessor = Phase1DataPreprocessor()
    preprocessor.run()
