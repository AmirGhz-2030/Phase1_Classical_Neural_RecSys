"""
Classical Recommendation Baselines for Phase 1
==============================================
Implements standard baseline recommendation models:
1. PopularityRecommender: Global frequency/rating based ranking.
2. ItemKNNRecommender: Item-based Collaborative Filtering using Cosine Similarity on item representations.
3. MatrixFactorizationRecommender: Latent Factor Model via TruncatedSVD on User-Item matrix.

All models adhere to a unified scikit-learn style interface:
- fit(train_df, n_users, n_items)
- recommend(user_idx, top_k, filter_history)
- recommend_batch(user_indices, top_k, filter_history)

Author: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger("ClassicalModels")


class BaseRecommender:
    """Abstract Base Class for Recommender Models."""

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.n_users = 0
        self.n_items = 0
        self.user_history: Dict[int, Set[int]] = {}

    def _build_user_history(self, train_df: pd.DataFrame):
        """Pre-computes seen items for fast candidate filtering during recommendation."""
        grouped = train_df.groupby("user_idx")["item_idx"].apply(set)
        self.user_history = grouped.to_dict()

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        raise NotImplementedError

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_history: bool = True
    ) -> List[int]:
        raise NotImplementedError

    def recommend_batch(
        self,
        user_indices: List[int],
        top_k: int = 10,
        filter_history: bool = True
    ) -> Dict[int, List[int]]:
        """Batch recommendation generation across multiple users."""
        recs = {}
        for u in user_indices:
            recs[u] = self.recommend(u, top_k=top_k, filter_history=filter_history)
        return recs


class PopularityRecommender(BaseRecommender):
    """
    Most Popular Items Recommender.
    Ranks items based on global interaction counts in the training set.
    """

    def __init__(self):
        super().__init__(name="Popularity")
        self.popular_items_ranked: List[int] = []

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._build_user_history(train_df)

        item_counts = train_df["item_idx"].value_counts()
        self.popular_items_ranked = item_counts.index.tolist()

        if len(self.popular_items_ranked) < n_items:
            seen_items = set(self.popular_items_ranked)
            all_items = set(range(n_items))
            unseen = list(all_items - seen_items)
            self.popular_items_ranked.extend(unseen)

        logger.info(f"Popularity model fitted on {len(train_df):,} interactions.")

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_history: bool = True
    ) -> List[int]:
        seen = self.user_history.get(user_idx, set()) if filter_history else set()
        recommendations = []
        for item in self.popular_items_ranked:
            if item not in seen:
                recommendations.append(item)
                if len(recommendations) == top_k:
                    break
        return recommendations


class ItemKNNRecommender(BaseRecommender):
    """
    Item-Based Collaborative Filtering using Cosine Similarity.
    Item profile vector is normalized (n_items x n_users).
    Scores for user = (Item_norm @ Item_norm.T) @ user_items
    Equivalent to Item_norm @ (Item_norm.T @ user_vector)
    """

    def __init__(self, top_similar_k: int = 50):
        super().__init__(name="Item-KNN (CF)")
        self.top_similar_k = top_similar_k
        self.normalized_item_matrix: Optional[csr_matrix] = None
        self.fallback_popularity: List[int] = []

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._build_user_history(train_df)

        self.fallback_popularity = train_df["item_idx"].value_counts().index.tolist()

        # Build Sparse User-Item Matrix (n_users x n_items)
        rows = train_df["user_idx"].values
        cols = train_df["item_idx"].values
        data = np.ones(len(train_df), dtype=np.float32)

        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        # Transpose to get Item Matrix: shape (n_items x n_users)
        item_matrix = interaction_matrix.T.tocsr()
        logger.info(f"Normalizing Item Matrix ({n_items:,} x {n_users:,}) for Cosine Similarity...")

        # Row-wise L2 norm for each item
        norms = np.sqrt(item_matrix.power(2).sum(axis=1)).A1
        norms[norms == 0] = 1.0  # Avoid division by zero

        inv_diag = diags(1.0 / norms)
        self.normalized_item_matrix = (inv_diag @ item_matrix).tocsr()  # (n_items x n_users)
        logger.info("ItemKNN model fitted successfully.")

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_history: bool = True
    ) -> List[int]:
        seen = self.user_history.get(user_idx, set())
        if not seen or user_idx >= self.n_users:
            filtered = [it for it in self.fallback_popularity if it not in seen]
            return filtered[:top_k]

        # Get items user interacted with
        seen_items = np.array(list(seen), dtype=np.int32)

        # Slice submatrix of items user interacted with: shape (len(seen), n_users)
        user_items_matrix = self.normalized_item_matrix[seen_items]

        # Compute user representation in user-space: sum over item vectors
        # shape: (1 x n_users)
        user_rep = user_items_matrix.sum(axis=0)

        # Compute similarity score for ALL items: (n_items x n_users) @ (n_users x 1) -> (n_items,)
        scores = (self.normalized_item_matrix @ user_rep.T).A1

        if filter_history:
            scores[seen_items] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_top_indices.tolist()


class MatrixFactorizationRecommender(BaseRecommender):
    """
    Matrix Factorization Baseline (TruncatedSVD / Latent Factor Model).
    Decomposes User-Item matrix R approx U * V^T.
    """

    def __init__(self, n_factors: int = 64, random_state: int = 42):
        super().__init__(name="Matrix Factorization (SVD)")
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.fallback_popularity: List[int] = []

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._build_user_history(train_df)

        self.fallback_popularity = train_df["item_idx"].value_counts().index.tolist()

        logger.info(f"Building Sparse User-Item Matrix for SVD ({n_users:,} x {n_items:,})...")
        rows = train_df["user_idx"].values
        cols = train_df["item_idx"].values
        data = np.ones(len(train_df), dtype=np.float32)

        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        logger.info(f"Fitting TruncatedSVD with {self.n_factors} latent factors...")
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.user_factors = svd.fit_transform(R)  # (n_users, n_factors)
        self.item_factors = svd.components_.T      # (n_items, n_factors)

        logger.info("Matrix Factorization (SVD) fitted successfully.")

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_history: bool = True
    ) -> List[int]:
        seen = self.user_history.get(user_idx, set())
        if user_idx >= self.n_users or self.user_factors is None:
            filtered = [it for it in self.fallback_popularity if it not in seen]
            return filtered[:top_k]

        u_vec = self.user_factors[user_idx]  # (n_factors,)
        scores = self.item_factors @ u_vec    # (n_items,)

        if filter_history and seen:
            scores[list(seen)] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_top_indices.tolist()
