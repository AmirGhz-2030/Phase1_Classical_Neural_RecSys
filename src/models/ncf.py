"""
Neural Collaborative Filtering (NCF / NeuMF) Implementation
============================================================
Implements the flagship NeuMF architecture (He et al., WWW 2017):
- Generalized Matrix Factorization (GMF) branch: Linear element-wise product of latent vectors.
- Multi-Layer Perceptron (MLP) branch: Non-linear interaction modeling with deep feedforward layers.
- NeuMF Fusion: Concatenates GMF and MLP outputs into a final prediction layer with Sigmoid activation.

Includes:
1. PyTorch NeuMF model definition.
2. Negative sampling dataset builder for implicit ranking.
3. Trainer with Binary Cross-Entropy (BCE) loss and Adam optimizer.
4. Fast Top-K recommendation inference.

Author: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import os
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("NCFModel")


class NCFDataset(Dataset):
    """
    Dataset for NCF training with implicit feedback and negative sampling.
    """

    def __init__(
        self,
        user_item_pairs: np.ndarray,
        n_items: int,
        user_history: Dict[int, Set[int]],
        num_negatives: int = 4
    ):
        self.users = user_item_pairs[:, 0]
        self.items = user_item_pairs[:, 1]
        self.n_items = n_items
        self.user_history = user_history
        self.num_negatives = num_negatives

        # Interleave positive and sampled negative interactions
        self.user_input, self.item_input, self.labels = self._sample_negatives()

    def _sample_negatives(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_users = []
        all_items = []
        all_labels = []

        for u, i in zip(self.users, self.items):
            # Positive sample
            all_users.append(u)
            all_items.append(i)
            all_labels.append(1.0)

            # Negative samples
            seen = self.user_history.get(u, set())
            for _ in range(self.num_negatives):
                neg_item = np.random.randint(0, self.n_items)
                while neg_item in seen:
                    neg_item = np.random.randint(0, self.n_items)
                all_users.append(u)
                all_items.append(neg_item)
                all_labels.append(0.0)

        return (
            torch.tensor(all_users, dtype=torch.long),
            torch.tensor(all_items, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_input[idx], self.item_input[idx], self.labels[idx]


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) Model combining GMF and MLP branches.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        mlp_layers = mlp_layers or [64, 32, 16]

        # 1. GMF Embedding Layers
        self.user_embed_gmf = nn.Embedding(n_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(n_items, latent_dim_gmf)

        # 2. MLP Embedding Layers
        self.user_embed_mlp = nn.Embedding(n_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(n_items, latent_dim_mlp)

        # 3. MLP Hidden Layers
        mlp_modules = []
        input_size = latent_dim_mlp * 2
        for hidden_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, hidden_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_modules)

        # 4. NeuMF Final Output Layer
        final_input_dim = latent_dim_gmf + mlp_layers[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Xavier Normal Initialization for stable gradient flow
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF Branch
        u_gmf = self.user_embed_gmf(user_indices)
        i_gmf = self.item_embed_gmf(item_indices)
        gmf_vector = u_gmf * i_gmf

        # MLP Branch
        u_mlp = self.user_embed_mlp(user_indices)
        i_mlp = self.item_embed_mlp(item_indices)
        mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        # Fusion
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.prediction_layer(fusion)
        return torch.sigmoid(logits).squeeze(-1)


class NCFRecommender:
    """
    Scikit-learn style wrapper for NeuMF training and Top-K recommendation.
    """

    def __init__(
        self,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_layers: Optional[List[int]] = None,
        lr: float = 0.001,
        batch_size: int = 1024,
        num_epochs: int = 5,
        num_negatives: int = 4,
        device: Optional[str] = None
    ):
        self.name = "NCF (NeuMF)"
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_negatives = num_negatives

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[NeuMF] = None
        self.n_users = 0
        self.n_items = 0
        self.user_history: Dict[int, Set[int]] = {}

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int, sample_ratio: float = 0.1):
        """
        Train NCF model. For fast and efficient learning on personal machines,
        we can sample a subset of positive interactions per epoch.
        """
        self.n_users = n_users
        self.n_items = n_items
        grouped = train_df.groupby("user_idx")["item_idx"].apply(set)
        self.user_history = grouped.to_dict()

        logger.info(f"Initializing NeuMF model on device: {self.device}...")
        self.model = NeuMF(
            n_users=n_users,
            n_items=n_items,
            latent_dim_gmf=self.latent_dim_gmf,
            latent_dim_mlp=self.latent_dim_mlp,
            mlp_layers=self.mlp_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # Sample positive pairs if full dataset is massive (19M records)
        if sample_ratio < 1.0:
            train_subset = train_df.sample(frac=sample_ratio, random_state=42)
        else:
            train_subset = train_df

        pos_pairs = train_subset[["user_idx", "item_idx"]].values
        logger.info(f"Training on {len(pos_pairs):,} positive pairs with {self.num_negatives} negatives/pos...")

        dataset = NCFDataset(
            user_item_pairs=pos_pairs,
            n_items=n_items,
            user_history=self.user_history,
            num_negatives=self.num_negatives
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device == "cuda"))

        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            for u_batch, i_batch, y_batch in dataloader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(u_batch, i_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            elapsed = time.time() - t_epoch
            logger.info(f"Epoch [{epoch}/{self.num_epochs}] - Loss: {avg_loss:.4f} ({elapsed:.1f}s)")

        logger.info("NCF Training Complete.")

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_history: bool = True
    ) -> List[int]:
        self.model.eval()
        seen = self.user_history.get(user_idx, set()) if filter_history else set()

        all_items = np.arange(self.n_items)
        u_tensor = torch.full((self.n_items,), user_idx, dtype=torch.long, device=self.device)
        i_tensor = torch.tensor(all_items, dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.model(u_tensor, i_tensor).cpu().numpy()

        if filter_history and seen:
            scores[list(seen)] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_top_indices.tolist()

    def recommend_batch(
        self,
        user_indices: List[int],
        top_k: int = 10,
        filter_history: bool = True,
        batch_size: int = 256
    ) -> Dict[int, List[int]]:
        """
        Optimized batch recommendation scoring across users.
        """
        self.model.eval()
        recommendations = {}
        all_items = np.arange(self.n_items)
        n_users_batch = len(user_indices)

        for i in range(0, n_users_batch, batch_size):
            batch_uids = user_indices[i : i + batch_size]
            u_expanded = np.repeat(batch_uids, self.n_items)
            i_expanded = np.tile(all_items, len(batch_uids))

            u_tensor = torch.tensor(u_expanded, dtype=torch.long, device=self.device)
            i_tensor = torch.tensor(i_expanded, dtype=torch.long, device=self.device)

            with torch.no_grad():
                scores_all = self.model(u_tensor, i_tensor).cpu().numpy()

            scores_matrix = scores_all.reshape(len(batch_uids), self.n_items)

            for idx, uid in enumerate(batch_uids):
                user_scores = scores_matrix[idx]
                seen = self.user_history.get(uid, set())
                if filter_history and seen:
                    user_scores[list(seen)] = -np.inf
                top_idx = np.argpartition(user_scores, -top_k)[-top_k:]
                sorted_top = top_idx[np.argsort(-user_scores[top_idx])]
                recommendations[uid] = sorted_top.tolist()

        return recommendations
