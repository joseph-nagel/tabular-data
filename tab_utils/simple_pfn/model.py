"""
Simple tabular PFN for classification.

Summary
-------
A simplified tabular PFN model for classification is provided in this module.
It is largely based on TabPFN-v2 (in how attention is applied over rows and columns),
but also uses some techniques from TabICL-v2 (e.g. repeated feature grouping).
The implementation serves educational purposes only.
It therefore prioritizes clarity over completeness and efficiency.

See also NanoTabPFN (https://github.com/automl/nanoTabPFN) and NanoTabICL (https://github.com/soda-inria/nanotabicl).

"""

import torch
import torch.nn as nn

from .layers import (
    FeatureNormalization,
    RepeatedFeatureGrouping,
    TargetAwareCellEmbedding,
    TabularTransformer,
)


class SimplePFN(nn.Module):
    """
    Simple tabular PFN architecture for classification.

    Parameters
    ----------
    num_classes : int
        Maximum number of classes.
    num_blocks : int
        Number of attention blocks in the transformer.
    num_heads : int
        Number of attention heads in the transformer.
    embed_dim : int
        Dimensionality of the cell embeddings.
    hidden_dim : int
        Hidden dimensionality of the MLPs.
    feature_group_size: int
        Number of features per group.

    """

    def __init__(
        self,
        num_classes,
        num_blocks: int = 3,
        num_heads: int = 4,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        feature_group_size: int = 1,
    ):
        super().__init__()

        # create feature normalization
        self.feature_normalization = FeatureNormalization()

        # create repeated feature grouping
        self.repeated_feature_grouping = RepeatedFeatureGrouping(feature_group_size)

        # create target-aware cell embedding
        self.target_aware_cell_embedding = TargetAwareCellEmbedding(
            feature_group_size=feature_group_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

        # create tabular transformer
        self.tabular_transformer = TabularTransformer(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

        # create final MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        # ensure features of shape (batch_size, num_rows, num_cols)
        if x.ndim != 3:
            raise ValueError(f"Invalid features shape: {x.shape}")

        # ensure targets of shape (batch_size, num_train) or (batch_size, num_train, 1)
        if y_train.ndim != 2 and not (y_train.ndim == 3 and y_train.shape[2] == 1):
            raise ValueError(f"Invalid targets shape: {y_train.shape}")

        # get train size
        num_train = y_train.shape[1]

        # normalize features
        x = self.feature_normalization(x, num_train)  # (batch_size, num_rows, num_cols)

        # create grouped features
        x = self.repeated_feature_grouping(x)  # (batch_size, num_rows, num_cols, feature_group_size)

        # compute target-aware cell embeddings
        x = self.target_aware_cell_embedding(x, y_train)  # (batch_size, num_rows, num_cols + 1, embed_dim)

        # run tabular transformer
        x = self.tabular_transformer(x, num_train)  # (batch_size, num_rows, num_cols + 1, embed_dim)

        # get test row target embeddings
        x = x[:, num_train:, -1]  # (batch_size, num_test, embed_dim)

        # run final MLP
        x = self.mlp(x)  # (batch_size, num_test, num_classes)

        return x
