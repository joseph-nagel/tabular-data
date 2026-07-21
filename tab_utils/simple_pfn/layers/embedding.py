"""Preprocessing and encoding layers."""

import torch
import torch.nn as nn


class FeatureNormalization(nn.Module):
    """Feature normalization layer (from TabPFN-v2)."""

    def __init__(self, eps: float = 1e-08, clip: float | None = 100.0):
        super().__init__()
        self.eps = abs(eps)
        self.clip = abs(clip) if clip is not None else None

    def forward(self, x: torch.Tensor, num_train: int | None = None) -> torch.Tensor:
        # ensure (batch_size, num_rows, num_cols)-shaped inputs
        if x.ndim != 3:
            raise ValueError(f"Invalid tensor shape: {x.shape}")

        # calculate mean and std. over rows
        if num_train is None:
            mean = x.mean(dim=1, keepdim=True)  # (batch_size, 1, num_cols)
            std = x.std(dim=1, keepdim=True)  # (batch_size, 1, num_cols)
        else:
            mean = x[:, :num_train].mean(dim=1, keepdim=True)  # (batch_size, 1, num_cols)
            std = x[:, :num_train].std(dim=1, keepdim=True)  # (batch_size, 1, num_cols)

        # scale tensor
        x_normalized = (x - mean) / (std + self.eps)  # (batch_size, num_rows, num_cols)

        # clip values
        if self.clip is not None:
            x_normalized = torch.clamp(x_normalized, min=-self.clip, max=self.clip)  # (batch_size, num_rows, num_cols)

        return x_normalized


class RepeatedFeatureGrouping(nn.Module):
    """
    Feature grouping through circular permutation (from TabICL-v2).

    Parameters
    ----------
    feature_group_size : int
        Number of features per group.

    """

    def __init__(self, feature_group_size: int = 3):
        super().__init__()

        if feature_group_size < 1:
            raise ValueError("Invalid feature group size")

        self.feature_group_size = feature_group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure (batch_size, num_rows, num_cols)-shaped inputs
        if x.ndim != 3:
            raise ValueError(f"Invalid tensor shape: {x.shape}")

        if self.feature_group_size == 1:
            x_grouped = x.unsqueeze(-1)  # (batch_size, num_rows, num_cols, 1)

        else:
            # group through circular permutation
            num_cols = x.shape[2]
            col_ids = torch.arange(num_cols, device=x.device)

            x_grouped = torch.stack(
                [x[:, :, (col_ids + 2**idx - 1) % num_cols] for idx in range(self.feature_group_size)], dim=-1
            )  # (batch_size, num_rows, num_groups=num_cols, feature_group_size)

        return x_grouped


# TODO: generalize to regression problems
class TargetAwareCellEmbedding(nn.Module):
    """
    Target-aware cell embedding (from TabPFN-v2).

    Parameters
    ----------
    feature_group_size : int
        Number of input features per group.
    max_classes : int
        Maximum number of classes for the targets.
    embed_dim : int
        Dimensionality of the input and target embeddings.

    """

    def __init__(self, feature_group_size: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.x_embed = nn.Linear(feature_group_size, embed_dim)
        self.y_embed = nn.Embedding(num_classes, embed_dim)  # note that this only works for classification

    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        # ensure (batch_size, num_rows, num_cols, feature_group_size)-shaped inputs
        if x.ndim != 4:
            raise ValueError(f"Invalid tensor shape: {x.shape}")

        # ensure (batch_size, num_train)-shaped targets
        if y_train.ndim == 3 and y_train.shape[2] == 1:
            y_train = y_train.squeeze(2)

        if y_train.ndim != 2:
            raise ValueError(f"Invalid tensor shape: {y_train.shape}")

        # get shapes
        num_rows = x.shape[1]
        num_train = y_train.shape[1]
        num_test = num_rows - num_train

        if num_test < 0:
            raise ValueError(f"Number of train rows ({num_train}) larger than total number ({num_rows})")

        # linearly embed input cells
        x = self.x_embed(x)  # (batch_size, num_rows, num_cols, embed_dim)

        # embed targets
        y_emb = self.y_embed(y_train)  # (batch_size, num_train, embed_dim)
        y_emb = y_emb.unsqueeze(2)  # (batch_size, num_train, 1, embed_dim)

        # add target embeddings (from TabICL-v2)
        # x[:, :num_train] += y_emb  # (batch_size, num_rows, num_cols, embed_dim)

        # concatenate target embeddings (from TabPFN-v2)
        mean = y_emb.mean(dim=1, keepdims=True)  # (batch_size, 1, 1, embed_dim)
        padding = mean.expand(-1, num_test, -1, -1)  # (batch_size, num_test, 1, embed_dim)
        y_emb = torch.cat((y_emb, padding), dim=1)  # (batch_size, num_rows, 1, embed_dim)
        x = torch.cat((x, y_emb), dim=2)  # (batch_size, num_rows, num_cols + 1, embed_dim)

        return x
