"""Attention mechanism and transformers."""

import torch
import torch.nn as nn


class TabularTransformer(nn.Sequential):
    """Tabular transformer (from TabPFN-v2)."""

    def __init__(self, num_blocks: int, num_heads: int, embed_dim: int, hidden_dim: int):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TabularTransformerBlock(num_heads=num_heads, embed_dim=embed_dim, hidden_dim=hidden_dim)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, num_train: int | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, num_train)  # (B, R, C, E)
        return x


class TabularTransformerBlock(nn.Module):
    """Tabular transformer block (from TabPFN-v2)."""

    def __init__(self, num_heads: int, embed_dim: int, hidden_dim: int):
        super().__init__()

        # create MHA layers (operating on (..., seq_length, embed_dim)-shaped tensors)
        self.attention_between_features = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_between_samples = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # create MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # create layer norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, num_train: int | None) -> torch.Tensor:
        # ensure (B, R, C, E)-shaped inputs
        if x.ndim != 4:
            raise ValueError(f"Invalid tensor shape: {x.shape}")

        # apply attention between features (within each row separately)
        x = self._attention_between_features(x) + x
        x = self.ln1(x)

        # apply attention between samples (within each column separately)
        x = self._attention_between_samples(x, num_train=num_train) + x
        x = self.ln2(x)

        # run MLP
        x = self.mlp(x) + x
        x = self.ln3(x)

        return x

    def _attention_between_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention within each row separately."""
        # get shape
        shape = x.shape  # (B, R, C, E)

        # flatten batch and row dimensions
        x = x.flatten(0, 1)  # (B*R, C, E)

        # apply self-attention
        x = self.attention_between_features(x, x, x, need_weights=False)[0]  # (B*R, C, E)

        # reshape to original size
        x = x.view(*shape)  # (B, R, C, E)

        return x

    def _attention_between_samples(self, x: torch.Tensor, num_train: int | None = None) -> torch.Tensor:
        """Apply attention within each column separately."""
        # transpose row and col axes
        x = x.transpose(1, 2)  # (B, C, R, E)

        # get shape
        shape = x.shape  # (B, C, R, E)

        # flatten batch and row dimensions
        x = x.flatten(0, 1)  # (B*C, R, E)

        # apply self-attention
        if num_train is None:
            x = self.attention_between_samples(x, x, x, need_weights=False)[0]  # (B*C, R, E)
        else:
            x_train = x[:, :num_train]
            x_test = x[:, num_train:]

            # let train data attend to itself
            x_train = self.attention_between_samples(x_train, x_train, x_train, need_weights=False)[0]

            # let test data attend to train data
            x_test = self.attention_between_samples(x_test, x_train, x_train, need_weights=False)[0]

            x = torch.cat((x_train, x_test), dim=1)  # (B*C, R, E)

        # reshape to original size
        x = x.view(*shape)  # (B, C, R, E)

        # transpose row and col axes
        x = x.transpose(1, 2)  # (B, R, C, E)

        return x
