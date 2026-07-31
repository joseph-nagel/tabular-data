"""Lightning base wrapper."""

from abc import ABC, abstractmethod
from collections.abc import Sequence, Callable

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class BasePFN(LightningModule, ABC):
    """
    Base module for tabular PFN classifiers.

    Parameters
    ----------
    loss : str | Callable
        Loss function.
    lr : float
        Initial learning rate.

    """

    def __init__(
        self,
        loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "ce",
        lr: float = 1e-04,
    ):
        super().__init__()

        # set loss function
        if isinstance(loss, str):
            if loss.lower() == "ce":
                self.criterion = nn.CrossEntropyLoss(reduction="mean")
            else:
                raise ValueError(f"Invalid loss function name: {loss}")
        elif callable(loss):
            self.criterion = loss
        else:
            raise ValueError(f"Invalid loss function type: {type(loss)}")

        # set initial learning rate
        self.lr = abs(lr)

        # save hyperparameters
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def loss(self, x: torch.Tensor, y: torch.Tensor, num_train: int) -> torch.Tensor:
        """Compute loss."""
        # split into train and test data
        y_train = y[:, :num_train]  # (batch_size, num_train)
        y_test = y[:, num_train:]  # (batch_size, num_test)

        # compute test predictions
        y_pred = self(x, y_train)  # (batch_size, num_test, num_classes)

        # reshape
        y_test = y_test.flatten(end_dim=1)  # (batch_size*num_test,)
        y_pred = y_pred.flatten(end_dim=1)  # (batch_size*num_test, num_classes)

        return self.criterion(y_pred, y_test.long())

    # TODO: generalize to unequal train sizes, nested tensors, etc.
    @staticmethod
    def _get_batch(batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get batch."""
        # get features, targets and training sizes
        x = batch[0]  # (batch_size, num_samples, num_features)
        y = batch[1]  # (batch_size, num_samples)
        # num_active_features = batch[2]  # (batch_size,)
        # num_samples = batch[3]  # (batch_size,)
        num_train = batch[4]  # (batch_size,)

        # ensure idential training size within the batch
        if torch.all(num_train == num_train.flatten()[0]):
            num_train = num_train.flatten()[0]
        else:
            raise ValueError("Training sizes need to be identical within a batch")

        return x, y, num_train

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, num_train = self._get_batch(batch)
        loss = self.loss(x, y, num_train)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int):
        x, y, num_train = self._get_batch(batch)
        loss = self.loss(x, y, num_train)
        self.log("val_loss", loss.item())

    def test_step(self, batch: Sequence[torch.Tensor], batch_idx: int):
        x, y, num_train = self._get_batch(batch)
        loss = self.loss(x, y, num_train)
        self.log("test_loss", loss.item())

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
