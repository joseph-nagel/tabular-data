"""Dataset prior."""

from collections.abc import Sequence
from typing import Any, Self

import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.fabric.utilities.seed import pl_worker_init_function
from tabicl.prior import PriorDataset


class TabICLPriorDataset(Dataset):
    """Finite-length version for the infinite TabICL dataset prior."""

    def __init__(self, num_batches: int, *args: Any, **kwargs: Any):
        self.prior_dataset = PriorDataset(*args, **kwargs)  # initialize infinite prior dataset
        self.num_batches = abs(num_batches)

    def __len__(self) -> int:
        return self.num_batches  # do not multiply by self.prior_dataset.batch_size

    def __getitem__(self, idx: int) -> Sequence[torch.Tensor]:
        return next(self.prior_dataset)


class TabICLPriorIterableDataset(PriorDataset):
    """TabICL dataset prior with finite number of batches."""

    def __init__(self, num_batches: int | None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)  # initialize infinite prior dataset parent class
        self.num_batches = num_batches
        self._batch_idx = 0 if num_batches is not None else None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Sequence[torch.Tensor]:
        if self.num_batches is None:
            return super().__next__()
        elif self._batch_idx < self.num_batches:
            self._batch_idx += 1
            return super().__next__()
        else:
            raise StopIteration()


class PriorDataModule(LightningDataModule):
    """
    DataModule for the TabICL dataset prior.

    Parameters
    ----------
    num_train_batches : int | None = None
        Number of batches for training.
    num_val_batches : int | None = None
        Number of batches for validation.
    num_test_batches : int | None = None
        Number of batches for testing.
    batch_size : int
        Total number of datasets to generate per batch.
    batch_size_per_gp : int
         Number of datasets per group with similar characteristics.
    min_features : int
        Minimum number of features per dataset.
    max_features : int
        Maximum number of features per dataset.
    max_classes : int
        Maximum number of target classes.
    min_seq_len : int | None
        Minimum number of samples per dataset.
    max_seq_len : int
        Maximum number of samples per dataset.
    min_train_size : int | float
        Minimum train split size.
    max_train_size : int | float
        Maximum train split size.
    prior_type : {"mlp_scm", "tree_scm", "mix_scm", "dummy"}
        Type of the dataset prior.
    num_workers : int
        Number of workers for the loader.
    prefetch_factor : int | None = None
        Number of batches per worker to be loaded in advance.

    """

    def __init__(
        self,
        num_train_batches: int | None = None,
        num_val_batches: int | None = None,
        num_test_batches: int | None = None,
        batch_size: int = 32,
        batch_size_per_gp: int = 4,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: int | None = None,
        max_seq_len: int = 1024,
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        prior_type: str = "mlp_scm",
        num_workers: int = 0,
        prefetch_factor: int | None = None,
    ):
        super().__init__()

        # set batch numbers
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches
        self.num_test_batches = num_test_batches

        # set dataset parameters
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.prior_type = prior_type

        # set loader parameters
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def _make_dataset(self, num_batches: int | None = None) -> TabICLPriorDataset:
        """Create prior dataset."""
        return TabICLPriorDataset(
            num_batches=num_batches,
            batch_size=self.batch_size,  # set batch size on the Dataset level
            batch_size_per_gp=self.batch_size_per_gp,
            min_features=self.min_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
            min_train_size=self.min_train_size,
            max_train_size=self.max_train_size,
            prior_type=self.prior_type,
            n_jobs=1,  # deactivate Dataset-level parallelism
            # device="cpu",
        )

    # TODO: Does this ensure different seeds for train, val. and test set?
    def _make_loader(self, dataset: Dataset) -> DataLoader:
        """Create prior dataloader."""
        return DataLoader(
            dataset,
            batch_size=None,  # turn off additional batching on the DataLoader level
            shuffle=False,  # turn off shuffling since not necessary
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,  # use page-locked memory if data is fetched in parallel subprocesses
            worker_init_fn=pl_worker_init_function if self.num_workers > 0 else None,  # set random seed per worker
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def setup(self, stage: str) -> None:
        """Set up train/val./test datasets."""
        if stage == "fit":
            self.train_set = self._make_dataset(self.num_train_batches)
        if stage in ("fit", "validate"):
            self.val_set = self._make_dataset(self.num_val_batches)
        elif stage == "test":
            self.test_set = self._make_dataset(self.num_test_batches)

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return self._make_loader(self.train_set)

    def val_dataloader(self) -> DataLoader:
        """Create val. dataloader."""
        return self._make_loader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self._make_loader(self.test_set)
