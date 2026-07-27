"""Dataset prior."""

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from tabicl.prior import PriorDataset


class PriorDataModule(LightningDataModule):
    """
    DataModule for the TabICL dataset prior.

    Parameters
    ----------
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
    prior_type : {`mlp_scm`, `tree_scm`, `mix_scm`, or `dummy`}
        Type of the dataset prior.
    num_workers : int
        Number of workers for the loader.
    prefetch_factor : int | None = None
        Number of batches per worker to be loaded in advance.

    """

    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__()

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
        self.kwargs = kwargs

        # set loader parameters
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    # def prepare_data(self) -> None:
    #     """Download data."""
    #     pass

    def setup(self, stage: str) -> None:
        """Set up dataset."""
        if stage in ("fit", "validate", "test"):
            self.dataset = PriorDataset(
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
                **self.kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=None,  # turn off additional batching on the DataLoader level
            shuffle=False,  # turn off shuffling since not necessary
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,  # use page-locked memory if data is fetched in a parallel subprocess
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Create val. dataloader."""
        return self.train_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self.train_dataloader()
