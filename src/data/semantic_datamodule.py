from pathlib import Path

import albumentations as A
import lightning as L
from torch.utils.data import DataLoader

from src.data.components.semantic_dataset import SemanticSegmentationDataset


class SemanticDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str | Path,
        val_data_dir: str | Path,
        test_data_dir: str | Path,
        num_classes: int = 8,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        ignore_index: int = 255,
        train_transform: A.Compose | None = None,
        test_transform: A.Compose | None = None,
    ) -> None:
        """Initialize semantic data module.

        Args:
            train_data_dir (str | Path): Root directory containing train data.
            val_data_dir (str | Path): Root directory containing test data.
            test_data_dir (str | Path): Root directory containing validation data.
            num_classes (int, optional): Number of segmentation classes. Defaults to 8.
            batch_size (int, optional): Batch size for training. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory in DataLoader. Defaults to `False`.
            persistent_workers (bool, optional): Whether to use persistent workers. Defaults to `True`.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
            train_transform (A.Compose | None): Train transform. Defaults to `None`.
            test_transform (A.Compose | None): Test transform. Defaults to `None`.
        """

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset: SemanticSegmentationDataset | None = None
        self.val_dataset: SemanticSegmentationDataset | None = None
        self.test_dataset: SemanticSegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Args:
            stage (str | None): Pipeline stage. Defaults to `None`.
        """

        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = SemanticSegmentationDataset(
                data_dir=self.hparams.train_data_dir,
                transform=self.hparams.train_transform,
                num_classes=self.hparams.num_classes,
                ignore_index=self.hparams.ignore_index,
            )
            self.val_dataset = SemanticSegmentationDataset(
                data_dir=self.hparams.val_data_dir,
                transform=self.hparams.test_transform,
                num_classes=self.hparams.num_classes,
                ignore_index=self.hparams.ignore_index,
            )
            self.test_dataset = SemanticSegmentationDataset(
                data_dir=self.hparams.test_data_dir,
                transform=self.hparams.test_transform,
                num_classes=self.hparams.num_classes,
                ignore_index=self.hparams.ignore_index,
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            DataLoader: Created train dataloader.
        """

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: Created validation dataloader.
        """

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            DataLoader: Created test dataloader.
        """

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
