from pathlib import Path

import albumentations as A
import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.components.instance_dataset import InstanceSegmentationDataset


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:  # pragma: no cover
    """Custom collate function for instance segmentation.

    Args:
        batch (list[dict[str, torch.Tensor]]): Default batch.

    Returns:
        dict[str, torch.Tensor]: Restructured batch.
    """

    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]

    return {
        "image": images,
        "target": targets,
    }


class InstanceDataModule(L.LightningDataModule):
    """Data module for instance segmentation."""

    def __init__(
        self,
        images_dir: str | Path,
        train_annotations_path: str | Path,
        val_annotations_path: str | Path,
        test_annotations_path: str | Path,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        train_transform: A.Compose | None = None,
        test_transform: A.Compose | None = None,
    ) -> None:
        """Initialize instance data module.

        Args:
            images_dir (str | Path): Root directory containing images.
            train_annotations_path (str | Path): Path to the training annotations file.
            val_annotations_path (str | Path): Path to the validation annotations file.
            test_annotations_path (str | Path): Path to the test annotations file.
            batch_size (int, optional): Batch size for training. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory in DataLoader. Defaults to `False`.
            persistent_workers (bool, optional): Whether to use persistent workers. Defaults to `True`.
            train_transform (A.Compose | None): Train transform. Defaults to `None`.
            test_transform (A.Compose | None): Test transform. Defaults to `None`.
        """

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset: InstanceSegmentationDataset | None = None
        self.val_dataset: InstanceSegmentationDataset | None = None
        self.test_dataset: InstanceSegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Args:
            stage (str | None, optional): Pipeline stage. Defaults to `None`.
        """

        # Error handling for dataset initialization
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = InstanceSegmentationDataset(
                images_dir=self.hparams.images_dir,
                annotations_path=self.hparams.train_annotations_path,
                transform=self.hparams.train_transform,
            )

            self.val_dataset = InstanceSegmentationDataset(
                images_dir=self.hparams.images_dir,
                annotations_path=self.hparams.val_annotations_path,
                transform=self.hparams.test_transform,
            )

            self.test_dataset = InstanceSegmentationDataset(
                images_dir=self.hparams.images_dir,
                annotations_path=self.hparams.test_annotations_path,
                transform=self.hparams.test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            DataLoader: Created train dataloader.
        """

        # Check for dataset existence
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: Created validation dataloader.
        """

        # Check for dataset existence
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            DataLoader: Created test dataloader.
        """

        # Check for dataset existence
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
