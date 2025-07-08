import tempfile
from pathlib import Path

import albumentations as A
import torch

from src.data.components.semantic_dataset import SemanticSegmentationDataset
from src.data.semantic_datamodule import SemanticDataModule
from tests.data.components.test_semantic_dataset import create_test_data


def test_semantic_datamodule(testing_semantic_transform: A.Compose) -> None:
    """Test `SemanticDataModule`.

    Args:
        testing_instance_transform (A.Compose): Albumentations transform pipeline.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        set_type_to_data_dir = {}
        for set_type in ("train", "val", "test"):
            set_dir = data_dir / set_type
            set_dir.mkdir()
            create_test_data(set_dir, num_images=8)
            set_type_to_data_dir[set_type] = set_dir

        semantic_datamodule = SemanticDataModule(
            set_type_to_data_dir["train"],
            set_type_to_data_dir["val"],
            set_type_to_data_dir["test"],
            num_classes=8,
            batch_size=2,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True,
            ignore_index=255,
            train_transform=testing_semantic_transform,
            test_transform=testing_semantic_transform,
        )
        semantic_datamodule.setup()
        assert isinstance(semantic_datamodule.train_dataset, SemanticSegmentationDataset)
        assert isinstance(semantic_datamodule.val_dataset, SemanticSegmentationDataset)
        assert isinstance(semantic_datamodule.test_dataset, SemanticSegmentationDataset)
        train_dataloader = semantic_datamodule.train_dataloader()
        val_dataloader = semantic_datamodule.val_dataloader()
        test_dataloader = semantic_datamodule.test_dataloader()
        for dataloader in (train_dataloader, val_dataloader, test_dataloader):
            for batch in dataloader:
                assert isinstance(batch["image"], torch.Tensor)
                assert isinstance(batch["mask"], torch.Tensor)
                assert isinstance(batch["image_id"], torch.Tensor)
                assert isinstance(batch["file_name"], list)
