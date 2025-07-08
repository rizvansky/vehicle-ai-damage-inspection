import tempfile
from pathlib import Path

import albumentations as A
import torch

from src.data.components.instance_dataset import InstanceSegmentationDataset
from src.data.instance_datamodule import InstanceDataModule
from tests.data.components.test_instance_dataset import create_coco_annotation_file


def test_instance_datamodule(testing_instance_transform: A.Compose) -> None:
    """Test `InstanceDataModule`.

    Args:
        testing_instance_transform (A.Compose): Albumentations transform pipeline.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "images"
        images_dir.mkdir()
        annotations_dir = tmpdir / "annotations"
        annotations_dir.mkdir()
        set_type_to_ann_filename = {}
        for set_type in ("train", "val", "test"):
            ann_file = annotations_dir / f"instances_{set_type}.json"
            create_coco_annotation_file(images_dir, ann_file, num_images=8)
            set_type_to_ann_filename[set_type] = ann_file

        instance_datamodule = InstanceDataModule(
            images_dir,
            set_type_to_ann_filename["train"],
            set_type_to_ann_filename["val"],
            set_type_to_ann_filename["test"],
            batch_size=2,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True,
            train_transform=testing_instance_transform,
            test_transform=testing_instance_transform,
        )
        instance_datamodule.setup()
        assert isinstance(instance_datamodule.train_dataset, InstanceSegmentationDataset)
        assert isinstance(instance_datamodule.val_dataset, InstanceSegmentationDataset)
        assert isinstance(instance_datamodule.test_dataset, InstanceSegmentationDataset)
        train_dataloader = instance_datamodule.train_dataloader()
        val_dataloader = instance_datamodule.val_dataloader()
        test_dataloader = instance_datamodule.test_dataloader()
        for dataloader in (train_dataloader, val_dataloader, test_dataloader):
            for batch in dataloader:
                assert isinstance(batch["image"], torch.Tensor)
                assert isinstance(batch["target"], list)
