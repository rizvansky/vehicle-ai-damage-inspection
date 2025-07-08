import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

from src.data.components.semantic_dataset import SemanticSegmentationDataset


def create_test_data(tmpdir: str | Path, num_images: int = 3) -> list[tuple[str, str]]:
    """Helper to create test dataset with real files.

    Args:
        tmpdir (str | Path): Directory where to save images and corresponding masks.
        num_images (int): Number of images to create.

    Returns:
        list[tuple[str, str]]: List containing pairs of images and corresponding masks paths.
    """

    data_dir = Path(tmpdir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    entries = []
    for img_idx in range(num_images):
        img_path = images_dir / f"image_{img_idx:03d}.png"
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path, "PNG")

        mask_path = masks_dir / f"image_{img_idx:03d}.png"
        mask_array = np.random.randint(0, 8, (256, 256), dtype=np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")
        mask_img.save(mask_path, "PNG")
        entries.append((str(img_path), str(mask_path)))

    return entries


def test_semantic_dataset(testing_semantic_transform: A.Compose) -> None:
    """Test basic dataset initialization.

    Args:
        testing_semantic_transform (A.Compose): Albumentations transform pipeline.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        create_test_data(tmpdir, num_images=5)
        dataset = SemanticSegmentationDataset(
            data_dir=str(tmpdir),
            num_classes=8,
            ignore_index=255,
            transform=testing_semantic_transform,
        )
        assert len(dataset) == 5
        assert dataset.num_classes == 8
        assert dataset.ignore_index == 255
        assert dataset.data_dir == Path(tmpdir)
        assert len(dataset.image_files) == 5
        assert len(dataset.mask_files) == 5

        for idx in range(len(dataset)):
            item = dataset[idx]

            assert "image" in item
            assert "mask" in item
            assert "image_id" in item
            assert "file_name" in item

            assert isinstance(item["image"], torch.Tensor)
            assert isinstance(item["mask"], torch.Tensor)
            assert isinstance(item["image_id"], int)
            assert isinstance(item["file_name"], str)

            assert item["image"].dim() == 3  # (C, H, W)
            assert item["image"].shape[0] == 3  # RGB channels
            assert item["image"].dtype == torch.float32

            assert item["mask"].dim() == 2  # (H, W)
            assert item["mask"].dtype == torch.int64

            assert item["file_name"].endswith(".png")
            assert f"{idx:03d}" in item["file_name"]
