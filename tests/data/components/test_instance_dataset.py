import json
import tempfile
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

from src.data.components.instance_dataset import InstanceSegmentationDataset


def create_coco_annotation_file(
    images_dir: Path, ann_file: Path, num_images: int = 3
) -> dict[str, Any]:
    """Create a real COCO annotation file with corresponding images.

    Args:
        images_dir (Path): Directory to save images.
        ann_file (Path): Path for annotation file.
        num_images (int, optional): Number of images to create. Defaults to 3.

    Returns:
        dict[str, Any]: Created COCO annotation file.
    """

    # COCO format structure
    coco_data = {
        "info": {
            "description": "Test COCO dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "test",
            "date_created": "2025-01-01",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "dent", "supercategory": "damage"},
            {"id": 2, "name": "crack", "supercategory": "damage"},
            {"id": 3, "name": "scratch", "supercategory": "damage"},
            {"id": 4, "name": "paint_chip", "supercategory": "damage"},
        ],
    }

    annotation_id = 1
    for img_id in range(1, num_images + 1):
        img_width, img_height = 640, 480
        img_filename = f"image_{img_id:03d}.jpg"
        coco_data["images"].append(
            {
                "id": img_id,
                "width": img_width,
                "height": img_height,
                "file_name": img_filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        # Create actual image file
        img_path = images_dir / img_filename
        img_array = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path, "JPEG")

        # Create annotations for this image
        for obj_idx in range(3):
            x_center = np.random.randint(50, img_width - 100)
            y_center = np.random.randint(50, img_height - 100)
            width = np.random.randint(30, 80)
            height = np.random.randint(30, 80)
            x = x_center - width // 2
            y = y_center - height // 2
            x = max(0, min(x, img_width - width))
            y = max(0, min(y, img_height - height))

            # Create segmentation mask (simple rectangle for testing)
            segmentation = [[x, y, x + width, y, x + width, y + height, x, y + height]]
            if obj_idx % 2:  # Alternate between polygon and RLE representation
                segmentation = coco_mask.frPyObjects(segmentation, img_height, img_width)[0]
                # Convert bytes to string for JSON serialization
                if isinstance(segmentation["counts"], bytes):
                    segmentation["counts"] = segmentation["counts"].decode("utf-8")

            area = width * height
            category_id = np.random.randint(1, 5)  # 1-4

            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": [float(x), float(y), float(width), float(height)],
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

    # Save annotation file
    with open(ann_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return coco_data


def test_instance_segm_dataset_on_valid_data(testing_instance_transform: A.Compose) -> None:
    """Test `InstanceSegmentationDataset` on valid data.

    Args:
        testing_instance_transform (A.Compose): Albumentations transform pipeline.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        images_dir = root_dir / "images"
        images_dir.mkdir()
        annotations_dir = root_dir / "annotations"
        annotations_dir.mkdir()
        annotations_path = annotations_dir / "instances_test.json"
        create_coco_annotation_file(images_dir, annotations_path, num_images=5)
        dataset = InstanceSegmentationDataset(
            images_dir=images_dir,
            annotations_path=annotations_path,
            transform=testing_instance_transform,
        )

        assert len(dataset) > 0, "Dataset should have at least one valid image"
        assert dataset.images_dir == images_dir
        assert dataset.annotations_path == annotations_path
        for idx in range(min(len(dataset), 3)):
            item = dataset[idx]

            assert "image" in item
            assert "target" in item

            assert isinstance(item["image"], torch.Tensor)
            assert item["image"].dim() == 3  # (C, H, W)
            assert item["image"].shape[0] == 3  # RGB channels
            assert item["image"].dtype == torch.float32

            target = item["target"]
            assert isinstance(target, dict)

            required_keys = ["boxes", "labels", "masks", "area", "image_id"]
            for key in required_keys:
                assert key in target, f"Missing key: {key}"

            num_objects = target["boxes"].shape[0]
            assert target["boxes"].shape == (num_objects, 4)
            assert target["boxes"].dtype == torch.float32

            assert target["labels"].shape == (num_objects,)
            assert target["labels"].dtype == torch.int64
            assert (target["labels"] > 0).all()
            assert (target["labels"] <= 4).all()  # We have 4 categories

            assert target["masks"].shape[0] == num_objects
            assert target["masks"].dtype == torch.uint8
            assert target["masks"].min() >= 0
            assert target["masks"].max() <= 1

            assert target["area"].shape == (num_objects,)
            assert target["area"].dtype == torch.float32
            assert (target["area"] > 0).all()

            assert target["image_id"].numel() == 1
            assert target["image_id"].dtype == torch.int64
