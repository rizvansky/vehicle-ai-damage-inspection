from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class InstanceSegmentationDataset(Dataset):
    """Dataset for instance segmentation tasks using COCO format."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            images_dir (str | Path): Images root directory.
            annotations_path (str | Path): Path to COCO format annotations file.
            transform (A.Compose | None, optional): Albumentations transform pipeline. Defaults to `None`.
        """

        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform

        # Load COCO annotations
        self.coco = COCO(str(self.annotations_path))
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out images without annotations
        self.image_ids = [
            img_id for img_id in self.image_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: Dataset length.
        """

        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get dataset item.

        Args:
            idx (int): Item index.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing image, boxes, labels, masks, etc.
        """

        image_id = self.image_ids[idx]

        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_dir / img_info["file_name"]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract targets
        boxes, labels, masks, areas = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x = max(0, min(x, orig_width - 1))
            y = max(0, min(y, orig_height - 1))
            w = max(1, min(w, orig_width - x))
            h = max(1, min(h, orig_height - y))
            boxes.append([x, y, w, h])  # keep COCO format [x, y, w, h]
            #####################################
            labels.append(ann["category_id"])

            if isinstance(ann["segmentation"], list):  # polygon format
                rle = coco_mask.frPyObjects(
                    ann["segmentation"], img_info["height"], img_info["width"]
                )
                mask = coco_mask.decode(rle)
                mask = mask.any(axis=2) if mask.ndim == 3 else mask
            else:  # RLE format
                mask = coco_mask.decode(ann["segmentation"])

            masks.append(mask)
            areas.append(ann["area"])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.stack(masks).astype(np.uint8)
        areas = np.array(areas, dtype=np.float32)

        # Apply albumentations transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                masks=masks,
                bboxes=boxes,
                category_ids=labels,
            )
            image = transformed["image"]
            masks = transformed["masks"]
            boxes = transformed["bboxes"]
            labels = transformed["category_ids"]

            # Convert boxes from COCO format [x, y, w, h] to [x1, y1, x2, y2]
            boxes = np.array(boxes, dtype=np.float32)
            if len(boxes) > 0:
                boxes_xyxy = np.zeros_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0]  # x1 = x
                boxes_xyxy[:, 1] = boxes[:, 1]  # y1 = y
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
                boxes = boxes_xyxy

            # Convert back to tensors
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Manual conversion if no transform
            image = self._manual_transform(image)
            boxes, labels, masks = self._manual_convert_targets(boxes, labels, masks)
        areas = torch.tensor(areas, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": areas,
            "image_id": torch.tensor([image_id]),
        }

        return {
            "image": image,
            "target": target,
        }

    def _manual_transform(self, image: np.ndarray) -> torch.Tensor:
        """Apply manual image transformation. Do conversion to `torch.Tensor`
            and normalize according to ImageNet mean and std values.

        Args:
            image (np.ndarray): Image to transform.

        Returns:
            torch.Tensor: Transformed image.
        """

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image

    def _manual_convert_targets(
        self, boxes: np.ndarray, labels: np.ndarray, masks: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert targets to tensors with proper format.

        Args:
            boxes (np.ndarray): Bounding boxes in COCO format.
            labels (np.ndarray): Labels.
            masks (np.ndarray): Masks.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Converted bounding boxes, labels, and masks.
        """

        if len(boxes) > 0:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
        masks = torch.tensor(masks, dtype=torch.uint8)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels, masks
