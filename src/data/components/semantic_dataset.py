from pathlib import Path

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTENSIONS = (".png",)


class SemanticSegmentationDataset(Dataset):
    """Dataset for semantic segmentation tasks."""

    def __init__(
        self,
        data_dir: str,
        transform: A.Compose | None = None,
        num_classes: int = 8,
        ignore_index: int = 255,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir (str): Root directory containing the dataset.
            transform (A.Compose | None, optional): Albumentations transform pipeline. Defaults to `None`.
            num_classes (int, optional): Number of segmentation classes. Defaults to 8.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
        """

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
        )
        self.mask_files = sorted(
            [p for p in self.masks_dir.iterdir() if p.suffix.lower() in MASK_EXTENSIONS]
        )
        assert len(self.image_files) == len(self.mask_files), (
            f"Mismatch between images ({len(self.image_files)}) and masks ({len(self.mask_files)})"
        )

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: Number of dataset entries.
        """

        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get dataset item.

        Args:
            idx (int): Item index.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing image and mask tensors.
        """

        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "image_id": idx,
            "file_name": image_path.name,
        }
