import albumentations as A
import pytest
from albumentations.pytorch import ToTensorV2


@pytest.fixture
def testing_instance_transform() -> A.Compose:
    """Fixture that defines the instance segmentation `albumentations.Compose` transform for testing purposes.

    Returns:
        A.Compose: Created albumentations-based transform.
    """

    """
    bbox_params:
    target: albumentations.BboxParams
    format: coco
    label_fields: [category_ids]
    min_area: 1.0
    min_visibility: 0
    clip: true
    """

    transform = A.Compose(
        [
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1.0,
            min_visibility=0.1,
            clip=True,
        ),
        additional_targets={"masks": "masks"},
    )

    return transform


@pytest.fixture
def testing_semantic_transform() -> A.Compose:
    """Fixture that defines the semantic segmentation `albumentations.Compose` transform for testing purposes.

    Returns:
        A.Compose: Created albumentations-based transform.
    """

    transform = A.Compose(
        [
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return transform
