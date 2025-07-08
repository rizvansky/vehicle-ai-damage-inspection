import torch
from transformers import Mask2FormerForUniversalSegmentation

from src.models.components.semantic.mask2former import Mask2FormerSemantic

NUM_CLASSES = 8
BACKBONE = "facebook/mask2former-swin-tiny-cityscapes-semantic"
PRETRAINED = False
IGNORE_INDEX = 255
BATCH_SIZE = 2
IN_CHANNELS = 3
HEIGHT = 128
WIDTH = 128


def test_semantic_mask2former() -> None:
    """Test Mask2Former semantic segmentation model: initialization and forward/predict methods."""

    model = Mask2FormerSemantic(NUM_CLASSES, BACKBONE, PRETRAINED, IGNORE_INDEX)

    # Test initialization
    assert model.num_classes == NUM_CLASSES + 1  # +1 for background class
    assert model.ignore_index == IGNORE_INDEX
    assert hasattr(model, "model")
    assert isinstance(model.model, Mask2FormerForUniversalSegmentation)
    assert model.model.config.num_labels == NUM_CLASSES + 1  # +1 for background class

    # Create test data
    pixel_values = torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    semantic_masks = torch.randint(
        0, NUM_CLASSES + 1, (BATCH_SIZE, HEIGHT, WIDTH), dtype=torch.long
    )

    # Test `forward` method in training mode
    model.train()

    # Create mask_labels in the format expected by the updated forward method
    mask_labels = []
    for batch_idx in range(BATCH_SIZE):
        mask = semantic_masks[batch_idx]
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes != IGNORE_INDEX]
        unique_classes = unique_classes[unique_classes < NUM_CLASSES + 1]

        if len(unique_classes) == 0:
            # Create dummy data for empty masks
            mask_labels.append(
                {
                    "masks": torch.zeros(1, HEIGHT, WIDTH, dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.long),
                }
            )
        else:
            # Create binary masks for each class
            batch_masks = []
            batch_classes = []
            for class_id in unique_classes:
                class_mask = (mask == class_id).float()
                if class_mask.sum() > 0:  # Only include non-empty masks
                    batch_masks.append(class_mask)
                    batch_classes.append(class_id)

            if batch_masks:
                mask_labels.append(
                    {
                        "masks": torch.stack(batch_masks),
                        "labels": torch.tensor(batch_classes, dtype=torch.long),
                    }
                )
            else:
                # Fallback for empty masks
                mask_labels.append(
                    {
                        "masks": torch.zeros(1, HEIGHT, WIDTH, dtype=torch.float32),
                        "labels": torch.tensor([0], dtype=torch.long),
                    }
                )

    # Test training forward pass
    outputs = model.forward(pixel_values, mask_labels)
    assert isinstance(outputs, dict)
    expected_keys = ["loss", "class_queries_logits", "masks_queries_logits"]
    present_keys = [key for key in expected_keys if key in outputs]
    assert len(present_keys) > 0, (
        f"None of expected keys {expected_keys} found in outputs: {list(outputs.keys())}"
    )
    if "loss" in outputs:
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.requires_grad
        assert loss.item() > 0

    # Test `forward` method in inference mode
    model.eval()
    with torch.no_grad():
        outputs = model.forward(pixel_values)
    assert isinstance(outputs, dict)
    assert "loss" not in outputs
    expected_keys = ["class_queries_logits", "masks_queries_logits"]
    present_keys = [key for key in expected_keys if key in outputs]
    assert len(present_keys) > 0, f"No prediction outputs found: {list(outputs.keys())}"
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad

    # Test `predict` method
    predictions = model.predict(pixel_values)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (BATCH_SIZE, HEIGHT, WIDTH)
    assert predictions.dtype in [torch.long, torch.int64]
    assert predictions.min() >= 0
    assert predictions.max() < NUM_CLASSES + 1  # +1 for background class
    assert not model.training
