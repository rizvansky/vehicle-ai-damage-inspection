import torch
from transformers import SegformerForSemanticSegmentation

from src.models.components.semantic.segformer import SegFormerSemantic

NUM_CLASSES = 8
BACKBONE = "nvidia/segformer-b0-finetuned-ade-512-512"
PRETRAINED = False
IGNORE_INDEX = 255
BATCH_SIZE = 2
IN_CHANNELS = 3
HEIGHT = 128
WIDTH = 128


def test_segformer() -> None:
    """Test SegFormer semantic segmentation model: initialization and forward/predict methods."""

    # Test initialization
    model = SegFormerSemantic(NUM_CLASSES, BACKBONE, PRETRAINED, IGNORE_INDEX)
    assert model.num_classes == NUM_CLASSES + 1  # +1 for background class
    assert model.ignore_index == IGNORE_INDEX
    assert hasattr(model, "model")
    assert isinstance(model.model, SegformerForSemanticSegmentation)
    assert model.model.config.num_labels == NUM_CLASSES + 1  # +1 for background class

    # Test `forward` method in training mode
    model.train()
    pixel_values = torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    labels = torch.randint(0, NUM_CLASSES + 1, (BATCH_SIZE, HEIGHT, WIDTH), dtype=torch.long)
    outputs = model.forward(pixel_values, labels)
    assert isinstance(outputs, dict)
    expected_keys = ["logits", "loss"]
    for key in expected_keys:
        assert key in outputs, f"Missing key: {key}"
    logits = outputs["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == BATCH_SIZE
    assert logits.shape[1] == NUM_CLASSES + 1
    assert logits.dim() == 4  # [B, C, H, W]
    loss = outputs["loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.requires_grad

    # Test `forward` method in inference mode
    model.eval()
    with torch.no_grad():
        outputs = model.forward(pixel_values)
    assert isinstance(outputs, dict)
    assert "loss" not in outputs
    assert "logits" in outputs
    logits = outputs["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == BATCH_SIZE
    assert logits.shape[1] == NUM_CLASSES + 1  # +1 for background class
    assert not logits.requires_grad

    # Test `predict` method
    predictions = model.predict(pixel_values)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (BATCH_SIZE, HEIGHT, WIDTH)
    assert predictions.dtype in [torch.long, torch.int64]
    assert predictions.min() >= 0
    assert predictions.max() < NUM_CLASSES + 1  # +1 for background class
    assert not model.training, "Model should be in eval mode after predict"
