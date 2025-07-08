import torch

from src.models.components.semantic.deeplabv3plus import DeepLabV3PlusSemantic

NUM_CLASSES = 21
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = None  # for faster testing
IN_CHANNELS = 3
ACTIVATION = None
IGNORE_INDEX = 255
BATCH_SIZE = 2
HEIGHT = 256
WIDTH = 256


def test_deeplabv3plus() -> None:
    """Test DeepLabV3+ semantic segmentation model: initialization, forward/predict methods."""

    # Test initialization
    model = DeepLabV3PlusSemantic(
        NUM_CLASSES, ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, ACTIVATION, IGNORE_INDEX
    )
    assert model.num_classes == NUM_CLASSES
    assert model.ignore_index == IGNORE_INDEX
    assert hasattr(model, "model")
    assert hasattr(model, "ce_loss")
    assert hasattr(model, "dice_loss")
    assert hasattr(model, "focal_loss")
    assert isinstance(model.ce_loss, torch.nn.CrossEntropyLoss)
    assert model.ce_loss.ignore_index == IGNORE_INDEX

    # Test forward method
    model.train()
    pixel_values = torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, HEIGHT, WIDTH), dtype=torch.long)
    outputs = model.forward(pixel_values, labels)
    assert isinstance(outputs, dict)
    required_keys = ["logits", "loss", "ce_loss", "dice_loss", "focal_loss"]
    for key in required_keys:
        assert key in outputs, f"Missing key: {key}"
    logits = outputs["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES, HEIGHT, WIDTH)
    assert isinstance(outputs["loss"], torch.Tensor)
    assert outputs["loss"].dim() == 0  # scalar loss
    assert not torch.isnan(outputs["loss"])
    assert not torch.isinf(outputs["loss"])
    assert outputs["loss"].requires_grad
    for loss_key in ["ce_loss", "dice_loss", "focal_loss"]:
        loss_value = outputs[loss_key]
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0
        assert not torch.isnan(loss_value)
        assert not torch.isinf(loss_value)

    # Test predict method
    predictions = model.predict(pixel_values)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (BATCH_SIZE, HEIGHT, WIDTH)
    assert predictions.dtype == torch.long
    assert predictions.min() >= 0
    assert predictions.max() < NUM_CLASSES
    assert not model.training
