import torch
from transformers import Mask2FormerForUniversalSegmentation

from src.models.components.instance.mask2former import Mask2FormerInstance

NUM_CLASSES = 5
BACKBONE = "facebook/mask2former-swin-small-coco-instance"
BATCH_SIZE = 2
HEIGHT = 256
WIDTH = 256
IN_CHANNELS = 3


def create_sample_instance_batch() -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Create sample batch data for testing."""

    pixel_values = torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    mask_labels, class_labels = [], []
    for _ in range(BATCH_SIZE):
        num_instances = torch.randint(1, 4, (1,)).item()
        masks = torch.randint(0, 2, (num_instances, HEIGHT, WIDTH), dtype=torch.float32)
        mask_labels.append(masks)
        labels = torch.randint(1, NUM_CLASSES + 1, (num_instances,), dtype=torch.long)
        class_labels.append(labels)

    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def test_instance_mask2former() -> None:
    """Test Mask2Former instance segmentation model: initialization and forward/predict methods."""

    model = Mask2FormerInstance(  # `pretrained` is set to `False` for faster testing
        NUM_CLASSES, backbone=BACKBONE, pretrained=False
    )
    assert isinstance(model.model, Mask2FormerForUniversalSegmentation)

    # Test `forward` method in training mode
    model.train()
    sample_instance_batch = create_sample_instance_batch()
    outputs = model.forward(
        pixel_values=sample_instance_batch["pixel_values"],
        mask_labels=sample_instance_batch["mask_labels"],
        class_labels=sample_instance_batch["class_labels"],
    )

    assert hasattr(outputs, "loss")
    assert hasattr(outputs, "masks_queries_logits")
    assert hasattr(outputs, "class_queries_logits")

    assert isinstance(outputs.loss, torch.Tensor)
    assert outputs.loss.dim() == 0
    assert outputs.loss.requires_grad
    assert not torch.isnan(outputs.loss)
    assert not torch.isinf(outputs.loss)

    # Test `forward` method in inference mode
    inference_batch = torch.rand(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(pixel_values=inference_batch)

    assert hasattr(outputs, "masks_queries_logits")
    assert hasattr(outputs, "class_queries_logits")

    batch_size = inference_batch.shape[0]
    num_queries = outputs.masks_queries_logits.shape[1]
    assert outputs.masks_queries_logits.shape[0] == batch_size
    assert outputs.class_queries_logits.shape[0] == batch_size
    assert outputs.class_queries_logits.shape[1] == num_queries

    # Test `predict` method
    predictions = model.predict(inference_batch)
    assert isinstance(predictions, list)
    assert len(predictions) == inference_batch.shape[0]
    for pred in predictions:
        assert isinstance(pred, dict)
        post_processed_keys = {"segmentation", "segments_info"}
        has_post_processed = all(key in pred for key in post_processed_keys)
        assert has_post_processed
