import pytest
import torch


@pytest.fixture
def sample_instance_batch() -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Create sample batch data for testing."""

    batch_size = 2
    height, width = 256, 256
    channels = 3

    # Create sample images
    pixel_values = torch.randn(batch_size, channels, height, width)

    # Create sample ground truth masks and labels
    mask_labels = []
    class_labels = []

    for _ in range(batch_size):
        # Random number of instances per image (1-3)
        num_instances = torch.randint(1, 4, (1,)).item()

        # Create random masks
        masks = torch.randint(0, 2, (num_instances, height, width), dtype=torch.uint8)
        mask_labels.append(masks)

        # Create random class labels (1-indexed for COCO format)
        labels = torch.randint(1, 6, (num_instances,), dtype=torch.long)
        class_labels.append(labels)

    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }
