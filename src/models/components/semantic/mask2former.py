import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerSemantic(nn.Module):
    """Mask2Former model for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "facebook/mask2former-swin-tiny-cityscapes-semantic",
        pretrained: bool = True,
        ignore_index: int = 255,
    ) -> None:
        """Initialize Mask2Former model.

        Args:
            num_classes (int): Number of segmentation classes.
            backbone (str, optional): HuggingFace model name or local path.
                Defaults to `"facebook/mask2former-swin-tiny-cityscapes-semantic"`.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to `True`.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
        """

        super().__init__()

        self.num_classes = num_classes + 1  # +1 for background class
        self.ignore_index = ignore_index

        if pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(backbone)
            # Update config if number of classes doesn't match
            if self.model.config.num_labels != self.num_classes:
                self.model.config.num_labels = self.num_classes
                self._reinit_classifier()
        else:
            config = Mask2FormerConfig.from_pretrained(backbone)
            config.num_labels = self.num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)

        self.processor = AutoImageProcessor.from_pretrained(backbone)

    def _reinit_classifier(self) -> None:
        """Reinitialize the classifier for new number of classes."""
        in_features = self.model.class_predictor.in_features
        self.model.class_predictor = nn.Linear(
            in_features, self.num_classes + 1
        )  # +1 for no-object class

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].
            mask_labels (list[dict[str, torch.Tensor]] | None, optional):
                List of label dictionaries for each image in batch. Each dict should contain:
                - 'masks': torch.Tensor of shape [num_instances, H, W]
                - 'labels': torch.Tensor of shape [num_instances] with class indices
                Defaults to `None`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing predictions and loss (if training).
        """

        if mask_labels is not None:
            # Convert our format to the expected format
            mask_labels_formatted = []
            class_labels_formatted = []

            for label_dict in mask_labels:
                masks = label_dict["masks"]  # [num_instances, H, W]
                labels = label_dict["labels"]  # [num_instances]
                mask_labels_formatted.append(masks)
                class_labels_formatted.append(labels)

            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels_formatted,
                class_labels=class_labels_formatted,
            )
        else:
            # Inference mode
            outputs = self.model(pixel_values=pixel_values)

        return outputs

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Generate predictions for inference.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].

        Returns:
            torch.Tensor: Predicted segmentation masks [B, H, W].
        """

        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values)

            # Handle the case where outputs might be in different formats
            if hasattr(outputs, "prediction_masks") and outputs.prediction_masks is not None:
                # Use prediction_masks if available
                predictions = outputs.prediction_masks
            else:
                # Fall back to post-processing
                batch_size, _, height, width = pixel_values.shape
                target_sizes = [(height, width)] * batch_size
                try:
                    predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(
                        outputs,
                        target_sizes=target_sizes,
                    )
                    predictions = torch.stack(predicted_segmentation_maps)
                except Exception as e:
                    # Fallback: create dummy predictions
                    print(f"Warning: Could not post-process Mask2Former outputs: {e}")
                    predictions = torch.zeros(
                        (batch_size, height, width), dtype=torch.long, device=pixel_values.device
                    )

            return predictions
