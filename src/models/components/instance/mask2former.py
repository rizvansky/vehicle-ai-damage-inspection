import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss


class Mask2FormerInstance(nn.Module):
    """Mask2Former model for instance segmentation."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "facebook/mask2former-swin-small-coco-instance",
        pretrained: bool = True,
    ) -> None:
        """Initialize Mask2Former model.

        Args:
            num_classes (int): Number of object classes.
            backbone (str, optional): HuggingFace model name or local path.
                Defaults to `"facebook/mask2former-swin-small-coco-instance"`.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to `True`.
        """

        super().__init__()

        self.num_classes = num_classes
        if pretrained:  # pragma: no cover
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(backbone)
            if self.model.config.num_labels != num_classes:
                self.model.config.num_labels = num_classes
                self._reinit_classifier()
        else:
            config = Mask2FormerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)
        self.processor = AutoImageProcessor.from_pretrained(backbone)

    def _reinit_classifier(self) -> None:  # pragma: no cover
        """Reinitialize the classifier for new number of classes."""

        in_features = self.model.class_predictor.in_features
        self.model.class_predictor = nn.Linear(
            in_features, self.num_classes + 1
        )  # +1 for background
        original_weight_dict = (
            getattr(self.model.criterion, "weight_dict", None)
            if hasattr(self.model, "criterion")
            else None
        )

        # Create new criterion with updated config
        self.model.criterion = Mask2FormerLoss(
            config=self.model.config,
            weight_dict=original_weight_dict,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].
            mask_labels (list[torch.Tensor] | None, optional): List of ground truth masks for each image.
                Defaults to `None`.
            class_labels (list[torch.Tensor] | None, optional): List of ground truth class labels for each image.
                Defaults to `None`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing predictions and loss (if training).
        """

        if self.training and mask_labels is not None and class_labels is not None:
            # Format labels for training
            formatted_mask_labels = []
            formatted_class_labels = []

            for mask_label, class_label in zip(mask_labels, class_labels, strict=False):
                # Ensure masks are on the correct device and have proper dtype
                if mask_label.numel() > 0:
                    mask_label = mask_label.to(pixel_values.device)
                    class_label = class_label.to(pixel_values.device)
                else:
                    # Handle empty labels (create dummy background mask)
                    h, w = pixel_values.shape[-2:]
                    mask_label = torch.zeros(
                        (1, h, w), dtype=torch.long, device=pixel_values.device
                    )
                    class_label = torch.zeros((1,), dtype=torch.long, device=pixel_values.device)

                formatted_mask_labels.append(mask_label)
                formatted_class_labels.append(class_label)

            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=formatted_mask_labels,
                class_labels=formatted_class_labels,
            )
        else:
            outputs = self.model(pixel_values=pixel_values)

        return outputs

    def predict(
        self,
        pixel_values: torch.Tensor,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate predictions for inference.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            mask_threshold (float, optional): Threshold for mask binarization. Defaults to 0.5.
            overlap_mask_area_threshold (float, optional): Threshold for mask overlap handling.
                Defaults to 0.8.

        Returns:
            list[dict[str, torch.Tensor]]: List of predictions for each image containing:
                - segmentation: Full segmentation map [H, W] with segment IDs.
                - segments_info: List of dicts with keys: id, label_id, score, area.
        """

        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values)

            # Prepare target sizes for post-processing
            target_sizes = [(pixel_values.shape[-2], pixel_values.shape[-1])] * pixel_values.shape[
                0
            ]

            # Post-process to get final segmentation results
            predictions = self.processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
            )

            # Ensure predictions have proper format
            formatted_predictions = []
            for pred in predictions:
                if "segmentation" not in pred:
                    # Create empty prediction if post-processing fails
                    h, w = pixel_values.shape[-2:]
                    pred = {
                        "segmentation": torch.zeros(
                            (h, w), dtype=torch.long, device=pixel_values.device
                        ),
                        "segments_info": [],
                    }
                formatted_predictions.append(pred)

            return formatted_predictions
