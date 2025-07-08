import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerSemantic(nn.Module):
    """SegFormer model for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        pretrained: bool = True,
        ignore_index: int = 255,
    ) -> None:
        """Initialize SegFormer model.

        Args:
            num_classes (int): Number of segmentation classes
            backbone (str, optional): HuggingFace model name or local path.
                Defaults to `"nvidia/segformer-b0-finetuned-ade-512-512"`.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to `True`.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
        """

        super().__init__()

        self.num_classes = num_classes + 1  # +1 for background class
        self.ignore_index = ignore_index

        if pretrained:  # pragma: no cover
            self.model = SegformerForSemanticSegmentation.from_pretrained(backbone)
            if self.model.config.num_labels != self.num_classes:
                self.model.config.num_labels = self.num_classes
                self._reinit_classifier()
        else:
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = self.num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def _reinit_classifier(self) -> None:  # pragma: no cover
        """Reinitialize the classifier for new number of classes."""

        self.model.decode_head.classifier = nn.Conv2d(
            self.model.decode_head.classifier.in_channels,
            self.num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].
            labels (torch.Tensor | None, optional): Ground truth masks [B, H, W] for training.
                Defaults to `None`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing logits and loss (if training).
        """

        return self.model(pixel_values=pixel_values, labels=labels)

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
            logits = outputs.logits
            logits = nn.functional.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return torch.argmax(logits, dim=1)
