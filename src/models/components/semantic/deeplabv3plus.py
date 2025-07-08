import torch
import torch.nn as nn
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss


class DeepLabV3PlusSemantic(nn.Module):
    """DeepLabV3+ model for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        activation: str | None = None,
        ignore_index: int = 255,
    ) -> None:
        """Initialize DeepLabV3+ model.

        Args:
            num_classes (int): Number of segmentation classes.
            encoder_name (int, optional): Encoder backbone name. Defaults to `"resnet50"`.
            encoder_weights (str, optional): Pretrained weights for encoder. Defaults to `"imagenet"`.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            activation (str | None, optional): Activation function for output. Defaults to `None`.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
        """
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.model = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.focal_loss = FocalLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Input images [B, C, H, W].
            labels (torch.Tensor | None): Ground truth masks [B, H, W] for training. Defaults to `None`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing logits and loss (if training).
        """

        logits = self.model(pixel_values)
        outputs = {"logits": logits}
        if labels is not None:
            ce_loss = self.ce_loss(logits, labels)
            dice_loss = self.dice_loss(logits, labels)
            focal_loss = self.focal_loss(logits, labels)
            total_loss = 0.5 * ce_loss + 0.3 * dice_loss + 0.2 * focal_loss
            outputs["loss"] = total_loss
            outputs["ce_loss"] = ce_loss
            outputs["dice_loss"] = dice_loss
            outputs["focal_loss"] = focal_loss

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
            logits = outputs["logits"]
            return torch.argmax(logits, dim=1)
