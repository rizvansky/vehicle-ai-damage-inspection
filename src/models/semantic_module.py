from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, JaccardIndex, MetricCollection


class SemanticSegmentationModule(L.LightningModule):
    """LightningModule for semantic segmentation tasks."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        compile: bool = False,
        num_classes: int = 8,
        ignore_index: int = 255,
    ) -> None:
        """Initialize semantic segmentation module.

        Args:
            net (nn.Module): Semantic segmentation model (DeepLabV3Plus, Mask2Former, SegFormer, or UNet).
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler | None, optional): Learning rate scheduler.
                Defaults to `None`.
            compile (bool, optional): Whether to compile the model. Defaults to `False`.
            num_classes (int, optional): Number of segmentation classes. Defaults to 8.
            ignore_index (int, optional): Index to ignore in loss computation. Defaults to 255.
        """

        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # Also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Save it here, as `torch.compile` resets the class name
        self.model_name = self.net.__class__.__name__

        # Compile model for faster training
        if compile:
            self.net = torch.compile(self.net)

        # Initialize metrics
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "accuracy_per_class": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="none",
                ),
                "jaccard": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "jaccard_per_class": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="none",
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # For tracking best validation mIoU
        self.val_jaccard_best = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """

        return self.net(x)

    def on_train_start(self) -> None:
        """Called at the beginning of training. Resets metrics objects."""

        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_jaccard_best = 0.0

    def _prepare_mask2former_labels(self, masks: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Prepare labels for Mask2Former training.

        Convert semantic masks from format [B, H, W] (where each pixel contains class ID)
        to Mask2Former format: list of dicts with binary masks and class labels.

        Args:
            masks (torch.Tensor): Ground truth semantic masks [B, H, W] with class IDs

        Returns:
            list[dict[str, torch.Tensor]]: List of label dictionaries for each image in batch.
                Each dict contains:
                - 'masks': torch.Tensor of shape [num_instances, H, W] - binary masks
                - 'labels': torch.Tensor of shape [num_instances] - class indices
        """

        batch_size, height, width = masks.shape
        labels = []

        for batch_idx in range(batch_size):
            mask = masks[batch_idx]  # [H, W]

            # Get unique classes present in this mask (excluding ignore_index)
            unique_classes = torch.unique(mask)
            unique_classes = unique_classes[unique_classes != self.ignore_index]
            unique_classes = unique_classes[unique_classes < self.num_classes]
            if len(unique_classes) == 0:
                # Handle case where no valid classes exist
                labels.append(
                    {
                        "masks": torch.zeros(
                            (1, height, width), dtype=torch.float32, device=masks.device
                        ),
                        "labels": torch.tensor([0], dtype=torch.long, device=masks.device),
                    }
                )
            else:
                # Create binary masks for each class
                class_masks = []
                class_labels = []
                for class_id in unique_classes:
                    # Create binary mask for this class
                    binary_mask = (mask == class_id).float()
                    if binary_mask.sum() > 0:  # Only include non-empty masks
                        class_masks.append(binary_mask)
                        class_labels.append(class_id)

                if class_masks:
                    labels.append(
                        {
                            "masks": torch.stack(class_masks),  # [num_instances, H, W]
                            "labels": torch.tensor(
                                class_labels, dtype=torch.long, device=masks.device
                            ),
                        }
                    )
                else:
                    # Fallback if all masks are empty
                    labels.append(
                        {
                            "masks": torch.zeros(
                                (1, height, width), dtype=torch.float32, device=masks.device
                            ),
                            "labels": torch.tensor([0], dtype=torch.long, device=masks.device),
                        }
                    )

        return labels

    def model_step(
        self, batch: dict[str, Any], compute_predictions: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch (dict[str, Any]): A batch of data containing images and masks.
            compute_predictions (bool): Whether to compute predictions. Set to `False` during training for efficiency.
                Defaults to `True`.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]: A tuple containing:
                - loss (torch.Tensor): The loss tensor.
                - preds (torch.Tensor | None): The predictions tensor (None if not computed).
                - targets (torch.Tensor): The targets tensor.
        """

        images = batch["image"]
        masks = batch["mask"]

        # Forward pass - handle different model types
        if self.model_name in ("DeepLabV3PlusSemantic", "SegFormerSemantic", "UNetSemantic"):
            outputs = self.net(images, masks)
            loss = outputs["loss"]
            if compute_predictions:
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1)
            else:
                preds = None

        elif self.model_name == "Mask2FormerSemantic":
            if self.training:
                # Prepare labels in the format expected by Mask2Former
                mask_labels = self._prepare_mask2former_labels(masks)
                outputs = self.net(images, mask_labels=mask_labels)
                loss = outputs["loss"]

                # Only compute predictions if requested (e.g., for validation)
                if compute_predictions:
                    with torch.no_grad():
                        preds = self.net.predict(images)
                else:
                    preds = None
            else:
                outputs = self.net(images)
                loss = torch.tensor(0.0, device=self.device)
                if compute_predictions:
                    preds = self.net.predict(images)
                else:
                    preds = None

        else:
            # Generic handling for other models
            outputs = self.net(images, masks)
            if isinstance(outputs, dict):
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
                if compute_predictions and "logits" in outputs:
                    preds = torch.argmax(outputs["logits"], dim=1)
                else:
                    preds = None
            else:
                # If outputs is just a tensor (logits)
                loss = torch.tensor(0.0, device=self.device)
                if compute_predictions:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = None

        return loss, preds, masks

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and masks.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between model predictions and targets.
        """

        loss, _, _ = self.model_step(batch, compute_predictions=False)

        # Skip metric computation during training for efficiency
        # Metrics will be computed during validation
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""
        # Since we're not computing training metrics for efficiency,
        # we only log the loss (which is already logged in training_step)
        pass

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and masks.
            batch_idx (int): The index of the current batch.
        """

        # During validation, we need predictions for metrics
        loss, preds, targets = self.model_step(batch, compute_predictions=True)

        # Only update metrics if we have valid predictions
        if preds is not None and preds.numel() > 0 and targets.numel() > 0:
            self.val_metrics.update(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""

        val_metrics_dict = self.val_metrics.compute()
        self.log("val/accuracy", val_metrics_dict["val/accuracy"], prog_bar=True)
        self.log("val/jaccard", val_metrics_dict["val/jaccard"], prog_bar=True)

        # Log per-class metrics
        for class_idx, (acc, jaccard) in enumerate(
            zip(
                val_metrics_dict["val/accuracy_per_class"],
                val_metrics_dict["val/jaccard_per_class"],
                strict=False,
            )
        ):
            self.log(f"val/accuracy_class_{class_idx}", acc)
            self.log(f"val/jaccard_class_{class_idx}", jaccard)

        # Track best validation mIoU
        current_jaccard = val_metrics_dict["val/jaccard"]
        if current_jaccard > self.val_jaccard_best:
            self.val_jaccard_best = current_jaccard

        self.log("val/jaccard_best", self.val_jaccard_best, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and masks.
            batch_idx (int): The index of the current batch.
        """

        # During testing, we need predictions for metrics
        loss, preds, targets = self.model_step(batch, compute_predictions=True)

        # Only update metrics if we have valid predictions
        if preds is not None and preds.numel() > 0 and targets.numel() > 0:
            self.test_metrics.update(preds, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch."""

        test_metrics_dict = self.test_metrics.compute()
        self.log("test/accuracy", test_metrics_dict["test/accuracy"])
        self.log("test/jaccard", test_metrics_dict["test/jaccard"])

        # Log per-class metrics
        for class_idx, (acc, jaccard) in enumerate(
            zip(
                test_metrics_dict["test/accuracy_per_class"],
                test_metrics_dict["test/jaccard_per_class"],
                strict=False,
            )
        ):
            self.log(f"test/accuracy_class_{class_idx}", acc)
            self.log(f"test/jaccard_class_{class_idx}", jaccard)

        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning-rate schedulers to be used for training.

        Returns:
            dict[str, Any]: A dict containing the configured optimizers and LR schedulers to be used for training.
        """

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/jaccard",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
