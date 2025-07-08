from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision


class InstanceSegmentationModule(L.LightningModule):
    """LightningModule for instance segmentation tasks."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        compile: bool = False,
    ) -> None:
        """Initialize instance segmentation module.

        Args:
            net (nn.Module): Instance segmentation model (e.g., Mask2FormerInstance).
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler | None, optional): Learning rate scheduler.
                Defaults to `None`.
            compile (bool, optional): Whether to compile the model. Defaults to `False`.
        """

        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # Also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # Save it here, as `torch.compile` resets the class name
        self.model_name = self.net.__class__.__name__

        # Compile model for faster training
        if compile:
            self.net = torch.compile(self.net)

        # Initialize metrics
        self.train_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            class_metrics=True,
        )
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            class_metrics=True,
        )
        self.test_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            class_metrics=True,
        )

        # For tracking best validation mAP
        self.val_map_best = 0.0

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

        self.train_map.reset()
        self.val_map.reset()
        self.val_map_best = 0.0

    def model_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Perform a single model step on a batch of data.

        Args:
            batch (dict[str, Any]): A batch of data containing images and targets.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: A tuple containing:
                - loss (torch.Tensor): The loss tensor.
                - preds (dict[str, Any]): The predictions dictionary.
        """

        images = batch["image"]
        targets = batch["target"]

        # Prepare labels for Mask2Former
        if self.model_name == "Mask2FormerInstance":
            # Format labels for Mask2Former training
            mask_labels = []
            class_labels = []
            for target in targets:
                masks = target["masks"]  # [N, H, W]
                labels = target["labels"]  # [N]
                # Convert masks to proper format for Mask2Former
                # Mask2Former expects masks as [N, H, W] with values 0/1
                processed_masks = masks.float()  # Convert to long tensor
                processed_labels = labels.long()  # Ensure labels are long
                mask_labels.append(processed_masks)
                class_labels.append(processed_labels)

            outputs = self.net(images, mask_labels=mask_labels, class_labels=class_labels)
        else:  # for other models (if added in the future)
            outputs = self.net(images, targets)

        loss = outputs["loss"]

        return loss, outputs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and targets.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between model predictions and targets.
        """

        loss, _ = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and targets.
            batch_idx (int): The index of the current batch.
        """

        images = batch["image"]
        targets = batch["target"]
        if self.model_name == "Mask2FormerInstance":
            predictions = self.net.predict(images)
        else:  # for other models
            with torch.no_grad():
                outputs = self.net(images)
                predictions = outputs

        # Convert predictions to format expected by torchmetrics
        preds = []
        targets_formatted = []
        for pred, target in zip(predictions, targets, strict=False):
            # Convert Mask2Former predictions to evaluation format
            if self.model_name == "Mask2FormerInstance":
                pred_dict = self._convert_mask2former_prediction(pred, images.shape[-2:])
            else:  # for other models
                pred_dict = pred

            preds.append(pred_dict)

            # Format target
            target_dict = {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "masks": target["masks"].bool(),
            }
            targets_formatted.append(target_dict)

        # Update metrics
        self.val_map.update(preds, targets_formatted)

    def _convert_mask2former_prediction(self, pred: dict, image_shape: tuple[int, int]) -> dict:
        """Convert Mask2Former prediction to evaluation format.

        Args:
            pred (dict): Predictions dictionary.
            image_shape (tuple[int, int]): Image shape.

        Returns:
            dict: Converted Mask2Former predictions.
        """

        if "segmentation" not in pred or not pred.get("segments_info"):
            # Return empty prediction if no valid segmentation
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=self.device),
                "masks": torch.zeros((0, *image_shape), dtype=torch.bool, device=self.device),
            }

        seg_map = pred["segmentation"]
        segments_info = pred["segments_info"]

        boxes, scores, labels, masks = [], [], [], []
        for segment in segments_info:
            # Extract mask for this segment
            mask = (seg_map == segment["id"]).float()
            if mask.sum() > 0:
                # Calculate bounding box from mask
                y_indices, x_indices = torch.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min().float(), x_indices.max().float()
                    y_min, y_max = y_indices.min().float(), y_indices.max().float()
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(segment.get("score", 0.5))
                    labels.append(segment.get("label_id", segment.get("category_id", 0)))
                    masks.append(mask.bool())

        if boxes:
            return {
                "boxes": torch.stack(
                    [torch.tensor(box, dtype=torch.float32, device=self.device) for box in boxes]
                ),
                "scores": torch.tensor(scores, dtype=torch.float32, device=self.device),
                "labels": torch.tensor(labels, dtype=torch.int64, device=self.device),
                "masks": torch.stack(masks),
            }
        else:
            # Return empty prediction if no valid segments
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=self.device),
                "masks": torch.zeros((0, *image_shape), dtype=torch.bool, device=self.device),
            }

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch. Aggregates validation metrics and logs them."""

        val_map_dict = self.val_map.compute()
        self.log("val/mAP", val_map_dict["map"], prog_bar=True)
        self.log("val/mAP_50", val_map_dict["map_50"], prog_bar=True)
        self.log("val/mAP_75", val_map_dict["map_75"])

        # Log per-class mAP, if available
        if "map_per_class" in val_map_dict:
            map_per_class = val_map_dict["map_per_class"]
            # Check if map_per_class is a tensor with more than 0 dimensions
            if map_per_class.dim() > 0:
                for class_idx, class_map in enumerate(map_per_class):
                    if not torch.isnan(class_map):  # Only log valid values
                        self.log(f"val/mAP_class_{class_idx}", class_map)
            # If it's a 0-d tensor (scalar), we can still log it as a single value
            elif not torch.isnan(map_per_class):
                self.log("val/mAP_class_0", map_per_class)

        # Track best validation mAP
        current_map = val_map_dict["map"]
        if current_map > self.val_map_best:
            self.val_map_best = current_map
        self.log("val/mAP_best", self.val_map_best, prog_bar=True)

        # Reset metrics
        self.val_map.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (dict[str, Any]): A batch of data containing images and targets.
            batch_idx (int): The index of the current batch.
        """

        images = batch["image"]
        targets = batch["target"]

        if self.model_name == "Mask2FormerInstance":
            predictions = self.net.predict(images)
        else:  # for other models
            with torch.no_grad():
                outputs = self.net(images)
                predictions = outputs

        # Convert predictions to format expected by torchmetrics
        preds = []
        targets_formatted = []
        for pred, target in zip(predictions, targets, strict=False):
            if self.model_name == "Mask2FormerInstance":
                pred_dict = self._convert_mask2former_prediction(pred, images.shape[-2:])
            else:
                pred_dict = pred

            preds.append(pred_dict)

            # Format target
            target_dict = {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "masks": target["masks"].bool(),
            }
            targets_formatted.append(target_dict)

        # Update metrics
        self.test_map.update(preds, targets_formatted)

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch. Aggregates validation metrics and logs them."""

        test_map_dict = self.test_map.compute()
        self.log("test/mAP", test_map_dict["map"])
        self.log("test/mAP_50", test_map_dict["map_50"])
        self.log("test/mAP_75", test_map_dict["map_75"])

        # Log per-class mAP, if available
        if "map_per_class" in test_map_dict:
            map_per_class = test_map_dict["map_per_class"]
            # Check if map_per_class is a tensor with more than 0 dimensions
            if map_per_class.dim() > 0:
                for i, class_map in enumerate(map_per_class):
                    if not torch.isnan(class_map):  # Only log valid values
                        self.log(f"test/mAP_class_{i}", class_map)
            # If it's a 0-d tensor (scalar), we can still log it as a single value
            elif not torch.isnan(map_per_class):
                self.log("test/mAP_class_0", map_per_class)

        # Reset metrics
        self.test_map.reset()

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
                    "monitor": "val/mAP",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
