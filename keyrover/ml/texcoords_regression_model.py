from typing import Literal

import numpy as np
import cv2

import torch
import lightning as pl

import segmentation_models_pytorch as smp


class TexCoordsRegressionModel(pl.LightningModule):
    def __init__(self,
                 arch: Literal["unet", "unetplusplus"],
                 encoder_name: str,
                 in_channels: int, out_classes: int, lr: float,
                 **kwargs) -> None:
        super().__init__()

        self.model = smp.create_model(arch, encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        self.loss_fn = torch.nn.MSELoss()

        self.learning_rate = lr
        self.lr = self.learning_rate
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def predict(self, image: torch.Tensor) -> np.ndarray:
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self(image).cpu().numpy()

        if len(pred) == 1:
            pred = pred[0,]

        return (pred + 1) / 2  # Tanh outputs [-1, 1], we want [0, 1]

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> float:
        image, truth = batch
        prediction = self.forward(image)

        loss = self.loss_fn(prediction, truth)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=(self.lr or self.learning_rate))
        return {"optimizer": optimizer}


__all__ = ["TexCoordsRegressionModel"]
