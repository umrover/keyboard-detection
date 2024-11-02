import torch
import lightning as pl

import segmentation_models_pytorch as smp


class BinarySegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, loss: str = "Jaccard", **kwargs):
        super().__init__()

        self.model = smp.create_model(arch, encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)

        if loss == "Jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE)
        elif loss == "Dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        else:
            raise ValueError("Unknown Loss Function")

        self.learning_rate = lr
        self.lr = self.learning_rate
        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image)

    def predict(self, image):
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self(image).cpu().numpy()

        if len(pred) == 1:
            return pred[0, 0]
        return pred[:, 0]

    def _step(self, batch, stage):
        image, truth = batch
        prediction = self.forward(image)

        loss = self.loss_fn(prediction, truth.int())
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=(self.lr or self.learning_rate))
        return {"optimizer": optimizer}


__all__ = ["BinarySegmentationModel"]
