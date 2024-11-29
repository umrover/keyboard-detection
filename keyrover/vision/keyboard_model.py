import torch
import lightning as pl


class KeyboardModel(pl.LightningModule):
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> float:
        raise NotImplementedError()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}


__all__ = ["KeyboardModel"]
