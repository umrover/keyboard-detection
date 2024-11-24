import torch


class BatchedLinearAlgebra:
    def __init__(self, batch_size: int, device):
        self.batch_size = batch_size
        self.device = device

        self.ones = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        self.zeros = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)


__all__ = ["BatchedLinearAlgebra"]
