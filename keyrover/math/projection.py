import torch
from torch import Tensor

from .linalg import BatchedLinearAlgebra
from .rotation import RotationMatrix


class ProjectionMatrix(BatchedLinearAlgebra):
    def __init__(self, batch_size: int, device):
        super().__init__(batch_size, device)

        fx = 888.889  # TODO calculation of fx, fy from blender camera data
        fy = 1000
        self.K = torch.tensor([[fx, 0, 320],
                               [0, fy, 240],
                               [0, 0, 1]], dtype=torch.float32, device=self.device)

        self.R_to_Blender = torch.tensor([[1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, -1]], dtype=torch.float32, device=self.device)

        self.rotation_matrix = RotationMatrix(batch_size, device)

    def __call__(self, alpha: Tensor, beta: Tensor, gamma: Tensor, position: Tensor) -> Tensor:
        return self.projection_matrix(alpha, beta, gamma, position)

    def extrinsic_matrix(self, alpha, beta, gamma, position):
        """
        position = [x, y, z]

        E = [R | -R @ position]
        """
        R = self.rotation_matrix(alpha, beta, gamma)

        position = position.unsqueeze(axis=-1)
        T = -R @ position

        R = self.R_to_Blender @ R
        T = self.R_to_Blender @ T
        return torch.concat([R, T], dim=-1)

    def intrinsic_matrix(self, fx, fy, cx, cy):
        """
        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
        """
        K1 = torch.stack([fx, self.zeros, cx], dim=1)
        K2 = torch.stack([self.zeros, fy, cy], dim=1)
        K3 = torch.stack([self.zeros, self.zeros, self.ones], dim=1)
        return torch.stack([K1, K2, K3], dim=1)

    def projection_matrix(self, alpha: Tensor, beta: Tensor, gamma: Tensor, position: Tensor) -> Tensor:
        """
        P = K @ E
        """
        E = self.extrinsic_matrix(alpha, beta, gamma, position)
        return self.K @ E


__all__ = ["ProjectionMatrix"]
