import torch

import numpy as np

from .linalg import BatchedLinearAlgebra, TensorType
from .projection import BatchedProjectionMatrix
from .homogenous import HomogenousChangeOfBasis

Vec3 = tuple[float, float, float]


class InverseBilinear(BatchedLinearAlgebra):
    def __init__(self, batch_size: int, device, width: int, height: int, p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3):
        super().__init__(batch_size, device)

        self.projection_matrix = BatchedProjectionMatrix(batch_size, device)

        # X = -0.21919
        # Y = -0.081833
        # Z = 0.13053

        # width = 2.7986
        # height = 0.95583

        self.coordinates = self._coordinates_mesh(width=width, height=height, resolution=2)

        self.p1 = torch.tensor([*p1, 1], dtype=torch.float32, device=self.device)
        self.p2 = torch.tensor([*p2, 1], dtype=torch.float32, device=self.device)
        self.p3 = torch.tensor([*p3, 1], dtype=torch.float32, device=self.device)
        self.p4 = torch.tensor([*p4, 1], dtype=torch.float32, device=self.device)
        self.change_of_basis = HomogenousChangeOfBasis(batch_size, device)

        uv1 = torch.tensor([[0], [0]], dtype=torch.float32)
        uv2 = torch.tensor([[255], [0]], dtype=torch.float32)
        uv3 = torch.tensor([[255], [255]], dtype=torch.float32)
        uv4 = torch.tensor([[0], [255]], dtype=torch.float32)

        self.UV_basis = self.change_of_basis(uv1, uv2, uv3, uv4)

    def __call__(self, alpha: TensorType, beta: TensorType, gamma: TensorType, position: TensorType) -> torch.Tensor:
        corners = self.project_corners(alpha, beta, gamma, position)

        XY_basis = self.change_of_basis(*corners)
        C = self.UV_basis @ torch.inverse(XY_basis)

        """
        [u, v, z] = C @ [x, y, 1]
        result = [u / z, y / z]
         """
        result = torch.einsum('bij, yxj -> byxi', C, self.coordinates)
        result = result[:, :, :, :3] / result[:, :, :, -1:]

        # filter any points outside the keyboard
        result[(result.max(dim=-1)[0] > 255)] = 0
        result[result.min(dim=-1)[0] < 0] = 0

        # reorder dimensions from (batch, y, x, channel) to (batch, channel, y, x)
        result = torch.einsum('byxc -> bcyx', result)
        return (result / 128) - 1

    def project_corners(self, alpha: TensorType, beta: TensorType, gamma: TensorType,
                        position: TensorType) -> tuple[torch.Tensor, ...]:
        projection = self.projection_matrix(alpha, beta, gamma, position)
        return projection(self.p1), projection(self.p2), projection(self.p3), projection(self.p4)

    def _coordinates_mesh(self, width: int, height: int, resolution: float) -> torch.Tensor:
        x = np.linspace(0, width, int(width / resolution))
        y = np.linspace(0, height, int(height // resolution))
        xx, yy = np.meshgrid(x, y)
        ones = np.ones(xx.shape)

        mesh = np.stack((xx, yy, ones), axis=-1)
        return torch.tensor(mesh, dtype=torch.float32, device=self.device)


__all__ = ['InverseBilinear']
