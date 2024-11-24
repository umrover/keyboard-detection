import torch
import numpy as np


def prediction_to_projection_matrix(pred):
    global ones, zeros

    batch_size = len(pred)
    if len(ones) != batch_size:
        ones = torch.ones(batch_size, dtype=torch.float32, device=device)
        zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)

    alpha = pred[:, 0]
    beta = pred[:, 1]
    gamma = pred[:, 2]
    position = pred[:, 3:]

    return projection_matrix(alpha, beta, gamma, position)


def project_point(P, v):
    """
    [x', y', z'] = P @ [x, y, 1]
    return [x' / z', y' / z']
    """
    v = P @ v
    return (v[:, :2] / v[:, -1:]).T


def prediction_to_corners(pred):
    P = prediction_to_projection_matrix(pred)

    return (project_point(P, P1),
            project_point(P, P2),
            project_point(P, P3),
            project_point(P, P4))


def xy_to_uv_matrix(p1, p2, p3, p4):
    global ones, zeros

    """
    M = [[x1, x2, x3],
         [y1, y2, y3],
         [ 1,  1,  1]]
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    batch_size = len(x1)
    if len(ones) != batch_size:
        ones = torch.ones(batch_size, dtype=torch.float32, device=device)
        zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)

    M1 = torch.stack([x1, x2, x3], dim=1)
    M2 = torch.stack([y1, y2, y3], dim=1)
    M3 = torch.stack([ones, ones, ones], dim=1)
    M = torch.stack((M1, M2, M3), dim=1)

    M_inv = torch.inverse(M)

    """
    X = [x4, y4, 1]
    """
    X = torch.stack([x4, y4, ones], dim=1)
    X = X.unsqueeze(dim=-1)

    """
    H = M @ X = [λ, μ, τ]
    """
    H = M_inv @ X
    H = H.squeeze(dim=-1)

    """
    A = [[λ * x1, μ * x2, τ * x3],
         [λ * y1, μ * y2, τ * y3],
         [     λ,      μ,     τ]]
    """
    A = torch.einsum('bij, bj -> bij', M, H)
    A_inv = torch.inverse(A)

    """
    C = B @ A_inv
    """
    return B @ A_inv


def corners_to_texture_coordinates(corners):
    C = xy_to_uv_matrix(*corners)

    """
    [u, v, z] = C @ [x, y, 1]
    result = [u / z, y / z]
    """
    result = torch.einsum('bij, yxj -> byxi', C, coordinates)
    result = result[:, :, :, :3] / result[:, :, :, -1:]

    # filter any points outside the keyboard
    result[(result.max(dim=-1)[0] > 255)] = 0
    result[result.min(dim=-1)[0] < 0] = 0

    # reorder dimensions from (batch, y, x, channel) to (batch, channel, y, x)
    result = torch.einsum('byxc -> bcyx', result)
    return (result / 128) - 1


def prediction_to_texture_coordinates(pred):
    corners = prediction_to_corners(pred)
    return corners_to_texture_coordinates(corners)


def xy_to_uv_change_of_basis_matrix():
    """
    M = [[x1, x2, x3],
         [y1, y2, y3],
         [ 1,  1,  1]]

    X = [x4, y4, 1]
    """
    M = np.array([[0, 255, 255],
                  [0, 0, 255],
                  [1, 1, 1]])
    M_inv = np.linalg.inv(M)

    X = np.array([0, 255, 1])

    """
    H = M @ X = [λ, μ, τ]
    """
    H = M_inv @ X

    """
    B = [[λ * x1, μ * x2, τ * x3],
         [λ * y1, μ * y2, τ * y3],
         [     λ,      μ,     τ]]
    """
    return torch.tensor(M * H, dtype=torch.float32, device=device)


def color_coordinates_mesh():
    x = np.linspace(0, 640, 320)
    y = np.linspace(0, 480, 240)
    xx, yy = np.meshgrid(x, y)
    one = np.ones(xx.shape)

    mesh = np.stack((xx, yy, one), axis=-1)
    return torch.tensor(mesh, dtype=torch.float32, device=device)


B = xy_to_uv_change_of_basis_matrix()

width = 2.7986
height = 0.95583

coordinates = color_coordinates_mesh()

X = -0.21919
Y = -0.081833
Z = 0.13053

ones = []
zeros = []

P1 = torch.tensor([-width / 2 + X, -height / 2 + Y, Z, 1], dtype=torch.float32, device=device)
P2 = torch.tensor([width / 2 + X, -height / 2 + Y, Z, 1], dtype=torch.float32, device=device)
P3 = torch.tensor([width / 2 + X, height / 2 + Y, Z, 1], dtype=torch.float32, device=device)
P4 = torch.tensor([-width / 2 + X, height / 2 + Y, Z, 1], dtype=torch.float32, device=device)
