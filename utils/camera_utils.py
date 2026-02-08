import torch
import numpy as np
from torch import Tensor


def orbit_camera(elevation_deg: float, azimuth_deg: float, radius: float,
                 target: Tensor = None) -> Tensor:
    """Generate orbit camera view matrix (world2cam, 4x4)."""
    if target is None:
        target = torch.zeros(3)

    ele = np.radians(elevation_deg)
    azi = np.radians(azimuth_deg)

    # Camera position in world space
    x = radius * np.cos(ele) * np.sin(azi)
    y = -radius * np.sin(ele)
    z = radius * np.cos(ele) * np.cos(azi)
    campos = torch.tensor([x, y, z], dtype=torch.float32) + target

    # Look-at matrix (OpenCV convention: camera looks down +Z)
    forward = target - campos
    forward = forward / forward.norm()
    # Scene uses Y-down convention, so world-up is -Y
    up_hint = torch.tensor([0.0, -1.0, 0.0])

    right = torch.linalg.cross(forward, up_hint)
    if right.norm() < 1e-6:
        up_hint = torch.tensor([0.0, 0.0, 1.0])
        right = torch.linalg.cross(forward, up_hint)
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    up = up / up.norm()

    # world2cam (OpenCV: x-right, y-down, z-forward)
    # up now points in world-up (-Y), negate for OpenCV y-down (+Y)
    down = -up
    R = torch.stack([right, down, forward], dim=0)  # [3, 3]
    t = -R @ campos  # [3]

    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def random_camera(elevation_range=(-30, 30), azimuth_range=(0, 360),
                  radius_range=(2.0, 4.0), target=None):
    """Randomly sample a camera pose."""
    ele = np.random.uniform(*elevation_range)
    azi = np.random.uniform(*azimuth_range)
    rad = np.random.uniform(*radius_range)
    return orbit_camera(ele, azi, rad, target)


def get_projection_matrix(fovy_deg: float, aspect: float,
                          near: float = 0.01, far: float = 100.0) -> Tensor:
    """Perspective projection matrix."""
    fovy = np.radians(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1.0
    return P


def get_intrinsics(fovy_deg: float, width: int, height: int):
    """Return fx, fy, cx, cy for perspective camera."""
    fovy = np.radians(fovy_deg)
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy
