import torch
from torch import Tensor


def normalize_quaternion(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize quaternion to unit length. (w,x,y,z) format."""
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quaternion_identity(batch_size: int, device: str = 'cuda') -> Tensor:
    """Return identity quaternions [1,0,0,0] of shape [B, 4]."""
    q = torch.zeros(batch_size, 4, device=device)
    q[:, 0] = 1.0
    return q


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions. (w,x,y,z) format.
    Shapes: q1 [*, 4], q2 [*, 4] -> [*, 4]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion to rotation matrix.
    q: [*, 4] (w,x,y,z) -> R: [*, 3, 3]
    """
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quaternion_apply(q: Tensor, v: Tensor) -> Tensor:
    """Apply quaternion rotation to 3D vectors.
    q: [*, 4], v: [*, 3] -> [*, 3]
    """
    R = quaternion_to_rotation_matrix(q)
    return torch.einsum('...ij,...j->...i', R, v)
