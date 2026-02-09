import torch
from torch import Tensor
from plyfile import PlyData
import numpy as np


class GaussianModel:
    """3DGS model loaded from a standard PLY file."""

    def __init__(self):
        self._means = None
        self._quaternions = None
        self._scales = None
        self._opacities = None
        self._colors = None       # f_dc [N, 3]
        self._sh_rest = None      # f_rest [N, num_rest, 3] or None

    def load_ply(self, path: str):
        ply = PlyData.read(path)
        el = ply.elements[0]

        # positions
        xyz = np.stack([el.data['x'], el.data['y'], el.data['z']], axis=1)
        self._means = torch.tensor(xyz, dtype=torch.float32, device='cuda')

        # rotations (w,x,y,z)
        rots = np.stack([el.data['rot_0'], el.data['rot_1'],
                         el.data['rot_2'], el.data['rot_3']], axis=1)
        self._quaternions = torch.tensor(rots, dtype=torch.float32, device='cuda')

        # scales (stored as log-scale in standard 3DGS PLY)
        scales = np.stack([el.data['scale_0'], el.data['scale_1'],
                           el.data['scale_2']], axis=1)
        self._scales = torch.tensor(scales, dtype=torch.float32, device='cuda')

        # opacity (stored as logit in standard 3DGS PLY)
        opacity = el.data['opacity'].reshape(-1, 1)
        self._opacities = torch.tensor(opacity, dtype=torch.float32, device='cuda')

        # DC color coefficients
        f_dc = np.stack([el.data['f_dc_0'], el.data['f_dc_1'],
                         el.data['f_dc_2']], axis=1)
        self._colors = torch.tensor(f_dc, dtype=torch.float32, device='cuda')

        # SH rest coefficients
        prop_names = [p.name for p in el.properties]
        rest_names = sorted([n for n in prop_names if n.startswith('f_rest_')],
                            key=lambda x: int(x.split('_')[-1]))
        if rest_names:
            sh_rest = np.stack([el.data[n] for n in rest_names], axis=1)  # [N, num_rest]
            # reshape to [N, num_rest//3, 3]
            num_rest = len(rest_names)
            sh_rest = sh_rest.reshape(-1, num_rest // 3, 3)
            self._sh_rest = torch.tensor(sh_rest, dtype=torch.float32, device='cuda')

    @property
    def num_gaussians(self) -> int:
        return self._means.shape[0]

    @property
    def means(self) -> Tensor:
        return self._means

    @property
    def quaternions(self) -> Tensor:
        return self._quaternions

    @property
    def scales(self) -> Tensor:
        return self._scales

    @property
    def opacities(self) -> Tensor:
        return self._opacities

    @property
    def colors(self) -> Tensor:
        return self._colors

    @property
    def sh_rest(self):
        return self._sh_rest

    def get_sh_coeffs(self) -> Tensor:
        """Return full SH coefficients [N, K, 3] for gsplat rendering.
        K = 1 + num_rest_bands (e.g., 1+15=16 for degree 3).
        """
        dc = self._colors.unsqueeze(1)  # [N, 1, 3]
        if self._sh_rest is not None:
            return torch.cat([dc, self._sh_rest], dim=1)  # [N, K, 3]
        return dc

    def get_activated_opacities(self) -> Tensor:
        """Apply sigmoid to get opacities in [0,1]."""
        return torch.sigmoid(self._opacities).squeeze(-1)  # [N]

    def get_activated_scales(self) -> Tensor:
        """Apply exp to get positive scales."""
        return torch.exp(self._scales)  # [N, 3]

    def apply_rotation(self, rot_x_deg: float = 0, rot_y_deg: float = 0, rot_z_deg: float = 0):
        """Apply extrinsic XYZ rotation to the scene (means + quaternions).
        Rotates around the scene center. Angles in degrees."""
        if rot_x_deg == 0 and rot_y_deg == 0 and rot_z_deg == 0:
            return

        device = self._means.device

        # Build rotation matrix (extrinsic XYZ = intrinsic ZYX)
        def _rot_x(a):
            c, s = np.cos(a), np.sin(a)
            return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32)
        def _rot_y(a):
            c, s = np.cos(a), np.sin(a)
            return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32)
        def _rot_z(a):
            c, s = np.cos(a), np.sin(a)
            return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32)

        R = (_rot_z(np.radians(rot_z_deg))
             @ _rot_y(np.radians(rot_y_deg))
             @ _rot_x(np.radians(rot_x_deg))).to(device)

        # Rotate means around scene center
        center = self._means.mean(dim=0)
        self._means = (R @ (self._means - center).T).T + center

        # Convert rotation matrix to quaternion (w,x,y,z)
        # Using Shepperd's method
        m = R
        tr = m[0,0] + m[1,1] + m[2,2]
        if tr > 0:
            s = torch.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (m[2,1] - m[1,2]) / s
            qy = (m[0,2] - m[2,0]) / s
            qz = (m[1,0] - m[0,1]) / s
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = torch.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            qw = (m[2,1] - m[1,2]) / s
            qx = 0.25 * s
            qy = (m[0,1] + m[1,0]) / s
            qz = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = torch.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            qw = (m[0,2] - m[2,0]) / s
            qx = (m[0,1] + m[1,0]) / s
            qy = 0.25 * s
            qz = (m[1,2] + m[2,1]) / s
        else:
            s = torch.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            qw = (m[1,0] - m[0,1]) / s
            qx = (m[0,2] + m[2,0]) / s
            qy = (m[1,2] + m[2,1]) / s
            qz = 0.25 * s
        q_rot = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=device)
        q_rot = q_rot / q_rot.norm()

        # Apply rotation to each gaussian's quaternion: q_new = q_rot * q_old
        # Hamilton product (w,x,y,z)
        w0, x0, y0, z0 = q_rot
        w1 = self._quaternions[:, 0]
        x1 = self._quaternions[:, 1]
        y1 = self._quaternions[:, 2]
        z1 = self._quaternions[:, 3]
        self._quaternions = torch.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ], dim=-1)

    def to(self, device):
        """Move all tensors to the specified device."""
        self._means = self._means.to(device)
        self._quaternions = self._quaternions.to(device)
        self._scales = self._scales.to(device)
        self._opacities = self._opacities.to(device)
        self._colors = self._colors.to(device)
        if self._sh_rest is not None:
            self._sh_rest = self._sh_rest.to(device)
        return self
