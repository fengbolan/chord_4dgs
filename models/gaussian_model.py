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
