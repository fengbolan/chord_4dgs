import torch
import torch.nn as nn
from torch import Tensor
from utils.math_utils import normalize_quaternion


class FenwickTreeDeformation(nn.Module):
    def __init__(self, num_control_points: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.K = num_control_points
        # Fenwick tree uses 1-based indexing, allocate T+1 nodes
        self.node_translations = nn.Parameter(
            torch.zeros(num_frames + 1, num_control_points, 3))
        rots = torch.zeros(num_frames + 1, num_control_points, 4)
        rots[..., 0] = 1.0  # identity quaternion
        self.node_rotations = nn.Parameter(rots)

    @staticmethod
    def _lowbit(x: int) -> int:
        return x & (-x)

    def query(self, t: int):
        """
        BIT prefix query for frame t (1-based).
        Eq.10: T_k^t = sum_{j in BIT(t)} T_k^[j]
        Eq.11: r_k^t = norm(sum_{j in BIT(t)} r_k^[j])
        Returns: translations [K, 3], rotations [K, 4]
        """
        trans = torch.zeros(self.K, 3, device=self.node_translations.device,
                            dtype=self.node_translations.dtype)
        rots = torch.zeros(self.K, 4, device=self.node_rotations.device,
                           dtype=self.node_rotations.dtype)
        idx = t
        while idx > 0:
            trans = trans + self.node_translations[idx]
            rots = rots + self.node_rotations[idx]
            idx -= self._lowbit(idx)
        rots = normalize_quaternion(rots)
        return trans, rots

    def query_all_frames(self):
        """Query all T frames. Returns [T, K, 3] and [T, K, 4]."""
        all_trans = []
        all_rots = []
        for t in range(1, self.num_frames + 1):
            tr, ro = self.query(t)
            all_trans.append(tr)
            all_rots.append(ro)
        return torch.stack(all_trans), torch.stack(all_rots)
