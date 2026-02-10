import torch
import torch.nn as nn
from torch import Tensor
from models.gaussian_model import GaussianModel
from models.control_points import ControlPoints
from models.fenwick_tree import FenwickTreeDeformation
from utils.math_utils import (
    normalize_quaternion, quaternion_multiply,
    quaternion_to_rotation_matrix, quaternion_apply
)


class Deformation4D(nn.Module):
    def __init__(self, gaussian_model: GaussianModel,
                 num_coarse: int = 80, num_fine: int = 300, num_frames: int = 41):
        super().__init__()
        means = gaussian_model.means.detach()

        self.coarse_cp = ControlPoints(means, num_coarse)
        self.fine_cp = ControlPoints(means, num_fine)
        self.coarse_fenwick = FenwickTreeDeformation(num_coarse, num_frames)
        self.fine_fenwick = FenwickTreeDeformation(num_fine, num_frames)
        self.num_frames = num_frames

        # Pre-compute blending weights (will be refreshed if sigma changes)
        self._cache_weights(means)

    def _cache_weights(self, means: Tensor, K_neighbors: int = 10):
        cw, ci = self.coarse_cp.compute_blending_weights(means, K_neighbors)
        fw, fi = self.fine_cp.compute_blending_weights(means, K_neighbors)
        # Detach to avoid holding computation graph across steps
        self.register_buffer('coarse_weights', cw.detach(), persistent=False)
        self.register_buffer('coarse_indices', ci, persistent=False)
        self.register_buffer('fine_weights', fw.detach(), persistent=False)
        self.register_buffer('fine_indices', fi, persistent=False)

    def refresh_weights(self, means: Tensor, K_neighbors: int = 10, batch_size: int = 20000):
        """Re-compute blending weights (call when sigma is updated).

        Uses batching for large gaussian counts to avoid OOM on cdist.
        """
        N = means.shape[0]
        if N <= batch_size:
            cw, ci = self.coarse_cp.compute_blending_weights(means, K_neighbors)
            fw, fi = self.fine_cp.compute_blending_weights(means, K_neighbors)
            self.coarse_weights = cw.detach()
            self.coarse_indices = ci
            self.fine_weights = fw.detach()
            self.fine_indices = fi
        else:
            cw_l, ci_l, fw_l, fi_l = [], [], [], []
            for start in range(0, N, batch_size):
                batch = means[start:start + batch_size]
                cw, ci = self.coarse_cp.compute_blending_weights(batch, K_neighbors)
                fw, fi = self.fine_cp.compute_blending_weights(batch, K_neighbors)
                cw_l.append(cw.detach())
                ci_l.append(ci)
                fw_l.append(fw.detach())
                fi_l.append(fi)
            self.coarse_weights = torch.cat(cw_l, dim=0)
            self.coarse_indices = torch.cat(ci_l, dim=0)
            self.fine_weights = torch.cat(fw_l, dim=0)
            self.fine_indices = torch.cat(fi_l, dim=0)

    def reinit_later_frames(self, t_ref: int):
        """Copy frame t_ref's deformation to all later frames (CHORD-style reinit)."""
        with torch.no_grad():
            self.coarse_fenwick.reinit_later_frames(t_ref)
            self.fine_fenwick.reinit_later_frames(t_ref)

    def _lbs_deform(self, means: Tensor, quaternions: Tensor,
                    weights: Tensor, indices: Tensor,
                    ctrl_translations: Tensor, ctrl_rotations: Tensor,
                    ctrl_positions: Tensor):
        """
        Linear Blend Skinning deformation.
        Eq.5: mu^t = sum_k beta_k (R_k^t (mu - p_k) + p_k + T_k^t)
        Eq.6: q^t = normalize(sum_k beta_k * r_k^t) x q

        means: [N, 3], quaternions: [N, 4]
        weights: [N, K_nb], indices: [N, K_nb]
        ctrl_translations: [K_total, 3], ctrl_rotations: [K_total, 4]
        ctrl_positions: [K_total, 3]
        """
        N, K_nb = weights.shape

        # Gather neighbor control point data
        nb_trans = ctrl_translations[indices]    # [N, K_nb, 3]
        nb_rots = ctrl_rotations[indices]        # [N, K_nb, 4]
        nb_pos = ctrl_positions[indices]          # [N, K_nb, 3]

        # Eq.5: position deformation
        # (mu - p_k) for each neighbor
        diff = means.unsqueeze(1) - nb_pos  # [N, K_nb, 3]

        # Rotate diff by control point rotation
        R_ctrl = quaternion_to_rotation_matrix(nb_rots)  # [N, K_nb, 3, 3]
        rotated_diff = torch.einsum('nkij,nkj->nki', R_ctrl, diff)  # [N, K_nb, 3]

        # R_k(mu - p_k) + p_k + T_k
        deformed_per_cp = rotated_diff + nb_pos + nb_trans  # [N, K_nb, 3]

        # Weighted sum
        w = weights.unsqueeze(-1)  # [N, K_nb, 1]
        deformed_means = (w * deformed_per_cp).sum(dim=1)  # [N, 3]

        # Eq.6: rotation deformation
        # Weighted average quaternion
        blended_rot = (w * nb_rots).sum(dim=1)  # [N, 4]
        blended_rot = normalize_quaternion(blended_rot)
        deformed_quats = quaternion_multiply(blended_rot, quaternions)  # [N, 4]

        return deformed_means, deformed_quats

    def deform_frame(self, gaussian_model: GaussianModel, t: int, use_fine: bool = False):
        """
        Deform frame t (0-indexed, internally 1-indexed for Fenwick).
        Frame 0 is the static reference — returns original means/quats unchanged.
        Returns: deformed_means [N, 3], deformed_quats [N, 4]
        """
        # Frame 0: static reference, no deformation
        if t == 0:
            return gaussian_model.means, gaussian_model.quaternions

        t_fenwick = t  # frame 1 → node 1, frame 2 → node 2, ...

        # Coarse deformation
        c_trans, c_rots = self.coarse_fenwick.query(t_fenwick)
        dm, dq = self._lbs_deform(
            gaussian_model.means, gaussian_model.quaternions,
            self.coarse_weights, self.coarse_indices,
            c_trans, c_rots, self.coarse_cp.positions
        )

        if use_fine:
            # Fine residual deformation (Eq.8, 9)
            f_trans, f_rots = self.fine_fenwick.query(t_fenwick)
            dm, dq = self._lbs_deform(
                dm, dq,
                self.fine_weights, self.fine_indices,
                f_trans, f_rots, self.fine_cp.positions
            )

        return dm, dq

    def deform_all_frames(self, gaussian_model: GaussianModel, use_fine: bool = False):
        """Deform all frames. Returns [T, N, 3] and [T, N, 4]."""
        all_means = []
        all_quats = []
        for t in range(self.num_frames):
            dm, dq = self.deform_frame(gaussian_model, t, use_fine)
            all_means.append(dm)
            all_quats.append(dq)
        return torch.stack(all_means), torch.stack(all_quats)

    def get_optimizable_params(self, use_fine: bool = False):
        """Return parameter groups for optimizer."""
        params = [
            {'params': [self.coarse_fenwick.node_translations], 'name': 'coarse_trans'},
            {'params': [self.coarse_fenwick.node_rotations], 'name': 'coarse_rots'},
            {'params': [self.coarse_cp.log_sigma], 'name': 'coarse_sigma'},
        ]
        if use_fine:
            params.extend([
                {'params': [self.fine_fenwick.node_translations], 'name': 'fine_trans'},
                {'params': [self.fine_fenwick.node_rotations], 'name': 'fine_rots'},
                {'params': [self.fine_cp.log_sigma], 'name': 'fine_sigma'},
            ])
        return params
