"""
Multi-object scene composition for 4DGS.

Manages N objects, each with its own GaussianModel + Deformation4D.
Concatenates all Gaussians for rendering (gsplat is object-agnostic).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from config import ObjectConfig
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D
from core.regularization import temporal_regularization, acceleration_regularization, arap_regularization, displacement_regularization


class MultiObjectScene(nn.Module):
    """Compose multiple 3DGS objects with independent deformations."""

    def __init__(self, obj_configs: list, num_frames: int,
                 global_num_coarse: int = 50, global_num_fine: int = 500,
                 K_neighbors: int = 10, device: str = 'cuda:0'):
        super().__init__()
        self.obj_configs = obj_configs
        self.num_frames = num_frames
        self.K_neighbors = K_neighbors

        self.gaussian_models = []
        self.deformations = nn.ModuleList()
        self._sizes = []       # per-object gaussian count
        self._cum_sizes = []   # cumulative sizes for slicing

        cum = 0
        for oc in obj_configs:
            gs = GaussianModel()
            gs.load_ply(oc.ply_path)
            gs.apply_rotation(oc.scene_rotate_x, oc.scene_rotate_y, oc.scene_rotate_z)
            if oc.center_to_origin:
                gs.center_to_origin()
            if oc.position_offset != (0.0, 0.0, 0.0):
                offset = oc.position_offset
                if isinstance(offset, (list, tuple)):
                    offset = list(offset)
                gs.apply_offset(offset)
            gs.to(device)

            num_coarse = oc.num_coarse_cp if oc.num_coarse_cp is not None else global_num_coarse
            num_fine = oc.num_fine_cp if oc.num_fine_cp is not None else global_num_fine

            deform = Deformation4D(
                gs,
                num_coarse=num_coarse,
                num_fine=num_fine,
                num_frames=num_frames,
            ).to(device)

            self.gaussian_models.append(gs)
            self.deformations.append(deform)
            self._sizes.append(gs.num_gaussians)
            self._cum_sizes.append(cum)
            cum += gs.num_gaussians

        self._total_gaussians = cum

    @property
    def num_objects(self):
        return len(self.obj_configs)

    @property
    def total_gaussians(self):
        return self._total_gaussians

    def get_object_slice(self, i: int) -> slice:
        """Return index slice for object i in concatenated tensors."""
        start = self._cum_sizes[i]
        end = start + self._sizes[i]
        return slice(start, end)

    def deform_all_frames(self, use_fine: bool = False, K_neighbors: int = 10):
        """
        Run each object's deformation and concatenate.
        Returns: all_means [T, N_total, 3], all_quats [T, N_total, 4]
        """
        per_obj_means = []
        per_obj_quats = []

        for i, (gs, deform) in enumerate(zip(self.gaussian_models, self.deformations)):
            means, quats = deform.deform_all_frames(gs, use_fine=use_fine, K_neighbors=K_neighbors)
            per_obj_means.append(means)
            per_obj_quats.append(quats)

        # Concatenate along gaussian dimension: [T, N_total, 3/4]
        all_means = torch.cat(per_obj_means, dim=1)
        all_quats = torch.cat(per_obj_quats, dim=1)
        return all_means, all_quats

    def get_static_properties(self):
        """
        Return concatenated static Gaussian properties for rendering.
        Returns: (scales, sh_coeffs, opacities, sh_degree)
        """
        all_scales = []
        all_sh = []
        all_opac = []
        min_sh_degree = None

        for gs in self.gaussian_models:
            all_scales.append(gs.get_activated_scales())
            all_sh.append(gs.get_sh_coeffs())
            all_opac.append(gs.get_activated_opacities())
            sd = int(math.sqrt(gs.get_sh_coeffs().shape[1])) - 1
            if min_sh_degree is None or sd < min_sh_degree:
                min_sh_degree = sd

        # Truncate SH to minimum degree across objects for compatibility
        K_min = (min_sh_degree + 1) ** 2
        all_sh = [sh[:, :K_min, :] for sh in all_sh]

        scales = torch.cat(all_scales, dim=0)
        sh_coeffs = torch.cat(all_sh, dim=0)
        opacities = torch.cat(all_opac, dim=0)
        return scales, sh_coeffs, opacities, min_sh_degree

    def get_optimizable_params(self, use_fine: bool = False):
        """Return optimizer param groups for all objects with prefixed names."""
        params = []
        for i, (oc, deform) in enumerate(zip(self.obj_configs, self.deformations)):
            obj_params = deform.get_optimizable_params(use_fine)
            for pg in obj_params:
                pg['name'] = f"{oc.name}/{pg['name']}"
            params.extend(obj_params)
        return params

    def compute_per_object_regularization(self, all_means, global_temp_w, global_spatial_w,
                                           global_accel_w=0.0, global_disp_w=0.0,
                                           num_arap_points=5000):
        """
        Compute temporal + acceleration + ARAP + displacement losses per object.
        Returns: total_reg_loss, per_object_metrics dict
        """
        total_loss = torch.tensor(0.0, device=all_means.device)
        metrics = {}

        for i, oc in enumerate(self.obj_configs):
            slc = self.get_object_slice(i)
            obj_means = all_means[:, slc, :]  # [T, N_i, 3]
            gs = self.gaussian_models[i]

            temp_w = global_temp_w * oc.temp_weight_mult
            spatial_w = global_spatial_w * oc.spatial_weight_mult

            L_temp = temporal_regularization(obj_means)
            L_accel = acceleration_regularization(obj_means)

            # ARAP on subsampled points
            N_i = gs.num_gaussians
            num_pts = min(num_arap_points, N_i)
            arap_idx = torch.randint(0, N_i, (num_pts,), device=all_means.device)
            arap_orig = gs.means[arap_idx].detach()
            L_arap = arap_regularization(arap_orig, obj_means[:, arap_idx, :])

            obj_loss = temp_w * L_temp + global_accel_w * L_accel + spatial_w * L_arap

            # Displacement regularization (per-axis anchoring)
            if global_disp_w > 0:
                L_disp = displacement_regularization(
                    obj_means, gs.means.detach(), oc.displacement_axis_weights)
                obj_loss = obj_loss + global_disp_w * L_disp
                metrics[f'{oc.name}/L_disp'] = L_disp.item()

            total_loss = total_loss + obj_loss

            metrics[f'{oc.name}/L_temp'] = L_temp.item()
            metrics[f'{oc.name}/L_accel'] = L_accel.item()
            metrics[f'{oc.name}/L_arap'] = L_arap.item()

        return total_loss, metrics

    def reinit_later_frames(self, t_ref: int):
        """Apply CHORD-style reinit to all objects."""
        for deform in self.deformations:
            deform.reinit_later_frames(t_ref)

    def refresh_weights(self, K_neighbors: int = 10):
        """Refresh blending weights for all objects."""
        for gs, deform in zip(self.gaussian_models, self.deformations):
            deform.refresh_weights(gs.means, K_neighbors)

    def scene_center(self):
        """Overall centroid across all objects."""
        all_means = torch.cat([gs.means for gs in self.gaussian_models], dim=0)
        return all_means.mean(dim=0)

    def get_global_color_white_point(self):
        """Return minimum color_white_point across objects."""
        return min(oc.color_white_point for oc in self.obj_configs)

    def state_dict_multi(self):
        """Save per-object deformation state dicts."""
        return {
            f'object_{i}_{oc.name}': deform.state_dict()
            for i, (oc, deform) in enumerate(zip(self.obj_configs, self.deformations))
        }

    def load_state_dict_multi(self, state_dicts: dict):
        """Load per-object deformation state dicts."""
        for i, (oc, deform) in enumerate(zip(self.obj_configs, self.deformations)):
            key = f'object_{i}_{oc.name}'
            if key in state_dicts:
                deform.load_state_dict(state_dicts[key])
            else:
                raise KeyError(f"Missing state dict for '{key}'. Available: {list(state_dicts.keys())}")
