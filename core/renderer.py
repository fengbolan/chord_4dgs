import torch
from torch import Tensor
from gsplat import rasterization


def render_gaussians(means: Tensor, quats: Tensor, scales: Tensor,
                     colors: Tensor, opacities: Tensor,
                     viewmat: Tensor, K, width: int, height: int,
                     bg_color=(0, 0, 0), sh_degree=None):
    """
    Differentiable rendering using gsplat.

    means: [N, 3], quats: [N, 4], scales: [N, 3]
    colors: [N, 3] or [N, K_sh, 3] (SH)
    opacities: [N, 1] or [N]
    viewmat: [4, 4] world-to-camera
    K: (fx, fy, cx, cy) tuple or [3, 3] matrix
    Returns: image [H, W, 3], alpha [H, W, 1]
    """
    device = means.device

    # Ensure opacities is [N]
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    # Build intrinsics matrix [1, 3, 3]
    if isinstance(K, (tuple, list)):
        fx, fy, cx, cy = K
        Ks = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
    else:
        Ks = K.unsqueeze(0) if K.dim() == 2 else K

    # viewmat: [1, 4, 4]
    viewmats = viewmat.unsqueeze(0) if viewmat.dim() == 2 else viewmat

    # Background
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3]

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats.to(device),
        Ks=Ks.to(device),
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB",
        backgrounds=bg,
        packed=False,
    )

    # render_colors: [1, H, W, 3] -> [H, W, 3]
    # render_alphas: [1, H, W, 1] -> [H, W, 1]
    image = render_colors[0]
    alpha = render_alphas[0]
    return image, alpha


def render_video(all_means: Tensor, all_quats: Tensor,
                 scales: Tensor, colors: Tensor, opacities: Tensor,
                 viewmat: Tensor, K, width: int, height: int,
                 num_frames: int, bg_color=(0, 0, 0), sh_degree=None):
    """
    Render video sequence.
    all_means: [T, N, 3], all_quats: [T, N, 4]
    Rest are static per-Gaussian properties.
    viewmat: [4, 4] (same view for all frames) or [T, 4, 4] (per-frame camera follow)
    Returns: video [T, H, W, 3]
    """
    per_frame_view = viewmat.dim() == 3
    frames = []
    for t in range(num_frames):
        vm = viewmat[t] if per_frame_view else viewmat
        img, _ = render_gaussians(
            all_means[t], all_quats[t], scales, colors, opacities,
            vm, K, width, height, bg_color, sh_degree
        )
        frames.append(img)
    return torch.stack(frames)
