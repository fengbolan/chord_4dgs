import torch
from torch import Tensor


def temporal_regularization(all_means: Tensor) -> Tensor:
    """
    Eq.12 simplified: L_temp = mean(||mu_t - mu_{t+1}||^2)
    all_means: [T, N, 3]
    """
    flow = all_means[:-1] - all_means[1:]  # [T-1, N, 3]
    return (flow ** 2).mean()


def acceleration_regularization(all_means: Tensor) -> Tensor:
    """
    Penalize acceleration (second-order temporal derivative) to suppress oscillation/jitter.
    L_accel = mean(||mu_{t-1} - 2*mu_t + mu_{t+1}||^2)

    Temporal reg (velocity) prevents drift but not oscillation.
    Acceleration reg directly penalizes direction changes between consecutive frames —
    if frames oscillate (left, right, left, ...), the acceleration is large even though
    per-frame velocity is small.

    all_means: [T, N, 3], requires T >= 3
    """
    accel = all_means[:-2] - 2 * all_means[1:-1] + all_means[2:]  # [T-2, N, 3]
    return (accel ** 2).mean()


def arap_regularization(original_points: Tensor, deformed_points_seq: Tensor,
                         k_neighbors: int = 10) -> Tensor:
    """
    Eq.13 ARAP loss.
    original_points: [M, 3]
    deformed_points_seq: [T, M, 3]
    """
    M = original_points.shape[0]
    # Find k nearest neighbors in original configuration
    dists = torch.cdist(original_points, original_points)  # [M, M]
    _, nn_idx = dists.topk(k_neighbors + 1, dim=1, largest=False)  # include self
    nn_idx = nn_idx[:, 1:]  # exclude self, [M, k]

    # Original edge vectors
    orig_edges = original_points[nn_idx] - original_points.unsqueeze(1)  # [M, k, 3]

    loss = torch.tensor(0.0, device=original_points.device)
    T = deformed_points_seq.shape[0]

    for t in range(T):
        def_pts = deformed_points_seq[t]  # [M, 3]
        def_edges = def_pts[nn_idx] - def_pts.unsqueeze(1)  # [M, k, 3]

        # SVD to find optimal local rotation
        # H = orig_edges^T @ def_edges -> [M, 3, 3]
        H = torch.bmm(orig_edges.transpose(1, 2), def_edges)  # [M, 3, 3]
        U, S, Vh = torch.linalg.svd(H)
        R_local = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))  # [M, 3, 3]

        # ARAP energy: ||def_edges - R @ orig_edges||^2
        rotated_orig = torch.bmm(orig_edges, R_local.transpose(1, 2))  # [M, k, 3]
        energy = ((def_edges - rotated_orig) ** 2).mean()
        loss = loss + energy

    return loss / T


def contact_proximity_loss(means_a: Tensor, means_b: Tensor,
                           target_distance: float = 0.05,
                           subsample: int = 2000) -> Tensor:
    """
    Penalize when minimum distance between two object clouds exceeds target.
    Encourages objects to make contact.
    means_a: [N_a, 3], means_b: [N_b, 3]
    """
    n_a = min(subsample, means_a.shape[0])
    n_b = min(subsample, means_b.shape[0])
    idx_a = torch.randint(0, means_a.shape[0], (n_a,), device=means_a.device)
    idx_b = torch.randint(0, means_b.shape[0], (n_b,), device=means_b.device)
    dists = torch.cdist(means_a[idx_a], means_b[idx_b])
    min_dist = dists.min()
    return torch.relu(min_dist - target_distance) ** 2


def impact_deformation_loss(pillow_means: Tensor, pillow_original: Tensor,
                            cat_means: Tensor,
                            influence_radius: float = 0.3) -> Tensor:
    """
    Encourage pillow points near the cat to deform (not stay rigid).
    Rewards pillow displacement where the cat is close.
    """
    dists = torch.cdist(pillow_means, cat_means)  # [N_pillow, N_cat]
    min_dist_to_cat = dists.min(dim=1).values      # [N_pillow]
    near_mask = (min_dist_to_cat < influence_radius).float()
    displacement = (pillow_means - pillow_original).norm(dim=-1)
    # Penalize near-cat pillow points that haven't moved
    return (near_mask * torch.relu(0.01 - displacement)).mean()


def displacement_regularization(all_means: Tensor, original_means: Tensor,
                                 axis_weights: tuple = (1.0, 1.0, 1.0)) -> Tensor:
    """Penalize displacement from original position with per-axis control.
    all_means: [T, N, 3], original_means: [N, 3]
    axis_weights: (wx, wy, wz) — 0 = free, higher = more anchored
    """
    displacement = all_means - original_means.unsqueeze(0)  # [T, N, 3]
    w = torch.tensor(axis_weights, dtype=displacement.dtype, device=displacement.device)
    return (displacement.pow(2) * w).mean()


def sample_surface_points(gaussian_means: Tensor, num_points: int = 5000) -> Tensor:
    """Sample points near Gaussian centers with small perturbation."""
    N = gaussian_means.shape[0]
    indices = torch.randint(0, N, (num_points,), device=gaussian_means.device)
    pts = gaussian_means[indices].clone()
    pts += torch.randn_like(pts) * 0.01
    return pts
