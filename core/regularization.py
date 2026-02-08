import torch
from torch import Tensor


def temporal_regularization(all_means: Tensor) -> Tensor:
    """
    Eq.12 simplified: L_temp = mean(||mu_t - mu_{t+1}||^2)
    all_means: [T, N, 3]
    """
    flow = all_means[:-1] - all_means[1:]  # [T-1, N, 3]
    return (flow ** 2).mean()


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


def sample_surface_points(gaussian_means: Tensor, num_points: int = 5000) -> Tensor:
    """Sample points near Gaussian centers with small perturbation."""
    N = gaussian_means.shape[0]
    indices = torch.randint(0, N, (num_points,), device=gaussian_means.device)
    pts = gaussian_means[indices].clone()
    pts += torch.randn_like(pts) * 0.01
    return pts
