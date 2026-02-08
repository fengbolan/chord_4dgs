import torch
import torch.nn as nn
from torch import Tensor


class ControlPoints(nn.Module):
    def __init__(self, gaussian_means: Tensor, num_points: int):
        super().__init__()
        positions = self._fps(gaussian_means.detach(), num_points)
        self.register_buffer('positions', positions)  # [K, 3], fixed

        # Initialize log_sigma: log(avg distance to 3 nearest control points)
        dists = torch.cdist(positions, positions)  # [K, K]
        dists.fill_diagonal_(float('inf'))
        knn_dists, _ = dists.topk(3, dim=1, largest=False)
        init_sigma = knn_dists.mean(dim=1).log()  # [K]
        self.log_sigma = nn.Parameter(init_sigma)

    def _fps(self, points: Tensor, num_samples: int) -> Tensor:
        """Farthest Point Sampling."""
        N = points.shape[0]
        if num_samples >= N:
            return points.clone()
        selected = [torch.randint(N, (1,), device=points.device).item()]
        min_dists = torch.full((N,), float('inf'), device=points.device)

        for _ in range(num_samples - 1):
            last = points[selected[-1]].unsqueeze(0)  # [1, 3]
            d = ((points - last) ** 2).sum(dim=1)     # [N]
            min_dists = torch.minimum(min_dists, d)
            selected.append(min_dists.argmax().item())

        return points[selected].clone()

    def compute_blending_weights(self, query_points: Tensor, K_neighbors: int = 10):
        """
        Compute blending weights (Eq.7).
        query_points: [N, 3]
        Returns: weights [N, K_nb], indices [N, K_nb]
        """
        sigma = torch.exp(self.log_sigma)  # [K]
        dists = torch.cdist(query_points, self.positions)  # [N, K]

        K_nb = min(K_neighbors, self.positions.shape[0])
        _, indices = dists.topk(K_nb, dim=1, largest=False)  # [N, K_nb]

        # Gather distances and sigmas for neighbors
        nb_dists = torch.gather(dists, 1, indices)  # [N, K_nb]
        nb_sigma = sigma[indices]  # [N, K_nb]

        # Eq.7: beta_hat = exp(-0.5 * ||mu - p_k||^2 / sigma_k^2)
        beta_hat = torch.exp(-0.5 * nb_dists ** 2 / (nb_sigma ** 2 + 1e-8))  # [N, K_nb]

        # Normalize
        weights = beta_hat / (beta_hat.sum(dim=1, keepdim=True) + 1e-8)

        return weights, indices
