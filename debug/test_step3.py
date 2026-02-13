import torch, sys
sys.path.insert(0, '.')
from models.gaussian_model import GaussianModel
from models.control_points import ControlPoints
from models.fenwick_tree import FenwickTreeDeformation

gs = GaussianModel()
gs.load_ply('../data/1.ply')

# Test control points
cp = ControlPoints(gs.means, num_points=80).cuda()
print(f"Control points: {cp.positions.shape}")
weights, indices = cp.compute_blending_weights(gs.means, K_neighbors=10)
assert weights.shape == (gs.num_gaussians, 10)
assert torch.allclose(weights.sum(dim=1), torch.ones(gs.num_gaussians, device='cuda'), atol=1e-5), \
    "Weights don't sum to 1"
# Test gradient flow through log_sigma
loss = weights.sum()
loss.backward()
assert cp.log_sigma.grad is not None, "Gradient not flowing to log_sigma"

# Test Fenwick Tree
ft = FenwickTreeDeformation(num_control_points=80, num_frames=41).cuda()
trans, rots = ft.query(1)
assert trans.shape == (80, 3)
assert rots.shape == (80, 4)

# Test temporal consistency
all_trans, all_rots = ft.query_all_frames()
assert all_trans.shape == (41, 80, 3)

# Test gradient
loss = all_trans.sum() + all_rots.sum()
loss.backward()
assert ft.node_translations.grad is not None

print("✅ Step 3 ALL TESTS PASSED")
