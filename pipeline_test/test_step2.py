import torch, sys
sys.path.insert(0, '.')
from utils.math_utils import *
from models.gaussian_model import GaussianModel

# Test quaternions
q1 = torch.randn(10, 4, device='cuda')
q1 = normalize_quaternion(q1)
q_id = quaternion_identity(10, 'cuda')
q_out = quaternion_multiply(q1, q_id)
assert torch.allclose(q1, q_out, atol=1e-5), "Identity multiply failed"

R = quaternion_to_rotation_matrix(q1)
assert R.shape == (10, 3, 3)
det = torch.det(R)
assert torch.allclose(det, torch.ones(10, device='cuda'), atol=1e-4), "Not valid rotation"

# Test gradients
q = torch.randn(5, 4, device='cuda', requires_grad=True)
R = quaternion_to_rotation_matrix(normalize_quaternion(q))
loss = R.sum()
loss.backward()
assert q.grad is not None, "Gradient not flowing"

# Test PLY loading
gs = GaussianModel()
gs.load_ply('../data/1.ply')
print(f"Loaded {gs.num_gaussians} gaussians")
print(f"Means range: {gs.means.min(0).values} to {gs.means.max(0).values}")
print(f"Scales range: {gs.scales.min()} to {gs.scales.max()}")
assert gs.num_gaussians > 0
assert gs.means.shape[1] == 3
assert gs.quaternions.shape[1] == 4
print("✅ Step 2 ALL TESTS PASSED")
