import torch, sys
sys.path.insert(0, '.')
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D

gs = GaussianModel()
gs.load_ply('../data/1.ply')
N = gs.num_gaussians

deform = Deformation4D(gs, num_coarse=50, num_fine=200, num_frames=16).cuda()

# Test single frame deformation (coarse only)
dm, dq = deform.deform_frame(gs, t=0, use_fine=False)
assert dm.shape == (N, 3), f"Expected ({N}, 3), got {dm.shape}"
assert dq.shape == (N, 4)

# t=0 should be ~identity deformation (params init to 0/identity)
assert torch.allclose(dm, gs.means, atol=1e-3), "Frame 0 should be ~identity deformation"

# Test fine layer
dm_f, dq_f = deform.deform_frame(gs, t=5, use_fine=True)
assert dm_f.shape == (N, 3)

# Test all frames
all_m, all_q = deform.deform_all_frames(gs, use_fine=False)
assert all_m.shape == (16, N, 3)
assert all_q.shape == (16, N, 4)

# Test gradient flow to Fenwick Tree parameters
loss = all_m.sum()
loss.backward()
assert deform.coarse_fenwick.node_translations.grad is not None, "Gradient not flowing!"

# Test parameter list
params = deform.get_optimizable_params(use_fine=False)
assert len(params) > 0
params_fine = deform.get_optimizable_params(use_fine=True)
assert len(params_fine) > len(params)

print(f"Gaussians: {N}, Coarse CPs: 50, Fine CPs: 200, Frames: 16")
print("✅ Step 4 ALL TESTS PASSED")
