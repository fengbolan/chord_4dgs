import torch, sys, math
sys.path.insert(0, '.')
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D
from core.renderer import render_gaussians, render_video
from utils.camera_utils import orbit_camera, get_intrinsics
from PIL import Image
import numpy as np

gs = GaussianModel()
gs.load_ply('../data/1.ply')

# Get activated properties
activated_opacities = gs.get_activated_opacities()
activated_scales = gs.get_activated_scales()
sh_coeffs = gs.get_sh_coeffs()
sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1
print(f"SH degree: {sh_degree}, SH coeffs shape: {sh_coeffs.shape}")
print(f"Activated opacities range: [{activated_opacities.min():.4f}, {activated_opacities.max():.4f}]")
print(f"Activated scales range: [{activated_scales.min():.6f}, {activated_scales.max():.6f}]")

# Render static frame
W, H = 512, 288
target = gs.means.mean(dim=0).detach().cpu()
viewmat = orbit_camera(elevation_deg=15, azimuth_deg=45, radius=3.0, target=target).cuda()
fx, fy, cx, cy = get_intrinsics(fovy_deg=49.1, width=W, height=H)

image, alpha = render_gaussians(
    gs.means, gs.quaternions, activated_scales, sh_coeffs, activated_opacities,
    viewmat, (fx, fy, cx, cy), W, H, sh_degree=sh_degree
)
assert image.shape == (H, W, 3), f"Got {image.shape}"
print(f"Render output range: [{image.min():.4f}, {image.max():.4f}]")
print(f"Alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")

# Save
img_np = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save('test_render_static.png')
print(f"Saved test_render_static.png")

# Test gradient
means = gs.means.clone().requires_grad_(True)
image2, _ = render_gaussians(means, gs.quaternions, activated_scales, sh_coeffs, activated_opacities,
                              viewmat, (fx, fy, cx, cy), W, H, sh_degree=sh_degree)
loss = image2.sum()
loss.backward()
assert means.grad is not None, "Gradient not flowing through renderer"
print("Gradient test passed")

# Render deformed video
deform = Deformation4D(gs, num_coarse=50, num_fine=100, num_frames=8).cuda()
# Give some deformation to verify it's not static
with torch.no_grad():
    deform.coarse_fenwick.node_translations.data[3, :, 0] = 0.1
all_m, all_q = deform.deform_all_frames(gs, use_fine=False)
video = render_video(all_m, all_q, activated_scales, sh_coeffs, activated_opacities,
                      viewmat, (fx, fy, cx, cy), W, H, num_frames=8, sh_degree=sh_degree)
assert video.shape == (8, H, W, 3)

# Save GIF
frames = [(video[t].detach().cpu().clamp(0,1).numpy()*255).astype(np.uint8) for t in range(8)]
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save('test_render_video.gif', save_all=True, append_images=imgs[1:], duration=200, loop=0)
print("Saved test_render_video.gif")

print("✅ Step 5 ALL TESTS PASSED")
