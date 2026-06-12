"""
learn_square.py: Fit a single superquadric splat to a red square target image.

Demonstrates end-to-end differentiable rendering — the splat starts random and
optimises its position, colour, opacity, scale, rotation, and shape exponents
purely through gradient descent on MSE loss against the target.

Output:
  Results/render_tests/learn_square/          individual frame PNGs
  Results/render_tests/learn_square.mp4       timelapse video
  Results/render_tests/learn_square_final.png final state vs target

Each video frame shows three panels side by side:
  [ Target (red square) | Current render | Absolute difference ]
"""

import sys
sys.path.insert(0, '/home/daniel/Documents/Projects/Sync/3D/3DSQS/submodules/diff-superquadric-rasterization')
from diff_superquadric_rasterization import SuperquadricRasterizationSettings, SuperquadricRasterizer

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR   = '/home/daniel/Documents/Projects/Sync/3D/Results/render_tests'
FRAMES_DIR = os.path.join(BASE_DIR, 'learn_square')
os.makedirs(FRAMES_DIR, exist_ok=True)

device, dtype = 'cuda', torch.float32

H, W     = 256, 256            # image resolution
FOV_DEG  = 60.0
TANFOV   = math.tan(math.radians(FOV_DEG / 2))

N_STEPS     = 800              # total gradient steps
LR          = 0.05             # Adam learning rate
SAVE_EVERY  = 5                # save a video frame every N steps
VIDEO_FPS   = 30

torch.manual_seed(0)

# ── Rasterizer setup ───────────────────────────────────────────────────────────
# Camera at world origin, looking along +z (identity viewmatrix).
# Splat means3D are given directly in camera space.

viewmat  = torch.eye(4, device=device, dtype=dtype)
bg_black = torch.zeros(3, device=device, dtype=dtype)

settings = SuperquadricRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=TANFOV, tanfovy=TANFOV,
    bg=bg_black, viewmatrix=viewmat)
rast = SuperquadricRasterizer(settings)

# ── Target image: red square ───────────────────────────────────────────────────
# Square occupies the central 50% of the image in each axis.
# Steps:
#   1. Build (H, W, 3) float32 numpy array filled with zeros (black background)
#   2. Fill the central region with pure red [1, 0, 0]
#   3. Convert to (3, H, W) torch tensor for loss computation

# Step 1
target_np = np.zeros((H, W, 3), dtype=np.float32)

# Step 2 — centre square, 50% of image
sq_lo = H // 4      # 64
sq_hi = 3 * H // 4  # 192

target_np[sq_lo:sq_hi, sq_lo:sq_hi] = [1., 0., 0.]

# Step 3
target = torch.tensor(target_np, device=device, dtype=dtype).permute(2, 0, 1)  # (3,H,W)
target_uint8 = (target_np * 255).astype(np.uint8)   # (H, W, 3) for display

# ── Learnable parameters (reparameterised to stay in valid ranges) ─────────────
#
# We optimise unconstrained "raw" tensors and apply activations before rendering:
#
#   raw_xyz   → means3D (camera space; z stays near 3 via initialisation)
#   raw_color → sigmoid → colours in [0, 1]
#   raw_op    → sigmoid → opacity in (0, 1)
#   log_scale → exp    → scales > 0
#   raw_quat  → normalise → unit quaternion for rotation
#   raw_e12   → sigmoid * 1.8 + 0.1 → e1, e2 in [0.1, 1.9]
#   raw_e3    → sigmoid * 4.1 + 0.9 → e3 in [0.9, 5.0]

# Initialise the splat at a random position but with a deliberately large scale
# so it covers enough pixels to get useful gradient signal from the start.
raw_xyz   = torch.tensor([[ 0.4, -0.3, 3.5]], device=device, dtype=dtype,
                          requires_grad=True)                          # off-centre start
raw_color = torch.randn(1, 3, device=device, dtype=dtype,
                         requires_grad=True)                          # random initial colour
raw_op    = torch.tensor([[1.0]], device=device, dtype=dtype,
                          requires_grad=True)                          # high initial opacity
log_scale = torch.tensor([[np.log(0.7), np.log(0.7), np.log(0.7)]],
                          device=device, dtype=dtype,
                          requires_grad=True)                          # scale ≈ 0.7 (big start)
raw_quat  = torch.randn(1, 4, device=device, dtype=dtype,
                         requires_grad=True)                          # random rotation
raw_e12   = torch.zeros(1, 2, device=device, dtype=dtype,
                         requires_grad=True)                          # e1=e2≈1 (sphere-like)
raw_e3    = torch.zeros(1, 1, device=device, dtype=dtype,
                         requires_grad=True)                          # e3≈2.9 (medium sharpness)

def get_splat_params():
    """Apply activations to raw params; return (m, c, o, s, q, e) ready for rasterizer.

    Steps:
      1. Position: use raw_xyz directly (z stays positive by initialisation)
      2. Colour:   sigmoid keeps RGB in (0, 1)
      3. Opacity:  sigmoid keeps in (0, 1)
      4. Scale:    exp keeps positive
      5. Rotation: L2-normalise raw_quat to unit quaternion
      6. Exps:     sigmoid-based mapping into allowed ranges
    """
    m = raw_xyz                                                 # Step 1
    c = torch.sigmoid(raw_color)                               # Step 2
    o = torch.sigmoid(raw_op)                                  # Step 3
    s = torch.exp(log_scale)                                   # Step 4
    q = raw_quat / (raw_quat.norm(dim=1, keepdim=True) + 1e-8)  # Step 5
    e12 = torch.sigmoid(raw_e12) * 1.8 + 0.1                  # Step 6a: [0.1, 1.9]
    e3  = torch.sigmoid(raw_e3)  * 4.1 + 0.9                  # Step 6b: [0.9, 5.0]
    e   = torch.cat([e12, e3], dim=1)
    return m, c, o, s, q, e

# ── Optimiser ──────────────────────────────────────────────────────────────────
# Use separate per-group learning rates:
#   position gets a higher lr to move quickly across the image plane
#   colour/opacity get standard lr
#   shape params (scale, rotation, exponents) get standard lr

optimizer = torch.optim.Adam([
    {'params': [raw_xyz],             'lr': LR * 2.0},  # move fast
    {'params': [raw_color, raw_op],   'lr': LR},
    {'params': [log_scale, raw_quat], 'lr': LR},
    {'params': [raw_e12, raw_e3],     'lr': LR * 0.5},  # shape changes slower
])

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_frame(render_t):
    """Build a (H, W*3, 3) uint8 numpy array: [target | render | |diff|].

    Steps:
      1. Convert render tensor to uint8
      2. Compute absolute difference
      3. Concatenate horizontally and add step label
    """
    # Step 1
    render_np = (render_t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255
                 ).astype(np.uint8)
    # Step 2
    diff_np   = np.abs(target_np * 255 - render_np.astype(np.float32)).astype(np.uint8)
    # Step 3
    return np.concatenate([target_uint8, render_np, diff_np], axis=1)

def annotate(frame_np, text):
    """Overlay text on a frame (returns PIL Image)."""
    pil  = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 6), text, fill=(255, 255, 150))
    return np.array(pil)

# ── Training loop ──────────────────────────────────────────────────────────────

frames     = []
loss_curve = []

print(f'Target: red square  ({sq_lo}:{sq_hi}, {sq_lo}:{sq_hi})  in {H}x{W} image')
print(f'Steps: {N_STEPS}  lr={LR}  save every {SAVE_EVERY} steps')
print()

for step in range(N_STEPS + 1):

    optimizer.zero_grad()

    m, c, o, s, q, e = get_splat_params()
    render, _, _ = rast(m, c, o, s, q, e)

    loss = F.mse_loss(render, target)

    if step > 0:
        loss.backward()
        optimizer.step()

    loss_val = loss.item()
    loss_curve.append(loss_val)

    # ── Save frame ──────────────────────────────────────────────────────────
    if step % SAVE_EVERY == 0 or step == N_STEPS:
        with torch.no_grad():
            m_, c_, o_, s_, q_, e_ = get_splat_params()
            col, _, _ = rast(m_, c_, o_, s_, q_, e_)

        e_vals = e_[0].tolist()
        label  = (f'step {step:4d}  loss={loss_val:.4f}\n'
                  f'e=[{e_vals[0]:.2f},{e_vals[1]:.2f},{e_vals[2]:.2f}]'
                  f'  s={[round(v,2) for v in s_[0].tolist()]}')

        frame = annotate(make_frame(col), label)
        frames.append(frame)

    # ── Console log every 50 steps ──────────────────────────────────────────
    if step % 50 == 0:
        with torch.no_grad():
            m_, c_, o_, s_, q_, e_ = get_splat_params()
        print(f'  step {step:4d}  loss={loss_val:.5f}'
              f'  pos=[{m_[0,0].item():.3f},{m_[0,1].item():.3f},{m_[0,2].item():.3f}]'
              f'  col={[round(v,2) for v in c_[0].tolist()]}'
              f'  e=[{e_[0,0].item():.2f},{e_[0,1].item():.2f},{e_[0,2].item():.2f}]')

print()
print(f'Final loss: {loss_curve[-1]:.5f}  (start: {loss_curve[0]:.5f})')

# ── Save video ─────────────────────────────────────────────────────────────────
video_path = os.path.join(BASE_DIR, 'learn_square.gif')
# Duration per frame in ms (1000 / fps)
imageio.mimsave(video_path, frames, duration=1000 // VIDEO_FPS, loop=0)
print(f'Video saved: {video_path}  ({len(frames)} frames @ {VIDEO_FPS}fps)')

# ── Save final comparison PNG ──────────────────────────────────────────────────
final_path = os.path.join(BASE_DIR, 'learn_square_final.png')
Image.fromarray(frames[-1]).save(final_path)
print(f'Final frame: {final_path}')

# ── Save individual key frames ─────────────────────────────────────────────────
for idx, step_n in enumerate([0, N_STEPS // 4, N_STEPS // 2, 3 * N_STEPS // 4, N_STEPS]):
    frame_idx = step_n // SAVE_EVERY
    if frame_idx < len(frames):
        path = os.path.join(FRAMES_DIR, f'step_{step_n:04d}.png')
        Image.fromarray(frames[frame_idx]).save(path)
print(f'Key frames saved in: {FRAMES_DIR}')
