import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =========================
# WCB Simulator Animation
# Pixel-art E. coli glow
# =========================

np.random.seed(3)

# -------------------------
# Simulated fluorescence
# -------------------------
time = np.arange(0, 31, 1)  # 0–30 min

# Example fluorescence response curve
F = 0.1 + 2.2 * (1 - np.exp(-time / 10))

# Normalize to 0–1
brightness = (F - F.min()) / (F.max() - F.min())

# -------------------------
# Pixel canvas
# -------------------------
H, W = 80, 120

# Cell center
cx0, cy0 = W // 2, H // 2

# Rod size
cell_length = 42
cell_width = 16

# Rotation angle in degrees
angle_deg = -18
angle = np.deg2rad(angle_deg)

# Coordinate grid
Y, X = np.mgrid[0:H, 0:W]

def rotated_coords(X, Y, cx, cy, angle):
    x = X - cx
    y = Y - cy
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)
    return xr, yr

def capsule_mask(X, Y, cx, cy, length, width, angle):
    xr, yr = rotated_coords(X, Y, cx, cy, angle)

    radius = width / 2
    rect_len = length - width

    center_rect = (np.abs(xr) <= rect_len / 2) & (np.abs(yr) <= radius)

    left_cap = ((xr + rect_len / 2) ** 2 + yr ** 2) <= radius ** 2
    right_cap = ((xr - rect_len / 2) ** 2 + yr ** 2) <= radius ** 2

    return center_rect | left_cap | right_cap

def make_frame(i):
    b = brightness[i]

    # Small alive-like jitter
    cx = cx0 + 2 * np.sin(i * 0.35)
    cy = cy0 + 1.5 * np.cos(i * 0.28)

    # Background
    img = np.zeros((H, W, 3), dtype=float)

    # Random microscope-like background noise
    noise = np.random.random((H, W))
    green_noise = noise > 0.992
    img[green_noise, 1] = 0.20 + 0.25 * b

    # Fluorescence halo
    xr, yr = rotated_coords(X, Y, cx, cy, angle)
    halo = np.exp(-((xr / 35) ** 2 + (yr / 16) ** 2))
    img[:, :, 1] += halo * (0.10 + 0.75 * b)

    # Cell body
    body = capsule_mask(X, Y, cx, cy, cell_length, cell_width, angle)
    img[body, 1] += 0.25 + 0.65 * b
    img[body, 0] += 0.05 + 0.20 * b

    # Inner darker region
    inner = capsule_mask(X, Y, cx, cy, cell_length - 8, cell_width - 6, angle)
    img[inner, 1] *= 0.65

    # Bright outline
    outline = capsule_mask(X, Y, cx, cy, cell_length + 2, cell_width + 2, angle) & ~body
    img[outline, 1] += 0.35 + 0.65 * b

    # One bright stripe inside the cell
    stripe = (np.abs(yr + 2) < 2) & (np.abs(xr) < cell_length / 2 - 6)
    img[stripe, 1] += 0.35 + 0.70 * b
    img[stripe, 0] += 0.15 + 0.25 * b

    # Clip values
    img = np.clip(img, 0, 1)

    return img

# -------------------------
# Plot animation
# -------------------------
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

im = ax.imshow(make_frame(0), interpolation="nearest")
ax.axis("off")

title = ax.text(
    0.5, 1.03,
    "",
    transform=ax.transAxes,
    ha="center",
    color="white",
    fontsize=14,
    family="monospace"
)

label = ax.text(
    0.5, -0.08,
    "",
    transform=ax.transAxes,
    ha="center",
    color="#8cff4a",
    fontsize=11,
    family="monospace"
)

def update(i):
    im.set_data(make_frame(i))

    title.set_text(f"E. coli GreenPegasos Response | {time[i]} min")
    label.set_text(f"Fluorescence: {F[i]:.2f} AU")

    return im, title, label

ani = FuncAnimation(
    fig,
    update,
    frames=len(time),
    interval=180,
    blit=False
)

plt.show()

# -------------------------
# Save as GIF
# -------------------------
ani.save(
    "wcb_pixel_ecoli_glow.gif",
    writer=PillowWriter(fps=8)
)