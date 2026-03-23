import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ── Load all grid sizes (both data and label) ────────────────────────────────
ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9")
filelist = ROOT / "qm9_filelist.txt"
indices = [line.strip() for line in filelist.read_text().splitlines() if line.strip()]

rows = []
for idx in tqdm(indices, desc="Reading grid sizes"):
    idx_int = int(idx)
    sample_name = f"dsgdb9nsd_{idx_int:06d}"

    data_gs = ROOT / "data" / sample_name / "grid_sizes_22.dat"
    label_gs = ROOT / "label" / sample_name / "grid_sizes_22.dat"

    if not data_gs.exists() or not label_gs.exists():
        continue

    dnx, dny, dnz = np.loadtxt(data_gs, dtype=int)
    lnx, lny, lnz = np.loadtxt(label_gs, dtype=int)

    rows.append({
        "index": idx_int,
        "data_nx": dnx, "data_ny": dny, "data_nz": dnz,
        "data_voxels": int(dnx * dny * dnz),
        "label_nx": lnx, "label_ny": lny, "label_nz": lnz,
        "label_voxels": int(lnx * lny * lnz),
    })

print(f"Loaded grid sizes for {len(rows)} / {len(indices)} samples")

# ── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = "z_out/qm9_grid_sizes.csv"
Path("z_out").mkdir(exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved → {csv_path}")

# ── Extract arrays ───────────────────────────────────────────────────────────
d_nx = np.array([r["data_nx"] for r in rows])
d_ny = np.array([r["data_ny"] for r in rows])
d_nz = np.array([r["data_nz"] for r in rows])
d_total = np.array([r["data_voxels"] for r in rows])

l_nx = np.array([r["label_nx"] for r in rows])
l_ny = np.array([r["label_ny"] for r in rows])
l_nz = np.array([r["label_nz"] for r in rows])
l_total = np.array([r["label_voxels"] for r in rows])

# ── Plot 1: Histograms of nx, ny, nz — data vs label ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (d_arr, l_arr, name) in zip(axes, [(d_nx, l_nx, "nx"), (d_ny, l_ny, "ny"), (d_nz, l_nz, "nz")]):
    ax.hist(d_arr, bins=40, alpha=0.6, color="steelblue", edgecolor="black", label=f"data (mean={d_arr.mean():.1f})")
    ax.hist(l_arr, bins=40, alpha=0.6, color="coral", edgecolor="black", label=f"label (mean={l_arr.mean():.1f})")
    ax.set_xlabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count")
fig.suptitle("Grid Dimensions: Data vs Label", fontsize=14)
plt.tight_layout()
plt.savefig("z_out/qm9_grid_dims_hist.png", dpi=150, bbox_inches="tight")

# ── Plot 2: Total voxels histogram — data vs label ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(d_total, bins=50, alpha=0.6, color="steelblue", edgecolor="black", label=f"data (mean={d_total.mean():.0f})")
ax.hist(l_total, bins=50, alpha=0.6, color="coral", edgecolor="black", label=f"label (mean={l_total.mean():.0f})")
ax.set_xlabel("Total voxels (nx × ny × nz)")
ax.set_ylabel("Count")
ax.set_title("Total Voxel Count: Data vs Label")
ax.legend()
plt.tight_layout()
plt.savefig("z_out/qm9_total_voxels_hist.png", dpi=150, bbox_inches="tight")

# ── Plot 3: Pairwise scatter (label) ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [(l_nx, l_ny, "nx", "ny"), (l_nx, l_nz, "nx", "nz"), (l_ny, l_nz, "ny", "nz")]
for ax, (a, b, na, nb) in zip(axes, pairs):
    ax.scatter(a, b, s=1, alpha=0.15, color="teal")
    ax.set_xlabel(na)
    ax.set_ylabel(nb)
    ax.set_title(f"{na} vs {nb}")
    ax.set_aspect("equal")
    lim = [min(a.min(), b.min()) - 2, max(a.max(), b.max()) + 2]
    ax.plot(lim, lim, "r--", lw=0.8, alpha=0.5)
fig.suptitle("Pairwise Grid Dimension Scatter (label)", fontsize=14)
plt.tight_layout()
plt.savefig("z_out/qm9_grid_scatter.png", dpi=150, bbox_inches="tight")

# ── Plot 4: Upscale ratio (label / data) per axis ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (d_arr, l_arr, name) in zip(axes, [(d_nx, l_nx, "nx"), (d_ny, l_ny, "ny"), (d_nz, l_nz, "nz")]):
    ratio = l_arr / d_arr
    ax.hist(ratio, bins=40, alpha=0.7, color="mediumpurple", edgecolor="black")
    ax.set_xlabel(f"label_{name} / data_{name}")
    ax.set_title(f"{name} ratio (mean={ratio.mean():.2f})")
axes[0].set_ylabel("Count")
fig.suptitle("Upscale Ratio per Axis (label / data)", fontsize=14)
plt.tight_layout()
plt.savefig("z_out/qm9_upscale_ratio.png", dpi=150, bbox_inches="tight")

# ── Summary stats ────────────────────────────────────────────────────────────
print(f"\n{'':─<60}")
print(f"{'':16s} {'min':>6} {'max':>6} {'mean':>8} {'std':>8} {'median':>8}")
print(f"{'':─<60}")
for name, arr in [
    ("data_nx", d_nx), ("data_ny", d_ny), ("data_nz", d_nz), ("data_voxels", d_total),
    ("label_nx", l_nx), ("label_ny", l_ny), ("label_nz", l_nz), ("label_voxels", l_total),
]:
    print(f"{name:<16s} {arr.min():>6} {arr.max():>6} {arr.mean():>8.1f} {arr.std():>8.1f} {np.median(arr):>8.0f}")

print(f"\nPlots saved to z_out/")
