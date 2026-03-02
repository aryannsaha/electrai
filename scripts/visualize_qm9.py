"""Visualize QM9 electron density data.

Usage:
    uv run python scripts/visualize_qm9.py --index 1
    uv run python scripts/visualize_qm9.py --index 1 --compare
    uv run python scripts/visualize_qm9.py --index 1 --isosurface
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

QM9_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9")
BOHR_TO_ANG = 1.88973


def load_sample(root: Path, folder: str, index: int):
    """Load density and grid sizes for a given sample."""
    sample_dir = root / folder / f"dsgdb9nsd_{index:06d}"
    grid_sizes = np.loadtxt(sample_dir / "grid_sizes_22.dat", dtype=int)
    rho = np.load(sample_dir / "rho_22.npy").reshape(grid_sizes)
    rho *= BOHR_TO_ANG**3  # convert a.u. to e/A^3
    return rho, grid_sizes


def load_atoms(root: Path, index: int):
    """Load atomic positions from centered.xyz."""
    xyz_file = root / "label" / f"dsgdb9nsd_{index:06d}" / "centered.xyz"
    if not xyz_file.exists():
        return None, None
    lines = xyz_file.read_text().strip().split("\n")
    n_atoms = int(lines[0])
    elements = []
    positions = []
    for line in lines[1 : 1 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    return elements, np.array(positions)


def load_box(root: Path, index: int):
    """Load box vectors."""
    box_file = root / "label" / f"dsgdb9nsd_{index:06d}" / "box.dat"
    if not box_file.exists():
        return None
    return np.loadtxt(box_file)


def plot_slices(rho, elements, positions, box, title="Electron Density"):
    """Plot 2D slices through the middle of the 3D density, with atom positions overlaid."""
    mid = [s // 2 for s in rho.shape]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{title}  (shape: {rho.shape})", fontsize=14)

    slice_labels = [
        ("YZ plane (x-mid)", rho[mid[0], :, :], (1, 2)),
        ("XZ plane (y-mid)", rho[:, mid[1], :], (0, 2)),
        ("XY plane (z-mid)", rho[:, :, mid[2]], (0, 1)),
    ]
    axis_names = ["x", "y", "z"]

    for ax, (label, slc, (dim1, dim2)) in zip(axes, slice_labels):
        im = ax.imshow(
            slc.T,
            origin="lower",
            cmap="inferno",
            aspect="equal",
        )
        ax.set_title(label)
        ax.set_xlabel(f"{axis_names[dim1]} (grid)")
        ax.set_ylabel(f"{axis_names[dim2]} (grid)")
        fig.colorbar(im, ax=ax, label="e/A³", shrink=0.8)

        # Overlay atom positions if available
        if positions is not None and box is not None:
            mid_dim = [0, 1, 2]
            mid_dim.remove(dim1)
            mid_dim.remove(dim2)
            slice_dim = mid_dim[0]

            # Convert positions to grid coordinates
            grid_pos = np.zeros_like(positions)
            for i in range(3):
                grid_pos[:, i] = positions[:, i] / box[i, i] * rho.shape[i]

            # Only show atoms near the slice plane (within 1 grid unit)
            slice_coord = grid_pos[:, slice_dim]
            near_slice = np.abs(slice_coord - mid[slice_dim]) < 2.0

            if np.any(near_slice):
                atom_colors = {
                    "C": "lime",
                    "H": "white",
                    "O": "red",
                    "N": "blue",
                    "F": "cyan",
                }
                for idx in np.where(near_slice)[0]:
                    color = atom_colors.get(elements[idx], "yellow")
                    ax.scatter(
                        grid_pos[idx, dim1],
                        grid_pos[idx, dim2],
                        c=color,
                        s=60,
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=5,
                    )

    plt.tight_layout()
    return fig


def plot_comparison(data_rho, label_rho, index):
    """Plot data (low-res) vs label (high-res) side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"QM9 Sample {index}: Data (low-res) vs Label (high-res)", fontsize=14)

    for row, (rho, name) in enumerate([(data_rho, "Data (input)"), (label_rho, "Label (target)")]):
        mid = [s // 2 for s in rho.shape]
        slices = [rho[mid[0], :, :], rho[:, mid[1], :], rho[:, :, mid[2]]]
        titles = ["YZ plane", "XZ plane", "XY plane"]

        vmin = min(s.min() for s in slices)
        vmax = max(s.max() for s in slices)

        for col, (slc, title) in enumerate(zip(slices, titles)):
            ax = axes[row, col]
            im = ax.imshow(slc.T, origin="lower", cmap="inferno", aspect="equal", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name} — {title}")
            fig.colorbar(im, ax=ax, label="e/A³", shrink=0.8)

    plt.tight_layout()
    return fig


def plot_isosurface(rho, elements, positions, box, title="Electron Density Isosurface"):
    """Plot 3D isosurface using matplotlib (no extra deps)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    threshold = rho.max() * 0.15
    verts, faces, _, _ = marching_cubes(rho, level=threshold)

    # Scale vertices to physical coordinates if box available
    if box is not None:
        for i in range(3):
            verts[:, i] = verts[:, i] / rho.shape[i] * box[i, i]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor="none", facecolor="cornflowerblue")
    ax.add_collection3d(mesh)

    # Plot atoms
    if positions is not None:
        atom_colors = {"C": "gray", "H": "white", "O": "red", "N": "blue", "F": "green"}
        atom_sizes = {"C": 120, "H": 60, "O": 100, "N": 100, "F": 80}
        for elem, pos in zip(elements, positions):
            ax.scatter(
                *pos,
                c=atom_colors.get(elem, "yellow"),
                s=atom_sizes.get(elem, 80),
                edgecolors="black",
                linewidths=0.5,
                depthshade=True,
            )

    if box is not None:
        ax.set_xlim(0, box[0, 0])
        ax.set_ylim(0, box[1, 1])
        ax.set_zlim(0, box[2, 2])
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(title)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize QM9 electron density data")
    parser.add_argument("--index", type=int, default=1, help="Sample index (default: 1)")
    parser.add_argument("--root", type=str, default=str(QM9_ROOT), help="QM9 data root directory")
    parser.add_argument("--compare", action="store_true", help="Compare data vs label side by side")
    parser.add_argument("--isosurface", action="store_true", help="Plot 3D isosurface (requires scikit-image)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: qm9_viz_{index}.png)")
    args = parser.parse_args()

    root = Path(args.root)
    idx = args.index
    out = args.output or f"qm9_viz_{idx}.png"

    print(f"Loading QM9 sample {idx}...")
    label_rho, label_sizes = load_sample(root, "label", idx)
    elements, positions = load_atoms(root, idx)
    box = load_box(root, idx)

    print(f"  Label shape: {label_rho.shape}")
    print(f"  Density range: [{label_rho.min():.4f}, {label_rho.max():.4f}] e/A³")
    if elements:
        print(f"  Atoms: {len(elements)} ({', '.join(elements)})")

    if args.compare:
        data_rho, _ = load_sample(root, "data", idx)
        print(f"  Data shape:  {data_rho.shape}")
        fig = plot_comparison(data_rho, label_rho, idx)
    elif args.isosurface:
        fig = plot_isosurface(label_rho, elements, positions, box, title=f"QM9 #{idx} Isosurface")
    else:
        fig = plot_slices(label_rho, elements, positions, box, title=f"QM9 Sample {idx}")

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
