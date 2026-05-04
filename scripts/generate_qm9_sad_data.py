from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

lib = None
elements = None
dft = None
gto = None
tools = None
_eval_rhoG = None
addons = None
atom_hf_pp = None
hf = None


def load_pyscf() -> None:
    global lib, elements, dft, gto, tools, _eval_rhoG, addons, atom_hf_pp, hf
    if lib is not None:
        return

    try:
        from pyscf import lib as pyscf_lib
        from pyscf.data import elements as pyscf_elements
        from pyscf.pbc import dft as pyscf_dft
        from pyscf.pbc import gto as pyscf_gto
        from pyscf.pbc import tools as pyscf_tools
        from pyscf.pbc.dft.multigrid.multigrid_pair import _eval_rhoG as pyscf_eval_rhoG
        from pyscf.scf import addons as pyscf_addons
        from pyscf.scf import atom_hf_pp as pyscf_atom_hf_pp
        from pyscf.scf import hf as pyscf_hf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PySCF is required to generate QM9 data. Install it first with: "
            "python -m pip install pyscf"
        ) from exc

    lib = pyscf_lib
    elements = pyscf_elements
    dft = pyscf_dft
    gto = pyscf_gto
    tools = pyscf_tools
    _eval_rhoG = pyscf_eval_rhoG
    addons = pyscf_addons
    atom_hf_pp = pyscf_atom_hf_pp
    hf = pyscf_hf


def read_filelist(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def read_split_indices(paths: list[Path]) -> set[int]:
    indices: set[int] = set()
    for path in paths:
        split = json.loads(path.read_text())
        for key in ("train", "validation", "test"):
            indices.update(split.get(key, []))
    return indices


def read_sample_ids(args: argparse.Namespace) -> list[int]:
    if args.sample_ids_file is not None:
        return [
            int(line.strip())
            for line in args.sample_ids_file.read_text().splitlines()
            if line.strip()
        ]
    if not args.splits:
        raise ValueError("provide --splits or --sample-ids-file")
    filelist = read_filelist(args.root / "qm9_filelist.txt")
    return sorted(filelist[index] for index in read_split_indices(args.splits))


def build_cell(xyz_path: Path, basis: str, cutoff: int, margin: int, xc: str):
    load_pyscf()
    with xyz_path.open() as fp:
        natom = int(fp.readline())
        fp.readline()
        atom_lines = fp.readlines()[:natom]

    coords = np.array([line.split()[1:4] for line in atom_lines], dtype=float)
    atoms = [" ".join(line.split()[:4]) for line in atom_lines]
    geom_cen = np.mean(coords, axis=0)
    box = np.max(coords, axis=0) - np.min(coords, axis=0) + margin
    box = np.ceil(box * np.sqrt(2 * cutoff) / np.pi / lib.param.BOHR)
    box = np.diag(box / np.sqrt(2 * cutoff) * np.pi * lib.param.BOHR - 1e-4)
    shift = np.diag(box) / 2 - geom_cen
    coords = coords + shift
    atoms = [
        f"{atom.split()[0]} {coord[0]} {coord[1]} {coord[2]}"
        for atom, coord in zip(atoms, coords, strict=True)
    ]

    cell = gto.Cell()
    cell.basis = basis
    cell.ke_cutoff = cutoff
    cell.a = box
    cell.pseudo = f"gth-{xc}"
    cell.atom = atoms
    cell.max_memory = 10000
    cell.precision = 1e-6
    cell.rcut_by_shell_radius = True
    cell.charge = 0
    cell.build()
    return cell, box


def make_mf(cell, xc: str, conv_tol: float):
    load_pyscf()
    multigrid_df = getattr(dft.multigrid, "MultiGridFFTDF2", None)
    if multigrid_df is None:
        multigrid_df = dft.multigrid.MultiGridNumInt2
    df = multigrid_df(cell)
    if not hasattr(df, "kpts"):
        df.kpts = np.zeros(3)
    mf = dft.rks.RKS(cell)
    mf.with_df = df
    mf.conv_tol = conv_tol
    mf.xc = xc
    mf.init_guess = "atom"
    mf.max_cycle = 200
    return mf


def get_init_guess(mf, basis1: str, basis2: str, box, pseudo: str):
    load_pyscf()
    atomic_configuration = elements.NRSRHF_CONFIGURATION
    dm_results = {}
    for atom in mf.cell.atom:
        symbol = atom[0]
        if symbol in dm_results:
            continue

        mol = gto.Cell()
        mol.atom = f"{symbol} 0 0 0"
        mol.charge = 0
        mol.enuc = 0
        mol.cart = False
        mol.basis = basis1
        mol.pseudo = pseudo
        mol.spin = elements.NUC[symbol] % 2
        mol.a = box
        mol.build()

        if mol.nelectron == 1:
            atm_hf = atom_hf_pp.AtomHF1ePP(mol)
            atm_hf.run()
            dm0 = hf.make_rdm1(atm_hf.mo_coeff, atm_hf.mo_occ)
        else:
            atm_hf = atom_hf_pp.AtomSCFPP(mol)
            atm_hf.atomic_configuration = atomic_configuration
            dm0 = atm_hf.get_init_guess(key="1e")

        mol2 = mol.copy()
        mol2.basis = basis2
        mol2.build()
        dm_results[symbol] = addons.project_dm_nr2nr(mol, dm0, mol2)

    slices = mf.cell.aoslice_by_atom()
    dm = np.zeros([mf.cell.nao] * 2)
    for i, atom in enumerate(mf.cell.atom):
        symbol = atom[0]
        p0, p1 = slices[i][2:]
        dm[p0:p1, p0:p1] = dm_results[symbol]
    return dm


def generate_one(
    label_dir: Path,
    out_dir: Path,
    *,
    basis1: str,
    basis2: str,
    cut1: int,
    cut2: int,
    xc: str,
    conv_tol: float,
    margin: int,
    match_label_grid: bool,
    overwrite: bool,
) -> str:
    if (out_dir / "rho_22.npy").exists() and not overwrite:
        return "skip"

    xyz_path = label_dir / "centered.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(xyz_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    cell, box = build_cell(xyz_path, basis1, cut1, margin, xc)
    cell22 = cell.copy()
    cell22.basis = basis2
    cell22.ke_cutoff = cut2
    cell22.build()

    mf22 = make_mf(cell22, xc, conv_tol)
    if match_label_grid:
        label_grid_path = label_dir / "grid_sizes_22.dat"
        if not label_grid_path.exists():
            raise FileNotFoundError(label_grid_path)
        mesh = np.loadtxt(label_grid_path, dtype=int)
        mf22.grids.mesh = mesh
        mf22.with_df.mesh = mesh
        if hasattr(mf22.with_df, "grids"):
            mf22.with_df.grids.mesh = mesh

    dm0 = get_init_guess(mf22, basis1, basis2, box, f"gth-{xc}")
    nelec = np.trace(dm0 @ mf22.get_ovlp())
    if abs(nelec - mf22.cell.nelectron) / mf22.cell.nelectron > 0.01:
        raise ValueError(f"nelectron mismatch: {nelec} vs {mf22.cell.nelectron}")
    dm0 = dm0 / nelec * mf22.cell.nelectron

    rho_g = _eval_rhoG(mf22.with_df, dm0, 1, np.zeros((1, 3)), 0)
    mesh = mf22.with_df.mesh
    ngrids = np.prod(mesh)
    weight = mf22.cell.vol / ngrids
    rho_r = tools.ifft(rho_g.reshape(-1, ngrids), mesh).real * (1.0 / weight)

    np.savetxt(out_dir / "grid_sizes_22.dat", mf22.grids.mesh, fmt="%d")
    np.save(out_dir / "dm_22.npy", dm0)
    np.save(out_dir / "rho_22.npy", rho_r[0])
    return "write"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QM9 SAD input density data/ folders from label/centered.xyz."
    )
    parser.add_argument("--root", type=Path, default=Path("/workspace/data/qm9"))
    parser.add_argument("--splits", type=Path, nargs="*", default=[])
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--basis1", default="gth-szv")
    parser.add_argument("--basis2", default="gth-tzv2p")
    parser.add_argument("--cut1", type=int, default=50)
    parser.add_argument("--cut2", type=int, default=200)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--conv-tol", type=float, default=1e-11)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument(
        "--match-label-grid",
        action="store_true",
        help="Evaluate SAD rho on label/grid_sizes_22.dat so data and label meshes match.",
    )
    args = parser.parse_args()

    sample_ids = read_sample_ids(args)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    print(f"generating data for {len(sample_ids)} samples under {args.root}")
    counts = {"skip": 0, "write": 0}
    for position, sample_id in enumerate(sample_ids, start=1):
        mol_dir = f"dsgdb9nsd_{sample_id:06d}"
        label_dir = args.root / "label" / mol_dir
        out_dir = args.root / "data" / mol_dir
        log_path = out_dir / "scanner.out"
        print(f"[{position}/{len(sample_ids)}] {mol_dir}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = generate_one(
                label_dir,
                out_dir,
                basis1=args.basis1,
                basis2=args.basis2,
                cut1=args.cut1,
                cut2=args.cut2,
                xc=args.xc,
                conv_tol=args.conv_tol,
                margin=args.margin,
                match_label_grid=args.match_label_grid,
                overwrite=args.overwrite,
            )
        counts[result] += 1

    print(f"done: wrote={counts['write']} skipped={counts['skip']}")


if __name__ == "__main__":
    main()
