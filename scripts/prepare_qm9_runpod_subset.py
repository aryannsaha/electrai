from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


DEFAULT_FILELIST = Path("/scratch/gpfs/ROSENGROUP/common/qm9/qm9_filelist.txt")
DEFAULT_SOURCE_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9")
DEFAULT_BASE_SPLIT = Path("examples/QM9/experiment_5/split_4000.json")


def read_filelist(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def read_split(path: Path) -> dict[str, list[int]]:
    split = json.loads(path.read_text())
    return {
        "train": list(split.get("train", [])),
        "validation": list(split.get("validation", [])),
        "test": list(split.get("test", [])),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def split_counts(split: dict[str, list[int]]) -> dict[str, int]:
    return {key: len(value) for key, value in split.items()}


def all_split_indices(split: dict[str, list[int]]) -> set[int]:
    return set(split["train"]) | set(split["validation"]) | set(split["test"])


def make_nested_splits(args: argparse.Namespace) -> None:
    filelist = read_filelist(args.filelist)
    base_split = read_split(args.base_split)
    base_indices = all_split_indices(base_split)

    if len(base_indices) != sum(split_counts(base_split).values()):
        raise ValueError(f"{args.base_split} contains duplicate indices across buckets")

    available = [idx for idx in range(len(filelist)) if idx not in base_indices]
    rng = random.Random(args.seed)
    rng.shuffle(available)

    used_extra = 0
    previous_extra_train = 0
    previous_extra_val = 0

    for target_size in args.sizes:
        base_total = len(base_indices)
        if target_size < base_total:
            raise ValueError(
                f"target size {target_size} is smaller than base split size {base_total}"
            )

        target_val = int(target_size * args.val_frac)
        target_train = target_size - target_val
        target_test = 0

        needed_train = target_train - len(base_split["train"])
        needed_val = target_val - len(base_split["validation"])
        needed_test = target_test - len(base_split["test"])

        if needed_train < 0 or needed_val < 0 or needed_test < 0:
            raise ValueError(
                f"target size {target_size} cannot preserve existing bucket counts "
                f"{split_counts(base_split)} with val_frac={args.val_frac}"
            )

        needed_total = needed_train + needed_val + needed_test
        if needed_total > len(available):
            raise ValueError("not enough remaining filelist entries for requested split")

        extra = available[:needed_total]
        extra_train = extra[:needed_train]
        extra_val = extra[needed_train : needed_train + needed_val]
        extra_test = extra[needed_train + needed_val :]

        if needed_train < previous_extra_train or needed_val < previous_extra_val:
            raise ValueError("requested sizes must be sorted ascending")
        previous_extra_train = needed_train
        previous_extra_val = needed_val
        used_extra = max(used_extra, needed_total)

        split = {
            "train": base_split["train"] + extra_train,
            "validation": base_split["validation"] + extra_val,
            "test": base_split["test"] + extra_test,
        }
        out_path = args.out_dir / f"split_{target_size}.json"
        write_json(out_path, split)

        sample_ids = sorted(filelist[idx] for idx in all_split_indices(split))
        samples_path = args.out_dir / f"samples_{target_size}.txt"
        samples_path.write_text("\n".join(str(sample_id) for sample_id in sample_ids) + "\n")

        print(
            f"wrote {out_path}: "
            f"train={len(split['train'])}, "
            f"validation={len(split['validation'])}, "
            f"test={len(split['test'])}"
        )
        print(f"wrote {samples_path}: samples={len(sample_ids)}")

    print(f"reserved {used_extra} extra indices from deterministic seed {args.seed}")


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"unknown copy mode: {mode}")


def copy_tree_files(src_dir: Path, dst_dir: Path, mode: str) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    for src in src_dir.iterdir():
        if src.is_file():
            copy_file(src, dst_dir / src.name, mode)


def collect_indices(split_paths: list[Path]) -> set[int]:
    indices: set[int] = set()
    for split_path in split_paths:
        indices |= all_split_indices(read_split(split_path))
    return indices


def stage_subset(args: argparse.Namespace) -> None:
    filelist = read_filelist(args.filelist)
    indices = collect_indices(args.splits)
    selected_ids = sorted(filelist[idx] for idx in indices)

    if args.dry_run:
        print(f"would stage {len(selected_ids)} samples into {args.dest}")
        print(f"categories: {', '.join(args.categories)}")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.filelist, args.dest / "qm9_filelist.txt")

    splits_dest = args.dest / "splits"
    splits_dest.mkdir(exist_ok=True)
    for split_path in args.splits:
        shutil.copy2(split_path, splits_dest / split_path.name)

    (args.dest / "selected_indices.txt").write_text(
        "\n".join(str(idx) for idx in sorted(indices)) + "\n"
    )
    (args.dest / "selected_sample_ids.txt").write_text(
        "\n".join(str(sample_id) for sample_id in selected_ids) + "\n"
    )

    for sample_id in selected_ids:
        mol_dir = f"dsgdb9nsd_{sample_id:06d}"
        for category in args.categories:
            src_dir = args.source_root / category / mol_dir
            dst_dir = args.dest / category / mol_dir
            copy_tree_files(src_dir, dst_dir, args.link_mode)

    print(f"staged {len(selected_ids)} samples into {args.dest}")


def verify_subset(args: argparse.Namespace) -> None:
    filelist = read_filelist(args.root / "qm9_filelist.txt")
    indices = collect_indices(args.splits)
    missing: list[Path] = []

    for idx in sorted(indices):
        sample_id = filelist[idx]
        mol_dir = f"dsgdb9nsd_{sample_id:06d}"
        for category in args.categories:
            for filename in args.required_files:
                path = args.root / category / mol_dir / filename
                if not path.exists():
                    missing.append(path)

    print(f"checked samples={len(indices)} categories={','.join(args.categories)}")
    if missing:
        print(f"missing files={len(missing)}")
        for path in missing[:25]:
            print(path)
        if len(missing) > 25:
            print(f"... and {len(missing) - 25} more")
        raise SystemExit(1)

    print("all required files found")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare nested QM9 subsets for RunPod while preserving split indices."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-splits")
    make.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    make.add_argument("--base-split", type=Path, default=DEFAULT_BASE_SPLIT)
    make.add_argument("--out-dir", type=Path, default=Path("examples/QM9/experiment_5"))
    make.add_argument("--sizes", type=int, nargs="+", default=[15000, 25000])
    make.add_argument("--val-frac", type=float, default=0.2)
    make.add_argument("--seed", type=int, default=42)
    make.set_defaults(func=make_nested_splits)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    stage.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    stage.add_argument("--dest", type=Path, required=True)
    stage.add_argument("--splits", type=Path, nargs="+", required=True)
    stage.add_argument("--categories", nargs="+", default=["label"], choices=["data", "label"])
    stage.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    stage.add_argument("--dry-run", action="store_true")
    stage.set_defaults(func=stage_subset)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--splits", type=Path, nargs="+", required=True)
    verify.add_argument("--categories", nargs="+", default=["label"], choices=["data", "label"])
    verify.add_argument(
        "--required-files",
        nargs="+",
        default=["rho_22.npy", "grid_sizes_22.dat"],
    )
    verify.set_defaults(func=verify_subset)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
