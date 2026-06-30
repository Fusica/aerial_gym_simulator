"""Build a dense npy cache from a raw branch risk dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contract import ACTION_DIM, STATE_DIM, TARGET_SPECS
from .dataset import (
    PursuitRiskBranchDataset,
    create_memmap,
    resize_lidar,
    split_branch_indices_by_source,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lidar-resize", type=int, nargs=2, default=(128, 384), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cache-items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    args = parser.parse_args()
    if args.num_workers == 0:
        args.persistent_workers = False
        args.prefetch_factor = None
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"cache output exists and is non-empty: {output_dir}; pass --overwrite to replace arrays")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    annotation_dir = output_dir / "annotation"
    label_dir = annotation_dir / "labels"
    split_dir = annotation_dir / "source_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = PursuitRiskBranchDataset(args.branch_dataset, cache_items=args.cache_items)
    num_anchors = len(dataset)
    first = dataset[0]
    stack_frames, raw_h, raw_w = first["lidar"].shape
    candidate_count = first["candidate_action"].shape[0]
    resize_hw = tuple(args.lidar_resize)
    print(
        f"building branch cache anchors={num_anchors} candidates={candidate_count} "
        f"resize={resize_hw} stack_frames={stack_frames}"
    )

    train_indices, val_indices = split_branch_indices_by_source(dataset, args.val_fraction, args.seed)
    np.save(split_dir / "train_indices.npy", np.asarray(train_indices, dtype=np.int64))
    np.save(split_dir / "val_indices.npy", np.asarray(val_indices, dtype=np.int64))

    arrays = {
        "lidar": create_memmap(data_dir / "lidar.npy", (num_anchors, stack_frames, *resize_hw), np.float16),
        "s_red": create_memmap(data_dir / "s_red.npy", (num_anchors, STATE_DIM), np.float32),
        "candidate_action": create_memmap(
            data_dir / "candidate_action.npy",
            (num_anchors, candidate_count, ACTION_DIM),
            np.float32,
        ),
    }
    for target_name, _, dtype in TARGET_SPECS:
        arrays[target_name] = create_memmap(label_dir / f"{target_name}.npy", (num_anchors, candidate_count), dtype)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    offset = 0
    for step, batch in enumerate(loader):
        lidar = resize_lidar(batch["lidar"], resize_hw, device)
        size = lidar.shape[0]
        end = offset + size
        arrays["lidar"][offset:end] = lidar
        arrays["s_red"][offset:end] = batch["s_red"].numpy()
        arrays["candidate_action"][offset:end] = batch["candidate_action"].numpy()
        for target_name, _, _ in TARGET_SPECS:
            arrays[target_name][offset:end] = batch["targets"][target_name].numpy()
        offset = end
        if step % args.log_interval == 0:
            print(f"cache step={step} written={offset}/{num_anchors}")

    for array in arrays.values():
        array.flush()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "pursuit_lidar_branch_risk_cache",
                "branch_dataset": args.branch_dataset,
                "anchors": num_anchors,
                "candidate_count": candidate_count,
                "raw_lidar_hw": [raw_h, raw_w],
                "lidar_resize": list(resize_hw),
                "stack_frames": stack_frames,
                "train_anchors": len(train_indices),
                "val_anchors": len(val_indices),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"finished branch cache: {output_dir}")


if __name__ == "__main__":
    main()
