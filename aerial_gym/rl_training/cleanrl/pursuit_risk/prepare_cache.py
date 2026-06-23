"""Build a fast memmap cache for behavior risk-model pretraining.

The source schema7 dataset remains the canonical raw-data artifact.  This script
only creates a training cache with resized float16 LiDAR stacks and aligned
state/action/label arrays so shuffled training does not repeatedly decompress
large `.npz` LiDAR chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contract import (
    ACTION_DIM,
    DEFAULT_LIDAR_STACK_FRAMES,
    STATE_DIM,
    TARGET_SPECS,
)
from .dataset import (
    RawBehaviorDataset,
    build_behavior_episode_split,
    create_memmap,
    resize_lidar,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-dataset", default="runs/risk_dataset/PE_20260520_110828_ladder_v1_500ep_schema7")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lidar-stack-frames", type=int, default=DEFAULT_LIDAR_STACK_FRAMES)
    parser.add_argument("--lidar-resize", type=int, nargs=2, default=(128, 384), metavar=("H", "W"))
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cache-items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"cache output exists and is non-empty: {output_dir}; pass --overwrite to replace arrays")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    annotation_dir = output_dir / "annotation"
    label_dir = annotation_dir / "labels"
    split_dir = annotation_dir / "episode_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    dataset = RawBehaviorDataset(
        args.behavior_dataset,
        lidar_stack_frames=args.lidar_stack_frames,
        sample_stride=args.sample_stride,
        cache_items=args.cache_items,
    )
    num_samples = len(dataset)
    resize_hw = tuple(args.lidar_resize)
    print(
        f"building behavior cache samples={num_samples} "
        f"resize={resize_hw} stack_frames={args.lidar_stack_frames} stride={args.sample_stride}"
    )
    split = build_behavior_episode_split(dataset, args.val_fraction, args.test_fraction, args.seed)

    arrays = {
        "lidar": create_memmap(data_dir / "lidar.npy", (num_samples, args.lidar_stack_frames, *resize_hw), np.float16),
        "s_red": create_memmap(data_dir / "s_red.npy", (num_samples, STATE_DIM), np.float32),
        "action": create_memmap(data_dir / "action.npy", (num_samples, ACTION_DIM), np.float32),
    }
    for target_name, _, dtype in TARGET_SPECS:
        arrays[target_name] = create_memmap(label_dir / f"{target_name}.npy", (num_samples,), dtype)

    for split_name in ("train", "val", "test"):
        np.save(split_dir / f"{split_name}_indices.npy", split[split_name])

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
        arrays["action"][offset:end] = batch["action"].numpy()
        for target_name, _, _ in TARGET_SPECS:
            arrays[target_name][offset:end] = batch["targets"][target_name].numpy()
        offset = end
        if step % args.log_interval == 0:
            print(f"cache step={step} written={offset}/{num_samples}")

    for array in arrays.values():
        array.flush()
    print(f"finished cache: {output_dir}")


if __name__ == "__main__":
    main()
