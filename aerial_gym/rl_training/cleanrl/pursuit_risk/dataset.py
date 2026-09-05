"""Branch dataset loaders for pursuit LiDAR observability-risk training."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .contract import TARGET_SPECS


def _read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class NpzLruCache:
    def __init__(self, max_items: int = 16):
        self.max_items = max_items
        self.get = lru_cache(maxsize=max_items)(self._load)

    def __getstate__(self):
        return {"max_items": self.max_items}

    def __setstate__(self, state):
        self.__init__(state["max_items"])

    @staticmethod
    def _load(path: Path) -> Dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def create_memmap(path: Path, shape: Tuple[int, ...], dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def resize_lidar(lidar: torch.Tensor, resize_hw: Tuple[int, int], device: torch.device) -> np.ndarray:
    batch, frames, height, width = lidar.shape
    if (height, width) == resize_hw:
        return lidar.numpy()
    x = lidar.reshape(batch * frames, 1, height, width).to(device=device, dtype=torch.float32, non_blocking=True)
    x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
    return x.reshape(batch, frames, resize_hw[0], resize_hw[1]).to(dtype=torch.float16).cpu().numpy()


class PursuitRiskBranchDataset(Dataset):
    """Raw same-anchor candidate-action branch dataset stored as compressed npz."""

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        cache_items: int = 32,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.anchor_cache = NpzLruCache(cache_items)
        self.lidar_cache = NpzLruCache(cache_items)
        self.rows = _read_jsonl(self.dataset_dir / "branch_index.jsonl")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        metadata_path = self.dataset_dir / row["metadata_relative_path"]
        meta = self.anchor_cache.get(metadata_path)
        lidar_path = self.dataset_dir / meta["anchor_lidar_relative_path"].item()
        lidar = self.lidar_cache.get(lidar_path)["lidar_range_norm"]
        return {
            "lidar": torch.from_numpy(lidar),
            "s_red": torch.from_numpy(meta["ego_obs"]),
            "candidate_action": torch.from_numpy(meta["candidate_action"]),
            "targets": {
                target_name: torch.as_tensor(meta[npz_key])
                for target_name, npz_key, _ in TARGET_SPECS
            },
        }


class PursuitRiskBranchCacheDataset(Dataset):
    """Dense branch cache produced from raw branch npz data."""

    def __init__(self, cache_dir: Union[str, Path], split: str):
        self.cache_dir = Path(cache_dir)
        self.indices = np.load(self.cache_dir / "annotation" / "source_split" / f"{split}_indices.npy", mmap_mode="r")
        self.length = len(self.indices)
        self._arrays = None

    def __len__(self) -> int:
        return self.length

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def _open_arrays(self) -> Dict[str, np.ndarray]:
        if self._arrays is None:
            data_dir = self.cache_dir / "data"
            label_dir = self.cache_dir / "annotation" / "labels"
            self._arrays = {
                "lidar": np.load(data_dir / "lidar.npy", mmap_mode="r"),
                "s_red": np.load(data_dir / "s_red.npy", mmap_mode="r"),
                "candidate_action": np.load(data_dir / "candidate_action.npy", mmap_mode="r"),
            }
            for target_name, _, _ in TARGET_SPECS:
                self._arrays[target_name] = np.load(label_dir / f"{target_name}.npy", mmap_mode="r")
        return self._arrays

    def __getitem__(self, index: int) -> Dict:
        arrays = self._open_arrays()
        index = self.indices[index]

        def tensor_copy(name: str):
            return torch.from_numpy(np.array(arrays[name][index], copy=True))

        return {
            "lidar": tensor_copy("lidar"),
            "s_red": tensor_copy("s_red"),
            "candidate_action": tensor_copy("candidate_action"),
            "targets": {
                target_name: tensor_copy(target_name)
                for target_name, _, _ in TARGET_SPECS
            },
        }


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def flatten_branch_batch(batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    lidar = batch["lidar"]
    s_red = batch["s_red"]
    action = batch["candidate_action"]

    batch_size, candidates = action.shape[:2]
    lidar_flat = lidar[:, None, :, :, :].expand(batch_size, candidates, *lidar.shape[1:]).reshape(
        batch_size * candidates, *lidar.shape[1:]
    )
    s_red_flat = s_red[:, None, :].expand(batch_size, candidates, s_red.shape[-1]).reshape(
        batch_size * candidates, s_red.shape[-1]
    )
    action_flat = action.reshape(batch_size * candidates, action.shape[-1])
    targets = {
        key: value.reshape(batch_size * candidates)
        for key, value in batch["targets"].items()
    }
    group_ids = torch.arange(batch_size, device=action.device).repeat_interleave(candidates)
    return lidar_flat, s_red_flat, action_flat, targets, group_ids


def _split_grouped_indices(
    sample_groups: Sequence[object],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    by_group: Dict[object, List[int]] = {}
    for sample_idx, group in enumerate(sample_groups):
        by_group.setdefault(group, []).append(sample_idx)
    groups = list(by_group)
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator).tolist()
    val_group_count = max(1, round(val_fraction * len(groups))) if len(groups) > 1 else 0
    val_groups = {groups[idx] for idx in order[:val_group_count]}
    train_indices = []
    val_indices = []
    for group, indices in by_group.items():
        if group in val_groups:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, val_indices


def split_branch_indices_by_source(
    dataset: PursuitRiskBranchDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    groups = [
        (
            row["update"],
            row["source_episode_idx"],
        )
        for row in dataset.rows
    ]
    return _split_grouped_indices(groups, val_fraction, seed)
