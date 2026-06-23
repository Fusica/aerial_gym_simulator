"""Dataset loader for cached pursuit LiDAR observability-risk training."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .contract import PRIMARY_RISK_HORIZONS, TARGET_SPECS


def _read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class NpzLruCache:
    def __init__(self, max_items: int = 16):
        self.max_items = max_items
        self.cache: OrderedDict[Path, Dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: Path) -> Dict[str, np.ndarray]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.cache[path] = arrays
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arrays


def create_memmap(path: Path, shape: Tuple[int, ...], dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def resize_lidar(lidar: torch.Tensor, resize_hw: Tuple[int, int], device: torch.device) -> np.ndarray:
    batch, frames, height, width = lidar.shape
    if (height, width) == resize_hw:
        return lidar.numpy()
    x = lidar.reshape(batch * frames, 1, height, width).to(device=device, dtype=torch.float32, non_blocking=True)
    x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
    x = x.reshape(batch, frames, resize_hw[0], resize_hw[1]).to(dtype=torch.float16).cpu().numpy()
    return x


@dataclass
class BehaviorEpisode:
    metadata_path: Path
    dataset_episode_id: int
    update: int


class RawBehaviorDataset(Dataset):
    """Schema7 behavior rollout reader used only for cache construction."""

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        lidar_stack_frames: int,
        sample_stride: int,
        require_horizons: Sequence[int] = PRIMARY_RISK_HORIZONS,
        cache_items: int = 64,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.lidar_stack_frames = lidar_stack_frames
        self.sample_stride = sample_stride
        self.require_horizons = tuple(require_horizons)
        self.chunk_cache = NpzLruCache(cache_items)
        self.metadata_cache = NpzLruCache(cache_items)
        self.episodes: List[BehaviorEpisode] = []
        self.samples: List[Tuple[int, int]] = []

        for row in _read_jsonl(self.dataset_dir / "index.jsonl"):
            ep_idx = len(self.episodes)
            self.episodes.append(
                BehaviorEpisode(
                    metadata_path=self.dataset_dir / row["metadata_relative_path"],
                    dataset_episode_id=row["dataset_episode_id"],
                    update=row["update"],
                )
            )
            meta = self.metadata_cache.get(self.episodes[ep_idx].metadata_path)
            valid = meta[f"label_valid_h{self.require_horizons[0]:03d}"].copy()
            for horizon in self.require_horizons[1:]:
                valid &= meta[f"label_valid_h{horizon:03d}"]
            for frame_idx in np.nonzero(valid)[0][:: self.sample_stride]:
                self.samples.append((ep_idx, frame_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_lidar_frame(self, meta: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
        counts = meta["lidar_chunk_num_frames"]
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        chunk_idx = np.searchsorted(starts, frame_idx, side="right") - 1
        frame_offset = frame_idx - starts[chunk_idx]
        chunk_path = self.dataset_dir / str(meta["lidar_chunk_relative_paths"][chunk_idx].item())
        return self.chunk_cache.get(chunk_path)["lidar_range_norm"][frame_offset]

    def _load_lidar_stack(self, meta: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
        return np.stack(
            [
                self._load_lidar_frame(meta, max(0, frame_idx - offset))
                for offset in range(self.lidar_stack_frames - 1, -1, -1)
            ],
            axis=0,
        )

    def __getitem__(self, index: int) -> Dict:
        ep_idx, frame_idx = self.samples[index]
        meta = self.metadata_cache.get(self.episodes[ep_idx].metadata_path)
        return {
            "lidar": torch.from_numpy(self._load_lidar_stack(meta, frame_idx)),
            "s_red": torch.from_numpy(meta["ego_obs"][frame_idx]),
            "action": torch.from_numpy(meta["behavior_action"][frame_idx]),
            "targets": {
                target_name: torch.as_tensor(np.asarray(meta[npz_key][frame_idx]))
                for target_name, npz_key, _ in TARGET_SPECS
            },
        }


class PursuitRiskBehaviorDataset(Dataset):
    """Memmap cache produced from schema7 behavior data."""

    def __init__(self, cache_dir: Union[str, Path], split: str):
        self.cache_dir = Path(cache_dir)
        split_path = self.cache_dir / "annotation" / "episode_split" / f"{split}_indices.npy"
        self.indices = np.load(split_path, mmap_mode="r")
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
            annotation_dir = self.cache_dir / "annotation"
            label_dir = annotation_dir / "labels"
            self._arrays = {
                "lidar": np.load(data_dir / "lidar.npy", mmap_mode="r"),
                "s_red": np.load(data_dir / "s_red.npy", mmap_mode="r"),
                "action": np.load(data_dir / "action.npy", mmap_mode="r"),
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
            "action": tensor_copy("action"),
            "targets": {
                target_name: tensor_copy(target_name)
                for target_name, _, _ in TARGET_SPECS
            },
        }


class PursuitRiskBranchDataset(Dataset):
    """Same-anchor candidate-action branch dataset."""

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


def build_behavior_episode_split(
    dataset: RawBehaviorDataset,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    by_update = {}
    sample_episode_ids = []

    for episode_index, _ in dataset.samples:
        ep = dataset.episodes[episode_index]
        sample_episode_ids.append(ep.dataset_episode_id)
        by_update.setdefault(ep.update, set()).add(ep.dataset_episode_id)

    rng = np.random.default_rng(seed)
    train_episodes = set()
    val_episodes = set()
    test_episodes = set()
    for _, episode_ids in sorted(by_update.items()):
        shuffled = sorted(episode_ids)
        rng.shuffle(shuffled)
        count = len(shuffled)
        if count >= 3:
            test_count = max(1, round(test_fraction * count))
            val_count = max(1, round(val_fraction * count))
            if test_count + val_count >= count:
                test_count = 1
                val_count = 1 if count > 2 else 0
        else:
            test_count = 0
            val_count = 0
        test_episodes.update(shuffled[:test_count])
        val_episodes.update(shuffled[test_count : test_count + val_count])
        train_episodes.update(shuffled[test_count + val_count :])

    def indices_for(episode_ids) -> np.ndarray:
        mask = np.isin(sample_episode_ids, sorted(episode_ids))
        return np.nonzero(mask)[0]

    return {
        "train": indices_for(train_episodes),
        "val": indices_for(val_episodes),
        "test": indices_for(test_episodes),
    }


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
