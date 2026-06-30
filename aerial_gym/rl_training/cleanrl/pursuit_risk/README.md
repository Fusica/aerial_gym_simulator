# Pursuit Risk 数据集

这个目录现在只保留 branch 数据链路。branch 数据表示同一个 anchor state 下的多组 candidate action，以及每个 candidate rollout 后得到的风险标签。

## Raw Branch Dataset

`export_lidar_rollouts.py` 采集 PPO policy pool 的 branch 数据，原始输出保留为 `.npz`：

- `branch_index.jsonl`
- `branches/upd_*/anchor_*/anchor_lidar.npz`
- `branches/upd_*/anchor_*/branch_metadata.npz`
- `manifest.json`

`PursuitRiskBranchDataset` 读取 raw `.npz`，返回：

- `lidar`：anchor 的 LiDAR stack
- `s_red`：anchor 的 ego 观测/状态向量
- `candidate_action`：同一 anchor 下的候选动作集合
- `targets`：每个 candidate action 对应的监督风险标签

当前主用 raw branch dataset：

- 路径：`runs/risk_dataset/D0_branch_full_v1_10ckpt_500ep_h150_k3_m16`
- 来源：`PE_20260520_110828` 的 `10` 个 checkpoint，每个 checkpoint 计划采 `50` episodes
- anchor 数：`3010`
- 每个 anchor 的 candidate action 数：`16`
- candidate 样本数：`48160`
- update 列表：`170, 220, 300, 350, 464, 610, 754, 900, 1110, 1300`
- 采样方式：`risk_edge`，`horizon=150`，`anchor_stride=25`，每个 source episode 最多 `8` anchors

## Branch Cache

`prepare_cache.py` 把 raw branch `.npz` 转成训练读取更快的 dense `.npy` cache：

- `data/lidar.npy`
- `data/s_red.npy`
- `data/candidate_action.npy`
- `annotation/labels/*.npy`
- `annotation/source_split/{train,val}_indices.npy`
- `manifest.json`

`PursuitRiskBranchCacheDataset` 是训练和评估时实际使用的 loader。它按 anchor 返回 `[M, ...]` candidate batch。

`flatten_branch_batch()` 会把 `[B, M, ...]` 展平成模型输入，同时返回 `group_ids`，用于同一个 anchor 内的 ranking loss 和 branch 评估。

## Train And Evaluate

训练入口只读取 branch cache：

```bash
python -m aerial_gym.rl_training.cleanrl.pursuit_risk.train \
  --branch-cache runs/risk_cache/<branch_cache> \
  --output-dir runs/risk_train/<run_name>
```

评估入口是 `evaluate_branch.py`，同样读取 branch cache：

```bash
python -m aerial_gym.rl_training.cleanrl.pursuit_risk.evaluate_branch \
  --run-dir runs/risk_train/<run_name> \
  --branch-cache runs/risk_cache/<branch_cache>
```

## Contract

`contract.py` 负责固定的模型/数据契约：

- `STATE_DIM`
- `ACTION_DIM`
- `TARGET_SPECS`
