# Pursuit Risk 数据集

这个目录用于训练 action-conditioned RiskNet，目前包含两种数据组织方式。

## Behavior Cache

`prepare_cache.py` 会把 schema7 behavior rollout 转成固定 cache 文件：

- `data/lidar.npy`
- `data/s_red.npy`
- `data/action.npy`
- `annotation/labels/*.npy`
- `annotation/episode_split/{train,val,test}_indices.npy`

`PursuitRiskBehaviorDataset` 是训练时实际使用的 loader。它返回的 batch 只包含：

- `lidar`：堆叠后的 LiDAR 图像张量
- `s_red`：ego 观测/状态向量
- `action`：实际执行的动作
- `targets`：来自 `TARGET_SPECS` 的监督风险标签

当前主用 behavior cache：

- 路径：`runs/risk_cache/PE_20260520_110828_schema7_k3_s2_128x384`
- 来源：`PE_20260520_110828` schema7 rollout cache
- 样本数：`325651`
- episode 数：`500`
- update 数：`15`，范围 `1..1300`
- LiDAR：`K=3`，resize 为 `128x384`
- sample stride：`2`
- split：train `270191` samples / `410` episodes，val `25748` samples / `45` episodes，test `29712` samples / `45` episodes

## Branch Dataset

branch 数据保存的是同一个 anchor state 下的多组 candidate action。`PursuitRiskBranchDataset` 读取：

- `branch_index.jsonl`
- 每个 anchor 对应的 metadata `.npz`
- 共享的 anchor LiDAR `.npz`

它返回的 batch 只包含：

- `lidar`：anchor 的 LiDAR stack
- `s_red`：anchor 的 ego 观测/状态向量
- `candidate_action`：候选动作集合
- `targets`：每个 candidate action 对应的监督风险标签

`flatten_branch_batch()` 会把 `[B, M, ...]` 的 candidate batch 展平成模型输入，同时返回 `group_ids`，这样 branch ranking loss 仍然能在同一个 anchor 内比较不同 candidate action。

当前主用 branch dataset：

- 路径：`runs/risk_dataset/D0_branch_full_v1_10ckpt_500ep_h150_k3_m16`
- 来源：`PE_20260520_110828` 的 `10` 个 checkpoint，每个 checkpoint 计划采 `50` episodes
- planned episodes：`500`
- 实际产生 anchor 的 source episode group：`485`
- anchor 数：`3010`
- 每个 anchor 的 candidate action 数：`16`
- candidate 样本数：`48160`
- update 列表：`170, 220, 300, 350, 464, 610, 754, 900, 1110, 1300`
- 采样方式：`risk_edge`，`horizon=150`，`anchor_stride=25`，每个 source episode 最多 `8` anchors
- anchor 可见率：`3010/3010`
- anchor detectable：`2851/3010`
- label 有连续值差异的 anchors：`2185/3010`
- binary loss 有差异的 anchors：`1000/3010`
- 默认 split（`val_fraction=0.1, seed=1`）：train `2714` anchors / `43424` candidate samples，val `296` anchors / `4736` candidate samples

## Contract

`contract.py` 负责固定的模型/数据契约：

- `STATE_DIM`
- `ACTION_DIM`
- `TARGET_SPECS`
