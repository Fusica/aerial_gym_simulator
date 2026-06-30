# 代码压缩风格

这个文件记录当前 `pursuit_risk` 目录后续重构时采用的压缩标准。

## 目标

压缩的目标不是追求最短代码，而是让训练链路更直接：

- 数据格式正确时，代码按固定 contract 直接运行。
- 错误数据尽早暴露，不用兼容路径掩盖问题。
- 逻辑能在当前文件内清楚表达时，不拆 helper。
- 保留真实语义边界，删除便利性包装。

## 优先压缩的地方

优先删这些内容：

- 旧数据格式 fallback，例如旧 cache layout、旧 manifest、旧 import fallback。
- 单调用 helper，并且 helper 只是转发参数或包装一行逻辑。
- 为 debug 或临时小样本训练保留的参数，例如 runtime dataset 的 `max_anchors`。
- 重复类型转换，例如来自 argparse、固定 schema、固定 npy/npz contract 的 `int()`、`float()`、`bool()`。
- 不进入训练、loss、metric、split 的 metadata 返回字段。
- 只为了友好报错存在的 shape guard，如果后续 tensor 操作本身会自然失败。
- 不被其它脚本消费的便利产物和 summary 文件。

## 可以保留的地方

这些不是冗余：

- 真实数据边界，例如 raw branch `.npz` 采集和 dense branch `.npy` cache。
- batch 结构转换，例如 branch 的 `[B,M,...] -> [B*M,...]` 展平和 `group_ids`。
- 防止 train/val 泄漏的 grouped split，例如 branch 按 `(update, source_episode_idx)` 分组。
- 物理或实验需要调参的 knob，例如数据采集阶段的 anchor 数、horizon、stride。
- cache 构建阶段需要的 annotation，例如 split 和 labels。
- 跨 numpy/json/torch 边界且确实改变类型语义的转换。

## 数据假设

当前风格默认数据已经符合 pipeline contract：

- raw branch loader 只读固定 branch index、metadata 和 anchor LiDAR。
- runtime branch loader 只读固定 `.npy` cache 路径。
- target 字段来自 `TARGET_SPECS`。
- runtime loader 不依赖 `manifest.json`。
- runtime batch 只包含模型训练需要的字段。

如果数据坏了，应该让代码直接报错，而不是走兜底路径继续训练。

## Helper 判断

一个 helper 是否应该存在，看它有没有真实语义：

- 只是包装一行或只被调用一次：倾向内联。
- 隐藏真实边界：保留。
- 多处共享同一 batch contract：可以保留。
- 为未来扩展预留：删除，未来需要时再加。

当前可接受的 helper 示例：

- `move_batch_to_device()`：服务 nested `targets` batch。
- `flatten_branch_batch()`：保留 branch ranking 的同 anchor 分组语义。
- `_split_grouped_indices()`：保留防泄漏 split 语义。

## 底线

不能为了压缩牺牲这些：

- 训练语义。
- train/val split 不泄漏。
- model/loss 的输入输出 contract。
- 数据采集阶段可复现实验所需的元数据。
- Python 3.8.20 兼容性。
- 代码能让后续读者看出数据从哪里来、进入模型前是什么形状。

## 默认审查顺序

后续检查文件时按这个顺序：

1. 先确认代码是否在当前 runtime 路径上。
2. 删除旧兼容、fallback、便利产物。
3. 删除未使用字段、未使用 import、未使用参数。
4. 内联薄 helper。
5. 删除重复类型转换。
6. 最后才考虑结构调整。

如果某段代码只是为了“更稳”，但当前 contract 已经保证输入正确，就删。
