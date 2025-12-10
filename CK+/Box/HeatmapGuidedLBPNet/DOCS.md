# LBPNet 项目说明

## 概览
- 目标：在不使用数据增强的前提下，基于可学习 LBP + MAC-free RP 融合，完成 MNIST 分类，并提供训练期软阈值/推理期硬阈值的完整实现，支持 STE 与温度退火。
- 主要模块：
  - `lbpnet/layers/lbp_layer.py`：LBP 比特生成（训练软/推理硬，支持 STE，alpha 温度退火，连续偏移映射，半径正则）。
  - `lbpnet/layers/rp_layer.py`：随机投影 + 门控（训练软/推理硬，支持 STE，tau 温度退火，alive_ratio 统计）。
  - `lbpnet/blocks/macfree_block.py`：组合 LBP 与 RP，支持 alpha/tau 更新、STE 开关。
  - `lbpnet/models/lbpnet_base.py`：模型搭建、stem、stages、head，与全局 alpha/tau/STE 注入接口。
  - `train_original_model.py`：训练脚本（温度退火、三阶段冻结、日志、预测直方图与模型保存）。

## 关键功能
### 1. 训练软阈值 + 推理硬阈值 + STE
- LBP 比特：
  - 训练：`sigmoid((sample-anchor)/alpha)`，可选 `STE`（前向硬，反向软）。
  - 推理：`(sample > anchor).float()` 硬比较。
- 门控（RP）：
  - 训练：`sigmoid(logit/tau)`，可选 `STE`。
  - 推理：硬门控 `(logit > 0).float()`。
- 退火：`alpha`、`tau` 支持指数/线性退火；训练日志显示。

### 2. 偏移与正则
- 偏移连续映射：`offsets = radius * tanh(raw/radius)`，避免硬 clamp 的零梯度；窗口内安全。
- 半径对称正则：`(r - target_radius)^2` 引导朝目标半径收敛（可选）。

### 3. 非负权重与稳定性
- `pattern_weights` 前向经 `softplus` 保证非负，避免“反向投票”。
- RP 投影矩阵按输入维度归一化，保证尺度稳定。

### 4. 训练策略
- 三阶段：
  1) 冻结 `offsets` 与 `gate_logits`（若设置）；
  2) 解冻 `gate_logits`；
  3) 全部解冻联合训练。
- 日志：每 epoch 打印 `alpha/tau/AliveRatio(hard/soft)/验证预测直方图`。
- 最佳模型自动保存；早停按耐心值。

## 目录结构
- `lbpnet/layers/lbp_layer.py`：LBP 层实现，含 `lbp_binarize`、`update_alpha` 安全更新、base grid 缓存、连续偏移+对称正则、非负权重。
- `lbpnet/layers/rp_layer.py`：RP 层实现，含 `gate_activate`、正偏置 `gate_logits`、alive_ratio(soft/hard)。
- `lbpnet/blocks/macfree_block.py`：Block 组合，提供 `update_alpha/update_tau/set_ste`。
- `lbpnet/models/lbpnet_base.py`：搭建、前向与注入接口。
- `train_original_model.py`：训练脚本，含温度退火、冻结策略、日志与保存。

## 运行
```
python train_original_model.py
```
- 主要参数在脚本内 `config` 中：
  - `lbp_layer.num_patterns/num_points`：LBP 采样配置；
  - `blocks.channels_per_stage`：网络宽度；
  - `head.hidden/dropout_rate`：分类头；
  - `temp_schedule`：alpha/tau 退火；
  - `freeze_schedule`：冻结 epoch。

## 诊断与常见问题
- 验证精度上不去：
  1) 门控初期全 0.5：已将 `gate_logits` 初始化为 0.2，提高 `AliveRatio`；可增大学习率/缩短冻结时长；
  2) 半径锁边：已改连续映射 `tanh` 避免硬边界零梯度；
  3) 权重为负导致不稳定：已改非负 `softplus`；
  4) 退火太快/太慢：调 `temp_schedule`；
  5) 模型容量不足：增大 `num_points/num_patterns` 与通道数；
  6) 学习率过小：提高 `lr` 或使用分组学习率（偏移/门控更大学习率）。
- 观察门控：
  - 训练日志中的 `AliveRatio(hard/soft)` 应逐步上升至 0.4~0.8 区间；若长期 ~0，说明门控未被有效训练，可：
    - 降低 `tau_start`，或提升 `gate_logits` 初始偏置；
    - 对 `gate_logits` 使用更大的学习率组。

## 提升到 99%+（无数据增强）的建议路径
1) 先做“单批过拟合自测”（固定 32 张，速达 99%+，验证链路正确）。
2) 逐步扩大模型容量（已调高）；必要时再增加 1 个 stage 或更宽 head。
3) 偏移半径目标合理设为 `~(window-1)/4` 到 `(window-1)/3`，调 `constraint_weight` 至 1e-3~1e-2 范围；
4) 学习率：`lr` 至 3e-4~1e-3，偏移与门控单独更高的 lr 组；
5) 退火：`alpha_start=1.5->alpha_end=0.03`; `tau_start=3.0->tau_end=0.2`；
6) 训练日程：总 epoch 150~200，耐心值提高；
7) 若验证仍瓶颈，用 EMA/更强正则（仅 head）与更小 dropout；
8) eval() 下严格硬阈值，确认训练/推理路径一致（除软/硬差异）。

## 变更摘要（本次修复）
- 修复 `update_alpha` 破坏 buffer 的问题；
- 修复 `extra_repr` 张量格式化异常；
- 偏移改连续映射 `tanh` 并使用对称半径正则；
- `pattern_weights` 非负化；
- 加入 base grid 缓存减少 meshgrid 开销；
- 门控正偏置初始化与 soft/hard alive_ratio 日志；
- 训练脚本清理重复 alpha 调度，加入三阶段冻结与更强退火；
- 日志增加预测直方图与 alive_ratio(soft/hard)。

## 后续工作
- 加入“单批 32 张过拟合自测”命令；
- 针对 `gate_logits` 与 `offsets_raw` 设置独立更大学习率组；
- 进一步调参并记录达到 99%+ 的全套配置与日志。
