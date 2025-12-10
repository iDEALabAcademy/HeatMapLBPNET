# Paper MNIST RP（全量 60k）配置说明

- 预设: `paper_mnist_rp_full60k`
- 源文件: `lbpnet/train_original_model.py` 中的 `PRESETS["paper_mnist_rp_full60k"]`
- 适用数据: 官方 MNIST 训练集 60k（不保留验证集），测试集 10k

## 当前成果（Results）

- **数据规模**: train=60000, val=0, test=10000
- **参数量（Total parameters）**: 31,629
- （原）论文口径 Size（仅 LBP+RP）: 384 Bytes（0.38 KB）
- （原）论文口径 GOPs（仅 LBP+RP）: 0.000367
- （原）论文口径 ops（cmp/add/mul）: 105840 / 261072 / 0
- **监控最佳准确率（无 val 时等价于训练监控）**: 95.53%
- **最终测试准确率（硬路径 + BN 重校准）**: 96.48%
- **日志**: `/home/hding22/binary/training_results_archive/train_paper_rp_full60k_v1.log`
- **模型文件**: `./outputs_mnist_original/best_model.pth`、`./outputs_mnist_original/final_model.pth`

### 配置更新（2025-09-08）

- 已将 `rp_config.n_bits_per_out` 由 4 调整为 8，用于对齐“约 0.0007 GOPs”的论文口径预算。
- 重新测得（仅 LBP+RP）：
  - **Size**: 5,472 Bytes（5.35 KB）
  - **GOPs**: 0.000677
  - **ops（cmp/add/mul）**: 105,840 / 571,536 / 0
  - 说明：LBP 比较次数不变；RP 的加法随 `k=n_bits_per_out` 线性增加，故 GOPs 约翻至 ~0.00068–0.00070。

## 模型结构（Architecture）

- LBP 层（局部二值模式）
  - **num_patterns=2**: 每像素使用两组 LBP 模式。
  - **num_points=8**: 每组采 8 个邻域点（经典 3×3 LBP-8）。
  - **window=3**: LBP 感受野窗口大小。
  - **share_across_channels=True**: 采样/偏移在所有通道共享（符合论文设定）。
  - **mode="bits"**: 输出为二值比特通道。
  - **alpha_init=0.2**: LBP 温度初值（越小越硬）；训练中按计划退火到更小值。
  - **learn_alpha=True**: 允许学习/更新 alpha。
  - **offset_init_std=0.8**: 采样偏移初始化标准差。

- Blocks（3 个阶段）
  - **stages=3**
  - **channels_per_stage=[39, 40, 80]**: 各阶段输出通道数。
  - **downsample_at=[1, 2]**: 在第 1、2 阶段后进行 2× 下采样（从 0 开始计数）。
  - **fusion_type="rp_paper"**: 论文版随机映射融合（MAC-free：定长比特路由 + popcount + 硬阈值，反传用 STE）。
  - **rp_config**
    - **n_bits_per_out=4**: 每个输出通道聚合的比特数（路由位宽）。
    - **seed=42**: 固定随机映射种子，保证可复现实验。
    - **threshold=None**: 使用实现内置的硬阈值（前向硬），反传使用 STE。
    - **tau=3.0**: 门控梯度温度基准（仅影响梯度形状，不改变硬前向）。
    - **use_ste=True**: 启用直通估计器（STE）。

- 分类头（Head）
  - **hidden=256**: 中间维度。
  - **dropout_rate=0.0**: Dropout 概率。
  - **num_classes=10**: MNIST 十类。
  - **use_bn=True**: 线性层后使用 BN。

## 训练与优化（Training & Optimization）

- training
  - **epochs=300**: 总训练轮数。
  - **batch_size=256**: 批大小。
  - **lr=1e-3**: 基础学习率。
  - **weight_decay=1e-4**: 权重衰减。
  - **lr_scheduler="cosine"**: 余弦退火学习率。
  - **warmup_epochs=5**: 线性预热轮数。
  - **patience=9999 / min_delta=5e-4**: 等价于不触发早停（因无验证集）。
  - **label_smoothing=0.1**: 标签平滑。

- 优化器与参数组（optim / optimizer_groups）
  - **type=AdamW**，betas=(0.9,0.999)。
  - **lr_mult**: `gates=2.0, offsets=2.0, base=1.0`（门控/偏移更大学习率）。
  - 细分组：
    - `lbp_offsets`: lr×5.0，wd×0.0（加速学习偏移，避免正则抑制）。
    - `lbp_alpha`: lr×1.0，wd×0.0（控制温度）。
    - `rp_gates`: lr×3.0，wd×0.0（提升门控学习速度）。
    - `classification_head`: lr×0.7，wd×1.0（头部略降学习率）。

- 其它
  - **gradient_clipping={enabled: True, max_norm: 1.0}**: 防梯度爆炸。
  - **amp=True**: 启用混合精度训练。

## 温度/冻结/稳定性/硬化（Scheduling & Stability）

- 温度调度（temp_schedule, cosine）
  - **alpha: 1.5 → 0.10**（逐步变硬；等价 `ste_scale_bits=1/alpha`）。
  - **tau: 3.0 → 0.90**（仅映射为门控 STE 缩放，不改变硬前向判定）。

- 冻结策略（freeze_schedule）
  - **freeze_offsets_epochs=8**: 前 8 个 epoch 冻结偏移/alpha。
  - **freeze_gates_extra_epochs=4**: 门控额外延长冻结 4 个 epoch。
  - **freeze_gate_epochs=10**: 门控总冻结至第 10 个 epoch。

- 稳定性守护（stability）
  - **collapse_guard=True**，滑窗=5，**acc_drop_thr=0.05**，**alive_upper=0.26**：
    - 若精度连续明显回撤且门控激活率偏高，则临时“软化”并短期冻结 alpha，避免塌缩。

- 硬化微调（hard_finetune）
  - **enable=True, start_epoch=260**：末段进入硬路径微调。
  - **lr_mult=0.1**：降低学习率。
  - **hard_forward_bits=True, hard_forward_gates=True**：前向全硬，反传用 STE。
  - **bn_mode="track"**：硬化期使用 BN 追踪；
  - **bn_recal_batches=200**：评估前在训练集上做 BN 重校准。

- RP 层额外参数（rp_layer）
  - **gate_logits_init=0.3**：门控偏置初始值（正值提升早期开门率，有助于稳定）。

- 稀疏/边际正则
  - **alive_ratio_reg={enable: True, target: 0.5, weight: 1e-3}**：控制全局激活率接近 0.5。
  - **gate_margin={enable: True, margin: 1.0, weight: 1e-3}**：推开门控边界，减少临界抖动。

## 数据与复现（Data & Reproducibility）

- **val_size=0**：使用官方 60k 全量训练，不保留验证集；测试在官方 10k。
- **split_seed=42, stratified=True**：若切分时保持分层与可复现（此预设 val_size=0）。
- **num_workers=4, pin_memory=True**：DataLoader 性能设置。
- **image_size=28**：输入分辨率。
- **seed=42, deterministic=True, benchmark=False**：固定随机性。

## 论文口径（Size / GOPs）说明

- 计算口径仅统计 LBP 比较与融合（RP/1×1）操作；不计入残差、BN、ReLU、池化、分类头等。
- `n_bits_per_out=4` 与 `channels_per_stage=[39,40,80]`、`window=3` 在 MNIST 28×28 下，GOPs 在 10^-4 量级（约 0.0003~0.0004），满足超低算力预算。

## 关键含义速查（Cheat Sheet）

- **alpha**: LBP 比特硬度的“温度”，越小越硬；训练用退火（cosine）。
- **tau**: 门控梯度“温度”，仅影响 STE 形状，前向始终硬判（g_hard）。
- **n_bits_per_out**: 每个输出通道聚合的比特数（位宽/路由宽度）。
- **alive_ratio**: 门控开启比例；通过正则与守护保持在健康区间。
- **BN 重校准**: 硬路径下，用训练集若干批刷新 BN 统计，贴近推理分布。

—— 本文件由当前工程配置自动导出，若有改动（如通道数、位宽、温度计划），请同步更新本说明或重新导出。


