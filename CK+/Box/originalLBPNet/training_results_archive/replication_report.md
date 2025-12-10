## LBPNet (WACV'20, MNIST) 复刻报告

日期: $(date)
运行目录: `/home/hding22/binary/training_results_archive`

### 1. 摘要（是否达标）
- 结论: ⚠️ 未完全达标（验证准确率最高 96.54%，低于论文 99.50%）。
- 亮点:
  - 结构与口径对齐论文：3 个 stage，通道 [39, 40, 80]，LBP 3×3 窗+8 点，RP 融合（bit routing + popcount + threshold，无 MAC）。
  - 训练路径硬前向 + STE 反向；加入温度调度、冷启动冻结、EMA、AMP、梯度裁剪与崩塌守护。
  - 指标统计工具完善：新增统一模型尺寸统计（混合位宽策略）与论文口径的 Size/GOPs。

### 2. 数据与预处理
- 数据集: MNIST 28×28 灰度。
- 划分: 训练=51k / 验证=9k / 测试=10k（论文未定义验证集，采用常用拆分）。
- 预处理: ToTensor + Normalize(mean=0.1307, std=0.3081)。
- 训练增广: RandomCrop(28, padding=2)、RandomRotation(±10°)（可关闭）。
- 随机性: seed=42，CuDNN deterministic=True, benchmark=False。

### 3. 模型结构对照
- Stem: `nn.Identity()`（直接使用原图）。
- LBP 层: num_patterns=2, num_points=8, window=3, share_across_channels=True, 输出 bit-array，硬前向 + STE。
- 融合（RP）: `RPFusionPaper`（bit routing + popcount + threshold），k=4 bits/out，确定性映射（seed=42），前向恒硬，支持 STE 梯度形状。
- Residual: 残差分支带通道对齐与 2×2 AvgPool 下采样（在 stage1→2、2→3）。
- Head: GAP → Linear(hidden) → BN → ReLU → Linear(num_classes)，MNIST 版本 hidden=256。

### 4. 训练与调度
- Optim: AdamW(lr=1e-3, wd=0.01, betas=0.9/0.999)，参数分组 gates/offsets/base。
- LR 调度: Cosine(epochs=220) + 线性 warmup(5)。
- Loss: CrossEntropy + label_smoothing=0.1；偏移正则 + 轻量分布正则。
- AMP: 使能；梯度裁剪: max_norm=0.3。
- EMA: decay=0.999，用 EMA 权重做验证与保存。
- 温度/门控: TemperatureScheduler（cosine）；alpha: 1.5→0.06；tau: 3.0→0.75（映射为 gate STE scale）。
- 冷启动冻结: offsets/alpha 冻结 5 epoch；门控额外冻结 2 epoch；early hardening 两阶段微调可选。
- 崩塌守护: 验证精度短期大幅回撤且 alive_hard 偏高时，临时“软化”并从 best 恢复。

### 5. 评估路径与 BN 校准
- 评估使用硬路径（LBP/RP 都走硬），与训练的硬前向 + STE 一致。
- 硬化期可选 BN 重校准（以硬路径跑若干 batch 更新 running stats）。

### 6. 指标对齐（当前结果 vs 论文）
- 训练日志与输出：
  - v2 日志: `/home/hding22/binary/training_results_archive/train_paper_rp_soft2hard_v2.log`（已完成）
  - v3 日志: `/home/hding22/binary/training_results_archive/train_paper_rp_soft2hard_v3.log`（进行中）
- 最佳验证 Top-1（v2）: 96.54%
- 论文口径 Size/GOPs（仅 LBP+融合）: 来自运行时 `[PAPER]` 行
  - RP: size_bytes=384 (0.38 KB), GOPs≈0.000339
- 通用模型尺寸（含权重/bias/可学习标量，按策略位宽折算）：
  - 工具: `tools/measure_model_size.py`
  - 输出: `/home/hding22/binary/outputs_mnist_original/size_report.{csv,json}`
  - 汇总（best_model.pth）：weight_only=75.30 KB；full_params=78.36 KB；位宽直方图：16bit≈46.28 KB，32bit≈33.96 KB。
- 论文表 3（MNIST）：
  - LBPNet (1×1): Error 0.50%，Size 1.27 MB，#Ops 0.011 GOP
  - LBPNet (RP): Error 0.50%，Size 715.5 Bytes，#Ops 0.0007 GOP
- 本实现差异小结：
  - 准确率偏低（96.5% vs 99.5%）：仍有软→硬退火/BN/门控稀疏度匹配等差异；
  - 论文口径 Size/GOPs 与实现口径一致；通用模型尺寸工具用于更全面的工程口径统计（与论文 Size 定义不同）。

### 7. 消融与对照（规划）
- RP vs 1×1：已提供 `paper_mnist_rp` 与 `paper_mnist_1x1` 预设，建议在相同增广/调度下跑对照，以复现实验量级。
- ROP/正交投影（可选）：可在 RP 矩阵上做近似正交化进行补充消融。

### 8. 一键命令
- 训练（RP 预设，后台运行）：
  ```bash
  cd /home/hding22/binary
  nohup env MODEL_PRESET=paper_mnist_rp PYTHONUNBUFFERED=1 \
    python -u train_original_model.py \
    > /home/hding22/binary/training_results_archive/train_paper_rp_soft2hard_v3.log 2>&1 &
  ```
- 模型尺寸（通用混合位宽统计）：
  ```bash
  python /home/hding22/binary/tools/measure_model_size.py \
    --ckpt /home/hding22/binary/outputs_mnist_original/best_model.pth \
    --out  /home/hding22/binary/outputs_mnist_original \
    --policy /home/hding22/binary/configs/size_policy.yaml
  ```

### 9. 待人工确认项
- 论文是否默认无增广；若严格无增广，需关闭增广复现实验再对齐温度退火与 BN 策略。
- RP 与 LBP 的阈值/半径/偏置是否存在更细的初始化与正则；
- 论文是否采用更长训练或更保守的退火下限；
- 1×1 融合对照的 hidden 维度/头部是否与论文完全一致（本实现 MNIST hidden=256）。

### 10. 关键改动摘要（主要文件）
- `train_original_model.py`: 预设与训练循环重构；温度调度器、EMA、AMP、梯度裁剪、冻结与崩塌守护；仅按 epoch 打印汇总。
- `lbpnet/blocks/macfree_block.py`: 融合层抽象、门控冻结与统计接口。
- `lbpnet/layers/rp_layer.py`, `lbpnet/layers/rp_paper_layer.py`: 硬前向 + STE；确定性映射与门控统计。
- `lbpnet/models/lbpnet_base.py`: 模型骨架与参数收集接口；Head 简化。
- `tools/metrics_paper.py`: 论文口径 Size/GOPs。
- 新增尺寸统计：`utils/model_size.py`、`tools/measure_model_size.py`、`configs/size_policy.yaml`。

### 11. 后续建议
- 分阶段硬化：延长软阶段并抬高 tau 下限，配合更密集的 BN 重校准；
- 更细的门控稀疏正则与 margin 正则；
- 关闭增广跑一版严格论文设置，再在其基础上微调；
- 覆盖 1×1 对照与可选 ROP 消融，形成完整复现实验表。

### 12. 测试覆盖与CI状态
- 当前快速测试（默认执行，均通过）：
  - 数据：形状/范围、种子复现、DataLoader 跨 workers 确定性
  - LBP/RP：比特二值与确定性、RP 形状/门控单调性
  - Block：eval 前向一致
  - 训练评估一致性：BN eval-eval 严格一致
  - 复现健壮性：dummy forward→load、state_dict 向后兼容
  - AMP 一致性：fp32 vs autocast 数值/Top-1 接近
  - STE 梯度健壮性：梯度有限且范数落于合理区间
  - CPU↔GPU 冒烟：Top-1 差 ≤0.3%
  - TorchScript 导出冒烟：与 PyTorch eval 对齐
- 条件性/可选测试：
  - BN 校准敏感度：已实现冒烟（需已训 ckpt），阈值放宽（≤2% 波动）；可在生产权重上收紧
  - FLOPs↔延迟趋势：受硬件/负载噪声影响大，默认跳过；GPU + PERF_TEST=1 可启用
  - ONNX 导出：若缺少 onnxruntime 自动跳过
- CI：提供 `scripts/ci_run_tests.sh`，默认跑所有快速测试并在有 CUDA 时附加 gpu 用例；性能类用例需显式 `PERF_TEST=1`。






