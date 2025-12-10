## 训练运行摘要（paper_mnist_rp, v2)

- 开始时间: Tue 02 Sep 2025 10:23:02 PM CDT
- 设备: cuda
- 日志: /home/hding22/binary/training_results_archive/train_paper_rp_long_v2.log

### 数据与规模
- 训练集: 51000
- 验证集: 9000
- 测试集: 10000
- 总参数量: 144,864

### LBP 配置
- num_patterns: 2
- num_points: 8
- window: 5
- share_across_channels: True
- mode: bits
- alpha_init: 0.2
- learn_alpha: True
- offset_init_std: 0.8

### 网络与融合
- stages: 3
- channels_per_stage: [64, 128, 192]
- downsample_at: [1, 2]
- fusion_type: rp_paper
- rp_config:
  - fusion_type: rp_paper
  - n_bits_per_out: 4
  - seed: 42
  - threshold: None
  - tau: 0.5
  - use_ste: True

### 论文口径指标（LBP+融合层）
- size_bytes: 588 (0.57 KB)
- GOPs: 0.000781
- ops(cmp/add/mul): 216384 / 564480 / 0

### 优化与训练
- 计划轮数: 220
- 训练迭代/轮（示例）: 399
- 参数组计数: offsets=3, alpha=3, gates=0, head=6, other=18

### 动态快照（训练早期节选）
- Alpha=1.5000, Tau=3.0000, Alive(hard)=0.273, Alive(soft)=0.419
- Alpha=1.4734, Tau=2.9631, Alive(hard)=0.283, Alive(soft)=0.422
- Alpha=1.4474, Tau=2.9267, Alive(hard)=0.315, Alive(soft)=0.419
- Alpha=1.4217, Tau=2.8907, Alive(hard)=0.360, Alive(soft)=0.428

### 结果
- 最佳验证准确率: 98.89%
- 最终模型: ./outputs_mnist_original/final_model.pth
