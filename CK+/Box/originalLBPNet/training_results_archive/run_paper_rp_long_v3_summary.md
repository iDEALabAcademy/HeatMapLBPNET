# Paper MNIST RP 训练总结

- 预设: paper_mnist_rp
- 融合方式: rp_paper
- 最近Epoch: 220/220
- 训练损失: 0.3237
- 训练准确率: 99.55%
- 验证损失: 0.3740
- 验证准确率: 97.32%
- 最佳验证准确率: 97.71%
- AliveRatio(hard): 0.171
- 最终模型: /home/hding22/binary/training_results_archive/outputs_paper_rp_long_v3/final_model.pth
- 日志: /home/hding22/binary/training_results_archive/train_paper_rp_long_v3.log
- 生成时间: 2025-09-03 20:16:17

### LBP 配置
- num_patterns: 2
- num_points: 8
- window: 3
- share_across_channels: True
- mode: bits
- alpha_init: 0.2
- stages: 3
- channels_per_stage: [39, 40, 80]
- downsample_at: [1, 2]
- fusion_type: rp_paper
- rp_config: { n_bits_per_out: 4, seed: 42, use_ste: True, threshold: None }
- head: { hidden: 512, use_bn: True, num_classes: 10 }
- image_size: 28

### 论文口径指标（LBP+融合层）
- size_bytes: 2928 (2.86 KB)
- GOPs: 0.000339
- ops(cmp/add/mul): 105840 / 232848 / 0
