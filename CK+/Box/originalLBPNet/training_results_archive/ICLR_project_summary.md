## 项目总结（ICLR / RuleExtraction / NeSy + Places365）

### 项目概览
- **目标**：在现有 NeSy（神经-符号）框架上完成环境搭建、数据准备（GTSRB 与 Places365 子集）、并进行冒烟训练与规则提取，形成可复现实验流程与产物归档。
- **范围**：
  - 项目根目录：`/home/hding22/ICLR`
  - NeSy 代码：`/home/hding22/ICLR/RuleExtraction`
  - Places365 子集：`/home/hding22/ICLR/places365/subsets*`
  - 训练与图表产物：`/home/hding22/ICLR/RuleExtraction/checkpoints` 与 `runs/`

### 关键实现与修复
- CLI 与训练：
  - 统一入口：`nesy/cli.py`（支持 train/compile/eval/paper/e2e）。
  - 训练脚本：`nesy/train.py` 增加 `--imagefolder_train`/`--imagefolder_val`，自动推断类别数并接入通用 ImageFolder 数据流。
  - 数据集模块：`nesy/model/dataset.py` 新增 `create_imagefolder_loaders`（Resize、Norm、DataLoader 等），复用到 NeSy 训练。
- 工具与评估：
  - `tools/rule_compile.py`、`tools/eval_full.py` 修正导入为 `from nesy.model import dataset`。
- Places365 数据准备：
  - 新脚本 `places365_subsets.py`：多镜像下载、断点重试、防路径穿越解压、官方文件列表整合、子集筛选复制。
  - 找到并使用官方 `places365_val.txt`（来自 `zhoubolei/places_devkit`），重建验证子集并替换早期“预测标签”版本。
  - 生成子集规模报告：`/home/hding22/ICLR/places365/subsets_size_report.md`。
- 训练脚本（通用分类）：`/home/hding22/ICLR/train_places_subsets.py`（按子集循环训练 ResNet-18）。

### 环境与运行
- Python：3.13（虚拟环境：`/home/hding22/ICLR/.venv`）
- 框架：PyTorch + torchvision（CUDA 12.1 兼容）
- GPU 监控：`watch -n 5 nvidia-smi`；日志：`nohup`/`tee`/`tail -f`。

### 数据集状态
- GTSRB：完成下载与组织（Kaggle API），可 e2e 冒烟。
- Places365：
  - 大包已下载并解压（`train_256_places365standard.tar`、`val_256.tar`、`test_256.tar`）。
  - 训练子集：`P2, P3_1, P3_2, P3_3, P5, P10` 位于 `places365/subsets/`。
  - 验证子集（官方标签）：位于 `places365/subsets_val_labeled/`。
  - 规模报告：`/home/hding22/ICLR/places365/subsets_size_report.md`。

### NeSy 冒烟训练（Places 子集）
- 统一命令样例：
  - `python -m nesy.cli train --epochs 2 --batch_size 64 \
    --imagefolder_train <train_path> --imagefolder_val <val_path> \
    --save_dir <ckpt_dir> --plots_dir <plots_dir>`
- 已完成的代表性结果（2 epochs，仅供趋势参考）：
  - P3.3：`num_classes=3`；val：`classifier=0.7400`（best），`rules(hamming)=0.1967`，`rules(L1)=0.3167`，`agree_coverage≈0.423`，`agree_acc≈0.740`。
  - P5：`num_classes=5`；val：`classifier=0.5260`（best），`rules(hamming)=0.2800`，`rules(L1)=0.2020`，`agree_coverage≈0.176`，`agree_acc≈0.636`。
  - P10：`num_classes=10`；val：`classifier=0.4470`（best），`rules(hamming)=0.0850`，`rules(L1)=0.0970`，`agree_coverage≈0.078`，`agree_acc≈0.410`。
- 可视化：相关性与使用率热力图（例如 `runs/nesy/plots_places_P*_smoke/epoch_000_*.png`）。

### 规则提取与评估
- 规则编译：`tools/rule_compile.py`，输出示例：`/home/hding22/ICLR/RuleExtraction/rules_out*/`（含 `rules.json` / `rules.pl`）。
- 评估：`tools/eval_full.py`；指标包括分类准确率、规则准确率（Hamming/L1）、一致性覆盖率/准确率、代码本指标（最小 Hamming、位利用度）。
- 对比报告：
  - 烟雾对比表：`/home/hding22/ICLR/runs/smoke_comparison.md`
  - 详细说明：`/home/hding22/ICLR/runs/smoke_report.md`

### 模型主干
- ResNet-18（可启用 ImageNet 预训练作为初始化），结合 NeSy 动态头部与符号规则模块。

### 产物与路径
- Checkpoints（示例）：
  - `checkpoints/nesy_places_P3_3_smoke/{best.pt,last.pt}`
  - `checkpoints/nesy_places_P5_smoke/{best.pt,last.pt}`
  - `checkpoints/nesy_places_P10_smoke/{best.pt,last.pt}`
- 图表：`runs/nesy/plots_places_P*_smoke/`；e2e 评估图：`runs/nesy/eval_*/*png`。
- 规则：`rules_out*/` 目录下的 `rules.json` / `rules.pl`。

### 风险与注意
- 下载镜像可变：已加入多镜像与重试，但上游变更仍可能导致 404，需要备用源。
- 验证集标注：务必使用官方 `places365_val.txt` 的版本，避免预测标签偏差。
- 冒烟设置偏轻：2 个 epoch 仅用于流程验证，不代表最终性能。

### 后续建议（提升性能与可解释）
- 训练：将 epochs 提升至 20–50（P10 建议 50+），保存 best ckpt。
- 学习率与调度：Cosine/OneCycleLR，warmup 3–5 epochs；初始 LR ≈ `0.1 × (batch/256)`。
- 数据增强：RandAugment/ColorJitter/RandomErasing；验证仅 Resize+Normalize。
- 规则头策略：第 5–10 个 epoch 后再启动生长/裁剪，加入稀疏正则以降低相关性。
- 主干扩展：必要时尝试 ResNet-34，或加载 ImageNet 预训练加速收敛与稳定。

### 附：参考命令
```bash
source /home/hding22/ICLR/.venv/bin/activate
cd /home/hding22/ICLR/RuleExtraction
CUDA_VISIBLE_DEVICES=0 python -m nesy.cli train \
  --epochs 50 --batch_size 128 \
  --imagefolder_train /home/hding22/ICLR/places365/subsets/P10 \
  --imagefolder_val /home/hding22/ICLR/places365/subsets_val_labeled/P10 \
  --save_dir checkpoints/nesy_places_P10_exp \
  --plots_dir runs/nesy/plots_places_P10_exp
```


