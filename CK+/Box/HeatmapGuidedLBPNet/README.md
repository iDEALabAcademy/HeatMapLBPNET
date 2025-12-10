# LBPNet: Local Binary Pattern Neural Network

一个基于可学习局部二值模式(LBP)特征的MAC-free神经网络架构。

## 🚀 特性

- **MAC-free架构**: 使用Random Projection (RP)层实现无乘法累加的特征融合
- **可学习LBP**: LBP采样偏移量通过反向传播自动学习
- **灵活配置**: 支持多种网络架构和超参数配置
- **高效训练**: 支持混合精度训练和梯度裁剪

## 📁 项目结构

```
lbpnet/
├── layers/           # 核心层实现
│   ├── lbp_layer.py # LBP特征提取层
│   └── rp_layer.py  # Random Projection融合层
├── blocks/           # 网络块
│   └── macfree_block.py # MAC-free残差块
├── models/           # 模型架构
│   ├── lbpnet_base.py   # 基础模型
│   ├── lbpnet_rp.py     # RP融合模型
│   └── lbpnet_conv1x1.py # 1x1卷积融合模型
├── data/             # 数据处理
│   └── mnist_dataset.py # MNIST数据集
└── configs/          # 配置文件
    └── default.yaml  # 默认配置
```

## 🏗️ 架构设计

### LBP层 (Local Binary Pattern Layer)
- 可学习的采样偏移量
- 软比较机制，支持梯度反向传播
- 可配置的模式数量和采样点数

### RP层 (Random Projection Layer)
- MAC-free特征融合
- 随机二值权重 {-1, 1}
- 可配置的温度参数

### MAC-free块
- 结合LBP特征提取和RP融合
- 残差连接
- 批量归一化

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### 训练模型

```python
from lbpnet import build_model, get_mnist_datasets

# 创建模型
config = {
    'model': 'lbpnet_rp',
    'lbp_layer': {
        'num_patterns': 1,
        'num_points': 8,
        'window': 5,
        'alpha_init': 0.2
    },
    'blocks': {
        'stages': 3,
        'channels_per_stage': [32, 64, 128]
    },
    'head': {
        'hidden': 512,
        'num_classes': 10
    }
}

model = build_model(config)

# 训练
python train_original_model.py
```

## 📊 性能

在MNIST数据集上的表现：
- **训练准确率**: ~88%
- **验证准确率**: ~85%
- **模型参数**: 可配置，默认约100K参数

## 🔧 配置选项

### LBP层配置
- `num_patterns`: LBP模式数量
- `num_points`: 每个模式的采样点数
- `window`: 采样窗口大小
- `alpha_init`: 软比较初始温度
- `offset_init_std`: 偏移量初始化标准差

### 网络配置
- `stages`: 网络阶段数
- `channels_per_stage`: 每阶段通道数
- `downsample_at`: 下采样位置

### 训练配置
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `lr`: 学习率
- `patience`: 早停耐心值

## 📝 引用

如果您在研究中使用了LBPNet，请引用：

```bibtex
@misc{lbpnet2024,
  title={LBPNet: Local Binary Pattern Neural Network},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/lbpnet}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License
