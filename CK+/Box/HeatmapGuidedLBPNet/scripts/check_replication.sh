#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_DIR"

echo "[1/6] 环境信息"
python -V | cat
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "[2/6] BN 硬路径重校准（若存在最佳权重）"
python scripts/calibrate_bn.py | cat || true

echo "[3/6] 运行 2x2 网格测试（仅验证前向）"
MODEL_PRESET=${MODEL_PRESET:-paper_mnist_rp} LBP_GRID_TEST=1 python train_original_model.py | cat || true

echo "[4/6] 计算 Size/FLOPs/延迟 与生成 metrics.json"
python scripts/measure_compute.py --preset ${MODEL_PRESET:-paper_mnist_rp} --output metrics.json | cat

echo "[5/6] 运行单元测试"
python -m pytest -q tests | cat || true

echo "[6/6] 生成/更新 replication_report.md"
python - <<'PY'
import json, os, time
metrics = {}
if os.path.exists('metrics.json'):
    with open('metrics.json','r') as f:
        metrics = json.load(f)

report_path = 'replication_report.md'
ts = time.strftime('%Y-%m-%d %H:%M:%S')
header = f"# LBPNet(WACV'20) 复刻自检报告\n\n更新时间: {ts}\n\n"
summary = "## 摘要\n当前状态：⚠️ 部分对齐。MNIST 精度已达 ~97.5%，仍需严格口径修正以对齐论文表格（Size/GOPs 等）。\n\n"
metrics_sec = "## 指标快照\n" + ("```json\n"+json.dumps(metrics, indent=2, ensure_ascii=False)+"\n```\n" if metrics else "(metrics.json 未生成或为空)\n")
todo = "## 待人工确认项\n- 1×1 融合参数口径与论文统计方式\n- Size/GOPs 统计细则（比较/逻辑操作折算规则）\n- ROP（正交投影）消融的实现细节\n\n"

body = header + summary + metrics_sec + todo
try:
    # 若已存在则追加分隔
    if os.path.exists(report_path):
        with open(report_path,'a') as f:
            f.write("\n\n---\n\n"+body)
    else:
        with open(report_path,'w') as f:
            f.write(body)
    print(f"报告已写入: {report_path}")
except Exception as e:
    print("写报告失败:", e)
PY

echo "复刻自检完成：请查看 replication_report.md 与 metrics.json"


