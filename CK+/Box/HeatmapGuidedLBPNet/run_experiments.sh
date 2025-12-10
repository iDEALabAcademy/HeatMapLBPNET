#!/bin/bash

# Bash script to run 5 training experiments with different configurations
# Each run modifies the PRESETS dictionary in the training script

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_original_model_cropped.py"
BACKUP_SCRIPT="${SCRIPT_DIR}/train_original_model_cropped.py.backup"
RESULTS_SUMMARY="${SCRIPT_DIR}/experiments_summary.txt"
RESULTS_CSV="${SCRIPT_DIR}/experiments_summary.csv"

echo "=========================================="
echo "Starting 3 sequential training experiments"
echo "=========================================="
echo ""

# Create backup of original script
cp "${TRAIN_SCRIPT}" "${BACKUP_SCRIPT}"
echo "✅ Backup created: ${BACKUP_SCRIPT}"
echo ""

# Initialize results files
echo "Experiment Summary - $(date)" > "${RESULTS_SUMMARY}"
echo "===========================================" >> "${RESULTS_SUMMARY}"
echo "" >> "${RESULTS_SUMMARY}"

echo "experiment,output_dir,test_acc_no_bn,test_loss_no_bn,test_acc_with_bn,test_loss_with_bn,channels,p_schedule,num_patterns" > "${RESULTS_CSV}"

# Function to restore original script
restore_script() {
    cp "${BACKUP_SCRIPT}" "${TRAIN_SCRIPT}"
    echo "✅ Restored original script"
}

# Trap to restore script on exit
trap restore_script EXIT

# ============================================================
# Experiment 1: Current code, no change
# ============================================================
echo "=========================================="
echo "EXPERIMENT 1/3: Current code (no change)"
echo "Output: outputs_svhn_cropped_hm_P_decay"
echo "=========================================="
echo ""

# No modifications needed - run as is
python3 "${TRAIN_SCRIPT}" 2>&1 | tee /tmp/exp1_output.txt

# Extract results
TEST_ACC_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp1_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp1_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")
TEST_ACC_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp1_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp1_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")

# Save to summary files
echo "Experiment 1: Baseline (no change)" >> "${RESULTS_SUMMARY}"
echo "  Output: outputs_svhn_cropped_hm_P_decay" >> "${RESULTS_SUMMARY}"
echo "  Channels: [37, 40, 80, 80, 160, 160, 320, 320]" >> "${RESULTS_SUMMARY}"
echo "  P Schedule: [12, 8, 6, 6, 4, 4, 4, 4]" >> "${RESULTS_SUMMARY}"
echo "  Patterns: 2" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (no BN recal): ${TEST_ACC_NO_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (no BN recal): ${TEST_LOSS_NO_BN}" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (with BN recal): ${TEST_ACC_WITH_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (with BN recal): ${TEST_LOSS_WITH_BN}" >> "${RESULTS_SUMMARY}"
echo "" >> "${RESULTS_SUMMARY}"

echo "1,outputs_svhn_cropped_hm_P_decay,${TEST_ACC_NO_BN},${TEST_LOSS_NO_BN},${TEST_ACC_WITH_BN},${TEST_LOSS_WITH_BN},\"[37,40,80,80,160,160,320,320]\",\"[12,8,6,6,4,4,4,4]\",2" >> "${RESULTS_CSV}"

echo ""
echo "✅ Experiment 1 completed!"
echo ""
sleep 2

# ============================================================
# Experiment 2: Less channels (37,40,80,80,100,100,120,120)
# ============================================================
echo "=========================================="
echo "EXPERIMENT 2/3: Less channels"
echo "Channels: [37, 40, 80, 80, 100, 100, 120, 120]"
echo "Output: outputs_svhn_cropped_hm_P_decay_less_channel"
echo "=========================================="
echo ""

# Restore original and modify
restore_script

# Modify channels_per_stage in PRESETS
python3 << 'EOF'
import re

with open('/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/train_original_model_cropped.py', 'r') as f:
    content = f.read()

# Change channels_per_stage
content = re.sub(
    r'"channels_per_stage": \[37, 40, 80, 80, 160, 160, 320, 320\]',
    r'"channels_per_stage": [37, 40, 80, 80, 100, 100, 120, 120]',
    content
)

# Change output_dir
content = re.sub(
    r'output_dir = "\./outputs_svhn_cropped_hm_P_decay"',
    r'output_dir = "./outputs_svhn_cropped_hm_P_decay_less_channel"',
    content
)

# Also change the per_image_json_dir
content = re.sub(
    r'per_image_json_dir = "\./outputs_svhn_cropped_hm_P_decay"',
    r'per_image_json_dir = "./outputs_svhn_cropped_hm_P_decay_less_channel"',
    content
)

# Change ckpt_default path
content = re.sub(
    r"ckpt_default = os\.path\.join\('\./outputs_svhn_cropped_hm_P_decay', 'best_model\.pth'\)",
    r"ckpt_default = os.path.join('./outputs_svhn_cropped_hm_P_decay_less_channel', 'best_model.pth')",
    content
)

with open('/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/train_original_model_cropped.py', 'w') as f:
    f.write(content)
EOF

python3 "${TRAIN_SCRIPT}" 2>&1 | tee /tmp/exp2_output.txt

# Extract results
TEST_ACC_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp2_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp2_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")
TEST_ACC_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp2_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp2_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")

# Save to summary files
echo "Experiment 2: Less channels (P decay)" >> "${RESULTS_SUMMARY}"
echo "  Output: outputs_svhn_cropped_hm_P_decay_less_channel" >> "${RESULTS_SUMMARY}"
echo "  Channels: [37, 40, 80, 80, 100, 100, 120, 120]" >> "${RESULTS_SUMMARY}"
echo "  P Schedule: [12, 8, 6, 6, 4, 4, 4, 4]" >> "${RESULTS_SUMMARY}"
echo "  Patterns: 2" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (no BN recal): ${TEST_ACC_NO_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (no BN recal): ${TEST_LOSS_NO_BN}" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (with BN recal): ${TEST_ACC_WITH_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (with BN recal): ${TEST_LOSS_WITH_BN}" >> "${RESULTS_SUMMARY}"
echo "" >> "${RESULTS_SUMMARY}"

echo "2,outputs_svhn_cropped_hm_P_decay_less_channel,${TEST_ACC_NO_BN},${TEST_LOSS_NO_BN},${TEST_ACC_WITH_BN},${TEST_LOSS_WITH_BN},\"[37,40,80,80,100,100,120,120]\",\"[12,8,6,6,4,4,4,4]\",2" >> "${RESULTS_CSV}"

echo ""
echo "✅ Experiment 2 completed!"
echo ""
sleep 2

# ============================================================
# Experiment 3: Less channels + P=8 for layers 2-8
# ============================================================
echo "=========================================="
echo "EXPERIMENT 3/3: Less channels + P=8 (layers 2-8)"
echo "Channels: [37, 40, 80, 80, 100, 100, 120, 120]"
echo "P schedule: [12, 8, 8, 8, 8, 8, 8, 8]"
echo "Output: outputs_svhn_cropped_hm_P_less_channel"
echo "=========================================="
echo ""

# Restore original and modify
restore_script

python3 << 'EOF'
import re

with open('/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/train_original_model_cropped.py', 'r') as f:
    content = f.read()

# Change channels_per_stage
content = re.sub(
    r'"channels_per_stage": \[37, 40, 80, 80, 160, 160, 320, 320\]',
    r'"channels_per_stage": [37, 40, 80, 80, 100, 100, 120, 120]',
    content
)

# Change num_points_per_stage to use P=8 for all layers except first
content = re.sub(
    r'"num_points_per_stage": \[12, 8, 6, 6, 4, 4, 4, 4\]',
    r'"num_points_per_stage": [12, 8, 8, 8, 8, 8, 8, 8]',
    content
)

# Change output_dir
content = re.sub(
    r'output_dir = "\./outputs_svhn_cropped_hm_P_decay"',
    r'output_dir = "./outputs_svhn_cropped_hm_P_less_channel"',
    content
)

# Also change the per_image_json_dir
content = re.sub(
    r'per_image_json_dir = "\./outputs_svhn_cropped_hm_P_decay"',
    r'per_image_json_dir = "./outputs_svhn_cropped_hm_P_less_channel"',
    content
)

# Change ckpt_default path
content = re.sub(
    r"ckpt_default = os\.path\.join\('\./outputs_svhn_cropped_hm_P_decay', 'best_model\.pth'\)",
    r"ckpt_default = os.path.join('./outputs_svhn_cropped_hm_P_less_channel', 'best_model.pth')",
    content
)

with open('/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/train_original_model_cropped.py', 'w') as f:
    f.write(content)
EOF

python3 "${TRAIN_SCRIPT}" 2>&1 | tee /tmp/exp3_output.txt

# Extract results
TEST_ACC_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp3_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_NO_BN=$(grep "\[FINAL\] (no BN recal)" /tmp/exp3_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")
TEST_ACC_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp3_output.txt | grep -oP 'test_acc=\K[0-9.]+' || echo "N/A")
TEST_LOSS_WITH_BN=$(grep "\[FINAL\] (with BN recal)" /tmp/exp3_output.txt | grep -oP 'test_loss=\K[0-9.]+' || echo "N/A")

# Save to summary files
echo "Experiment 3: Less channels + P=8 (all layers)" >> "${RESULTS_SUMMARY}"
echo "  Output: outputs_svhn_cropped_hm_P_less_channel" >> "${RESULTS_SUMMARY}"
echo "  Channels: [37, 40, 80, 80, 100, 100, 120, 120]" >> "${RESULTS_SUMMARY}"
echo "  P Schedule: [12, 8, 8, 8, 8, 8, 8, 8]" >> "${RESULTS_SUMMARY}"
echo "  Patterns: 2" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (no BN recal): ${TEST_ACC_NO_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (no BN recal): ${TEST_LOSS_NO_BN}" >> "${RESULTS_SUMMARY}"
echo "  Test Acc (with BN recal): ${TEST_ACC_WITH_BN}%" >> "${RESULTS_SUMMARY}"
echo "  Test Loss (with BN recal): ${TEST_LOSS_WITH_BN}" >> "${RESULTS_SUMMARY}"
echo "" >> "${RESULTS_SUMMARY}"

echo "3,outputs_svhn_cropped_hm_P_less_channel,${TEST_ACC_NO_BN},${TEST_LOSS_NO_BN},${TEST_ACC_WITH_BN},${TEST_LOSS_WITH_BN},\"[37,40,80,80,100,100,120,120]\",\"[12,8,8,8,8,8,8,8]\",2" >> "${RESULTS_CSV}"

echo ""
echo "✅ Experiment 3 completed!"
echo ""

# Restore original script (will also happen via trap)
restore_script

echo ""
echo "=========================================="
echo "ALL 3 EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  1. outputs_svhn_cropped_hm_P_decay"
echo "  2. outputs_svhn_cropped_hm_P_decay_less_channel"
echo "  3. outputs_svhn_cropped_hm_P_less_channel"
echo ""
echo "📊 Summary files:"
echo "  - ${RESULTS_SUMMARY}"
echo "  - ${RESULTS_CSV}"
echo ""
echo "📋 Results Summary:"
cat "${RESULTS_SUMMARY}"
echo ""
echo "📊 CSV Summary:"
cat "${RESULTS_CSV}"
echo ""
