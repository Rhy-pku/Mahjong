#!/bin/bash
# 完整预训练流程脚本
# 使用data.txt中的1000万行专家对局数据进行监督学习预训练

set -e  # 遇到错误立即退出

# ============================================================
# 配置参数
# ============================================================

# 数据路径
DATA_FILE="data.txt"
DATA_DIR="./pretrain_data_full"
VAL_DATA_DIR=""  # 可选：验证集路径

# 预训练参数
OUTPUT_DIR="./pretrain_checkpoints"
MODEL_TYPE="full"  # full 或 light
HIDDEN_DIM=256
USE_VEC="--use_vec"  # 使用向量特征

# 训练超参数
BATCH_SIZE=128
EPOCHS=20
LR=0.001
VALUE_COEFF=1.0
WEIGHT_DECAY=0.0001
NUM_WORKERS=4

# Wandb配置
WANDB_PROJECT="mahjong-pretrain"
WANDB_NAME="full_pretrain_$(date +%Y%m%d_%H%M%S)"

# GPU设置
DEVICE="cuda"  # 或 "cpu"

# 其他
SAVE_INTERVAL=2  # 每2个epoch保存一次
GAMMA=0.99  # 折扣因子（数据处理用）
SAMPLES_PER_FILE=50000  # 每个数据文件的样本数

# ============================================================
# 步骤1: 数据预处理
# ============================================================

echo "============================================================"
echo "步骤 1/2: 预处理专家对局数据"
echo "============================================================"
echo "输入文件: $DATA_FILE"
echo "输出目录: $DATA_DIR"
echo ""

if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件 $DATA_FILE 不存在！"
    exit 1
fi

# 统计数据规模
TOTAL_LINES=$(wc -l < "$DATA_FILE")
TOTAL_MATCHES=$(grep -c "^Match" "$DATA_FILE" || true)
echo "数据规模:"
echo "  总行数: $TOTAL_LINES"
echo "  总局数: $TOTAL_MATCHES"
echo ""

# 运行数据预处理
if [ -d "$DATA_DIR" ]; then
    echo "警告: 输出目录 $DATA_DIR 已存在"
    read -p "是否删除并重新处理? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DATA_DIR"
        echo "已删除旧数据"
    else
        echo "跳过数据预处理，使用现有数据"
        SKIP_PREPROCESSING=1
    fi
fi

if [ -z "$SKIP_PREPROCESSING" ]; then
    echo "开始数据预处理..."
    conda activate KNP
    python prepare_full_data.py \
        --input "$DATA_FILE" \
        --output "$DATA_DIR" \
        --gamma $GAMMA \
        --samples_per_file $SAMPLES_PER_FILE

    echo ""
    echo "✓ 数据预处理完成！"
    echo "  输出目录: $DATA_DIR"
    ls -lh "$DATA_DIR" | head -10
    echo ""
fi

# ============================================================
# 步骤2: 预训练
# ============================================================
