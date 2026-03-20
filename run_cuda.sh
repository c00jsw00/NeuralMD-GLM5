#!/bin/bash
# NeuralMD CUDA 版本執行腳本
# GPU 加速的蛋白質 - 配體結合動力學模擬

set -e

echo "=========================================="
echo "NeuralMD CUDA 版本"
echo "GPU Accelerated Protein-Ligand Binding"
echo "=========================================="

# 檢查 CUDA
echo ""
echo "檢查 CUDA 環境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU 檢測到"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ NVIDIA GPU 未檢測到，將使用 CPU"
fi

python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# 下載 PDB 文件
PDB_FILE="4MXC.pdb"
if [ ! -f "$PDB_FILE" ]; then
    echo ""
    echo "下載 PDB 文件：$PDB_FILE"
    wget https://files.rcsb.org/download/$PDB_FILE -O $PDB_FILE
fi

# 模式選擇
MODE=${1:-"train"}

case $MODE in
    train)
        echo ""
        echo "=========================================="
        echo "訓練模式 (CUDA)"
        echo "=========================================="
        python3 train_cuda.py --pdb_file $PDB_FILE --epochs 100
        ;;
    infer)
        echo ""
        echo "=========================================="
        echo "推論模式 (CUDA)"
        echo "=========================================="
        python3 inference_cuda.py --pdb_file $PDB_FILE --checkpoint neuralmd_cuda.pt
        ;;
    benchmark)
        echo ""
        echo "=========================================="
        echo "性能 Benchmark"
        echo "=========================================="
        python3 benchmark_cuda.py --compare
        ;;
    full)
        echo ""
        echo "=========================================="
        echo "完整流程：訓練 + 推論 + Benchmark"
        echo "=========================================="
        python3 train_cuda.py --pdb_file $PDB_FILE --epochs 100
        python3 inference_cuda.py --pdb_file $PDB_FILE --checkpoint neuralmd_cuda.pt
        python3 benchmark_cuda.py --test
        ;;
    *)
        echo "用法：$0 {train|infer|benchmark|full}"
        echo ""
        echo "選項:"
        echo "  train  - GPU 訓練"
        echo "  infer  - GPU 推論"
        echo "  benchmark - 性能測試"
        echo "  full   - 完整流程"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
