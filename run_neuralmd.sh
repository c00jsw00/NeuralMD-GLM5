#!/bin/bash
# NeuralMD 訓練與推論腳本
# PDB ID: 4MXC

set -e  # 遇到錯誤立即退出

echo "=========================================="
echo "NeuralMD - 蛋白質 - 配體結合動力學模擬"
echo "PDB ID: 4MXC"
echo "=========================================="

# 檢查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本：$python_version"

# 檢查依賴
echo ""
echo "檢查依賴..."
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ PyTorch 未安裝"
python3 -c "import torch_geometric; print('✓ PyG:', torch_geometric.__version__)" || echo "✗ PyG 未安裝"
python3 -c "import e3nn; print('✓ e3nn:', e3nn.__version__)" || echo "✗ e3nn 未安裝"
python3 -c "import MDAnalysis; print('✓ MDAnalysis:', MDAnalysis.__version__)" || echo "✗ MDAnalysis 未安裝"
python3 -c "from Bio import PDB; print('✓ Biopython: 已安裝')" || echo "✗ Biopython 未安裝"

# 下載 PDB 文件
PDB_FILE="4MXC.pdb"
if [ ! -f "$PDB_FILE" ]; then
    echo ""
    echo "下載 PDB 文件：$PDB_FILE"
    wget https://files.rcsb.org/download/$PDB_FILE -O $PDB_FILE
    echo "✓ PDB 文件下載完成"
else
    echo "PDB 文件已存在：$PDB_FILE"
fi

# 訓練模式
MODE=${1:-"train"}

if [ "$MODE" == "train" ]; then
    echo ""
    echo "=========================================="
    echo "訓練模式"
    echo "=========================================="
    
    python3 train.py --pdb_file $PDB_FILE --epochs 50
    
elif [ "$MODE" == "infer" ]; then
    echo ""
    echo "=========================================="
    echo "推論模式"
    echo "=========================================="
    
    python3 inference.py --pdb_file $PDB_FILE --checkpoint neuralmd_checkpoint.pt
    
elif [ "$MODE" == "full" ]; then
    echo ""
    echo "=========================================="
    echo "完整流程：訓練 + 推論"
    echo "=========================================="
    
    python3 train.py --pdb_file $PDB_FILE --epochs 50
    python3 inference.py --pdb_file $PDB_FILE --checkpoint neuralmd_checkpoint.pt
    
else
    echo "用法：$0 {train|infer|full}"
    echo ""
    echo "選項:"
    echo "  train  - 僅訓練模型"
    echo "  infer  - 僅執行推論 (需要預訓練模型)"
    echo "  full   - 完整流程：訓練 + 推論"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
