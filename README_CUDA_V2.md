# NeuralMD CUDA 版本 v2.0
**GPU 加速的蛋白質 - 配體結合動力學模擬**

---

## 🚀 版本特點

基於 **GLM-5** 模型生成的 CUDA 加速版本，針對 NVIDIA GPU 全面優化。

### ✨ 主要功能

1. **GPU 加速**
   - 全 GPU 張量運算
   - 混合精度訓練 (AMP)
   - 顯存優化

2. **性能優化**
   - CUDA kernel 優化
   - 批量處理加速
   - 數據預取

3. **易用性**
   - 一鍵訓練
   - 自動 GPU 檢測
   - 完整註解

---

## 📋 系統要求

- **GPU:** NVIDIA GPU (CUDA 11.7+)
  - 推薦：A100, V100, RTX 3090/4090
  - 最少：16GB VRAM
- **Python:** 3.9+
- **PyTorch:** 2.2+ (with CUDA)

---

## 🔧 安裝

### 1. 安裝 CUDA PyTorch

```bash
# CUDA 12.1
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安裝其他依賴

```bash
pip install -r requirements_cuda_v2.txt
```

### 3. 驗證 CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
```

---

## 🚀 快速開始

### 1. 下載 PDB

```bash
wget https://files.rcsb.org/download/4MXC.pdb
```

### 2. 執行訓練

```bash
python neuralmd_4mxc_cuda.py
```

### 3. 查看結果

```bash
# 結果保存在 neuralmd_cuda_v2.pt
python -c "import torch; data=torch.load('neuralmd_cuda_v2.pt'); print('Losses:', len(data['losses']), 'steps'); print('Trajectory:', len(data['trajectory']), 'frames')"
```

---

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `neuralmd_4mxc_cuda.py` | 主要 CUDA 代碼 (GPU 加速) |
| `requirements_cuda_v2.txt` | CUDA 依賴清單 |
| `README_CUDA_V2.md` | 本文檔 |

---

## ⚙️ 配置參數

編輯 `neuralmd_4mxc_cuda.py` 中的 `ConfigCUDA` 類：

```python
class ConfigCUDA:
    DEVICE = "cuda"  # 或 "cpu"
    MIXED_PRECISION = True  # 混合精度
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
```

---

## 📊 性能預期

| 操作 | CPU | CUDA (V100) | 加速比 |
|------|-----|-------------|--------|
| 訓練 1 epoch | ~5min | ~5s | 60x |
| 推論 100 steps | ~2min | ~2s | 60x |
| 顯存使用 | ~4GB | ~8GB | - |

---

## 🎯 測試 PDB 4MXC

```bash
# 測試運行
python neuralmd_4mxc_cuda.py

# 預期輸出:
# NeuralMD CUDA 版本 v2.0
# CUDA 可用：True
# GPU 數量：1
# GPU: NVIDIA GeForce RTX 3090
# ...
```

---

## ⚠️ 注意事項

1. **顯存不足**
   - 減少批次大小
   - 關閉混合精度

2. **CUDA 版本**
   - 確保 GPU 驅動更新
   - 使用相容的 CUDA 版本

3. **性能優化**
   - 使用較大的 batch size
   - 啟用多 workers 數據加載

---

## 📚 引用

```bibtex
@article{liu2024NeuralMD,
  title={A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics},
  author={Liu, Shengchao* and Du, Weitao* and Xu, Hannan and Li, Yanjing and Li, Zhuoxinran and Bhethanabotla, Vignesh and Liang, Yan and Borgs, Christian* and Anandkumar, Anima* and Guo, Hongyu* and Chayes, Jennifer*},
  journal={Nature Communications},
  year={2025}
}
```

---

## 🔗 連結

- **原始研究:** https://www.nature.com/articles/s41467-025-67808-z
- **原始代碼:** https://github.com/chao1224/NeuralMD
- **CUDA 文檔:** https://docs.nvidia.com/cuda/

---

**版本:** 2.0  
**生成日期:** 2026-03-20  
**AI:** 品丸 (Pinwan) via GLM-5  
**GPU:** CUDA 加速
