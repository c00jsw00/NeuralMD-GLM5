# NeuralMD for PDB 4MXC
**Protein-Ligand Binding Dynamics Simulation**

---

## 📋 專案說明

基於 Nature Communications 2025 發表的 **NeuralMD** 研究，為 PDB ID: 4MXC 生成的完整訓練與推論代碼。

### 🎯 研究重點

1. **Multi-grained modeling** - 三層粒度建模
   - 原子層 (配體)
   - 骨架層 (蛋白質)
   - 殘基層 (結合複合體)

2. **SE(3)-equivariant** - 旋轉和平移等變
   - Vector frames 向量框架
   - e3nn 實現

3. **Differential Equation Model** - 微分方程模型
   - 二階 ODE (Newtonian dynamics)
   - 二階 SDE (Langevin dynamics)

### 📊 目標 PDB

**PDB ID: 4MXC**
- 蛋白質 - 配體結合複合體
- 用於訓練和推論測試

### 🔧 技術棧

- **Python 3.9+**
- **PyTorch 2.2**
- **PyTorch Geometric (PyG)**
- **e3nn** (SE(3) 等變網路)
- **torchdiffeq** (微分方程求解器)
- **MDAnalysis** (分子動力學分析)
- **Biopython** (PDB 解析)

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝基本依賴
pip install torch==2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝其他依賴
pip install pyg torch-scatter torch-sparse torch-cluster
pip install e3nn torchdiffeq MDAnalysis biopython numpy networkx scikit-learn
pip install matplotlib seaborn
```

### 2. 下載 PDB 文件

```bash
# 下載 4MXC 的 PDB 文件
wget https://files.rcsb.org/download/4MXC.pdb
```

### 3. 執行訓練

```bash
python train.py --pdb_file 4MXC.pdb --mode single_traj
```

### 4. 執行推論

```bash
python inference.py --pdb_file 4MXC.pdb --checkpoint model.pt
```

---

## 📁 專案結構

```
NeuralMD-4MXC/
├── data_loader.py           # 數據加載器
├── model_binding.py         # BindingNet 模型
├── dynamics_solver.py       # 微分方程求解器
├── train.py                 # 訓練腳本
├── inference.py             # 推論腳本
├── visualize.py             # 可視化工具
├── utils.py                 # 工具函數
├── config.py                # 配置參數
├── requirements.txt         # Python 依賴
├── README.md                # 專案說明
└── notebooks/
    └── analysis.ipynb       # 分析筆記本
```

---

## 📚 引用

如果這個代碼對您有幫助，請引用原始研究：

```bibtex
@article{liu2024NeuralMD,
  title={A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics},
  author={Liu, Shengchao* and Du, Weitao* and Xu, Hannan and Li, Yanjing and Li, Zhuoxinran and Bhethanabotla, Vignesh and Liang, Yan and Borgs, Christian* and Anandkumar, Anima* and Guo, Hongyu* and Chayes, Jennifer*},
  journal={Nature Communications},
  year={2025}
}
```

---

## ⚠️ 注意事項

1. **計算資源** - 需要 GPU (建議 >= 16GB VRAM)
2. **數據集** - 建議使用 MISATO 數據集進行訓練
3. **PDB 文件** - 確保 PDB 文件格式正確
4. **訓練時間** - 根據數據量可能需要數小時至數天

---

## 📞 聯絡

如有問題，請聯繫：
- Email: chao1224@gmail.com
- GitHub: https://github.com/chao1224/NeuralMD

---

**版本：** 1.0.0  
**生成日期：** 2026-03-20  
**AI 助手：** 品丸 (Pinwan)  
**模型：** GLM-5 via NVIDIA NIM
