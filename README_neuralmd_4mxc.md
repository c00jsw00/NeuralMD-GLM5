# NeuralMD - PDB 4mxc 測試版本

基於 GLM-5 模型編寫的完整 NeuralMD 訓練與推論程式碼

---

## 📚 項目背景

### Nature 文章
- **標題：** A multi-grained symmetric differential equation model for learning protein-ligand binding dynamics
- **期刊：** Nature Communications 2025
- **鏈接：** https://www.nature.com/articles/s41467-025-67808-z

### 原始 GitHub
- **倉庫：** https://github.com/chao1224/NeuralMD
- **作者：** Shengchao Liu, Weitao Du, et al.

---

## 🎯 項目目標

使用 GLM-5 生成完整的 Python 程式碼，用於：
1. 下載並解析 PDB ID 4mxc
2. 實現 BindingNet（SE(3)-equivariant 模型）
3. 實現 Dynamics Solver（ODE/SDE）
4. 訓練和推論
5. 測試 PDB 4mxc 結構

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd /home/c00jsw00/workspace/NeuralMD
pip install -r requirements_neuralmd_4mxc.txt
```

### 2. 運行程式

```bash
python neuralmd_4mxc.py
```

### 3. 查看結果

```bash
ls -lh neuralmd_4mxc_results.pt
```

---

## 📋 程式碼結構

```
neuralmd_4mxc.py
├── VectorFrame           # 向量框架類（SE(3)-equivariant）
├── BindingNet            # 多粒度結合模型
│   ├── 配體原子層級
│   ├── 蛋白質骨幹層級
│   └── 殘基 - 原子配對層級
├── SE3AttentionLayer     # SE(3)-equivariant 圖注意力層
├── DynamicsSolver        # 動力學求解器
│   ├── Newtonian (ODE)
│   └── Langevin (SDE)
├── PDBParser             # PDB 文件解析器
└── NeuralMD              # 主模型類
    ├── 數據準備
    ├── 訓練
    └── 推論
```

---

## 🔬 核心功能

### 1. 多粒度建模

NeuralMD 使用三個粒度的向量框架：

1. **配體原子層級** - 小分子配體的原子級表示
2. **蛋白質骨幹層級** - 蛋白質的 N-Cα-C 骨架
3. **殘基 - 原子配對層級** - 蛋白質 - 配體相互作用的殘基級表示

### 2. SE(3)-Equivariant

使用向量框架實現旋轉和平移等變性：

```python
# 投影到局部框架
projected_vectors = frames.project(vectors)
# 在框架內進行計算，保持等變性
```

### 3. 動力學模擬

使用 Velocity Verlet 積分算法：

```python
# 更新位置
new_pos = pos + vel*dt + 0.5*acc*dt^2
# 更新速度
new_vel = vel + 0.5*acc*dt
```

---

## 📊 數據集

### MISATO 數據集
- **來源：** Protein Data Bank (PDB)
- **規模：** 16,972 個蛋白質 - 配體複合物
- **軌跡：** 每個複合物 100 個 snapshot，8 納秒

### PDB 4mxc
- **類型：** 蛋白質 - 配體複合物
- **用途：** 測試目標
- **下載：** 自動從 RCSB PDB 下載

---

## 🎓 使用示例

### 基本使用

```python
from neuralmd_4mxc import NeuralMD

# 初始化模型
model = NeuralMD(device="cpu")

# 準備數據
data = model.prepare_data(pdb_id="4mxc")

# 訓練
losses = model.train(data, num_epochs=100)

# 推論
trajectory = model.infer(data, num_steps=1000)
```

### 自定義設置

```python
# 使用 Langevin 動力學
solver = DynamicsSolver(
    timestep=1e-3,
    num_steps=1000,
    dynamics_type="langevin",
    temperature=300.0,
    friction=0.1
)

# 使用 GPU
model = NeuralMD(device="cuda")
```

---

## 📈 預期結果

### 訓練損失
- 初始損失：~1.0
- 最終損失：~0.1-0.3
- 收斂時間：~50-100 epochs

### 推論軌跡
- 步數：100-1000
- 時間跨度：0.1-1.0 ns
- 輸出：位置序列 [N_steps, N_atoms, 3]

---

## 🔧 進階設置

### 修改模型超參數

```python
binding_net = BindingNet(
    ligand_atom_features=10,
    protein_backbone_features=15,
    residue_features=20,
    hidden_features=256,      # 增加隱藏層
    num_layers=6,            # 增加層數
    num_heads=8              # 增加注意力頭數
)
```

### 調整動力學參數

```python
solver = DynamicsSolver(
    timestep=0.5e-3,         # 更小的時間步
    num_steps=5000,          # 更多步數
    dynamics_type="langevin",
    temperature=310.0,       # 生理溫度
    friction=0.05            # 降低摩擦
)
```

---

## 📝 注意事項

### 1. 依賴安裝
- **必需：** torch, torch-geometric, numpy
- **可選：** biopython, MDAnalysis
- 如果 BioPython 未安裝，將使用模擬數據

### 2. GPU 加速
- 建議使用 GPU 進行訓練
- 設置 `device="cuda"` 啟用 GPU

### 3. 數據下載
- PDB 文件會緩存在 `data/pdb/` 目錄
- 首次運行會自動下載

---

## 🎯 測試目標：PDB 4mxc

### 結構信息
- **蛋白質：** 目標蛋白質
- **配體：** 結合小分子
- **類型：** 半剛性設定（蛋白質剛性，配體柔性）

### 模擬設置
- **時間步長：** 1 fs (1e-3 ps)
- **模擬長度：** 1-10 ns
- **溫度：** 300 K
- **系綜：** NVT (Langevin) 或 NVE (Newtonian)

---

## 📚 引用

如果這個工作對您有幫助，請引用：

```bibtex
@article{liu2024NeuralMD,
  title={A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics},
  author={Liu, Shengchao* and Du, Weitao* and Xu, Hannan and Li, Yanjing and Li, Zhuoxinran and Bhethanabotla, Vignesh and Liang, Yan and Borgs, Christian* and Anandkumar, Anima* and Guo, Hongyu* and Chayes, Jennifer*},
  journal={Nature Communications},
  year={2025}
}
```

---

## 🔗 相關資源

- **NeuralMD 原始代碼：** https://github.com/chao1224/NeuralMD
- **MISATO 數據集：** https://huggingface.co/datasets/chao1224/NeuralMD
- **PDB 數據庫：** https://www.rcsb.org/
- **GLM-5：** NVIDIA NIM Platform

---

## 📝 版本歷史

- **v1.0 (2026-03-20)** - 初始版本，基於 GLM-5 生成
  - 完整的 BindingNet 實現
  - Newtonian 和 Langevin 動力學
  - PDB 4mxc 測試支持
  - 可選的 BioPython 解析

---

## ⚖️ 授權

本代碼供研究和學習使用。

---

**編寫者：** 品丸 (Pinwan)  
**生成模型：** GLM-5 (NVIDIA NIM)  
**生成日期：** 2026-03-20
