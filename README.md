# NeuralMD

**A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics**

📄 **Nature Communications 2025** | 📑 **ArXiv:2401.15122**

---

## Authors

- Shengchao Liu*
- Weitao Du*
- Hannan Xu
- Yanjing Li
- Zhuoxinran Li
- Vignesh Bhethanabotla
- Divin Yan
- Christian Borgs*
- Anima Anandkumar*
- Hongyu Guo*
- Jennifer Chayes*

*Equal contribution

---

## Links

- [Project Page](https://chao1224.github.io/NeuralMD)
- [ArXiv](https://arxiv.org/abs/2401.15122)
- [Datasets on HuggingFace](https://huggingface.co/datasets/chao1224/NeuralMD/tree/main)
- [Checkpoints on HuggingFace](https://huggingface.co/chao1224/NeuralMD/tree/main)
- [Paper (Nature Communications)](https://www.nature.com/articles/s41467-025-67808-z)

---

## Overview

NeuralMD is a deep learning framework for protein-ligand binding simulation using multi-grained symmetric differential equation models.

---

## Setup

### 1. Install Anaconda

```bash
wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh -b
export PATH=$PWD/anaconda3/bin:$PATH
```

### 2. Create Conda Environment

```bash
conda create -n Geom3D python=3.9
conda activate Geom3D
```

### 3. Install Basic Packages

```bash
conda install -y numpy networkx scikit-learn
conda install -y -c conda-forge rdkit
conda install -y pytorch==2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c pyg -c conda-forge pyg=2.5
conda install -y -c pyg pytorch-scatter
conda install -y -c pyg pytorch-sparse
conda install -y -c pyg pytorch-cluster
```

### 4. Install Python Packages

```bash
pip install ogb==1.2.1
pip install sympy
pip install ase
pip install lie_learn  # for TFN and SE3-Trans
pip install packaging  # for SEGNN
pip3 install e3nn  # for SEGNN
pip install transformers  # for smiles
pip install selfies  # for selfies
pip install atom3d  # for Atom3D
pip install cffi  # for Atom3D
pip install biopython  # for Atom3D
pip install cython  # for pyximport
conda install -y -c conda-forge py-xgboost-cpu  # for XGB
pip install pymatgen  # for CIF loading
pip install h5py
pip install torch-ema
pip install MDAnalysis
```

### 5. Install torchdiffeq

```bash
git clone git@github.com:chao1224/torchdiffeq.git
cd torchdiffeq
pip install .
```

### 6. Install This Package

```bash
pip install -e .
```

---

## Dataset Setup

We provide two ways to generate the datasets for MISATO:

### Option 1: Download from HuggingFace

The data folder structure looks like this:

```
.
├── MISATO_1000/
│   └── raw/
│       ├── train_MD.txt
│       ├── test_MD.txt
│       ├── MD.hdf5
│       └── val_MD.txt
├── MISATO/
│   └── raw/
│       ├── train_MD.txt
│       ├── test_MD.txt
│       ├── MD.hdf5
│       └── val_MD.txt
├── README.md
└── MISATO_100/
    └── raw/
        ├── train_MD.txt
        ├── test_MD.txt
        ├── MD.hdf5
        └── val_MD.txt
```

### Option 2: Download from Zenodo

```bash
wget -O data/MD/h5_files/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
```

For more details, check `data/README.md`.

---

## Usage

### Task Types

We have two types of tasks:

- `multi_traj`
- `single_traj`

### ML Methods

We provide four ML methods:

1. **VerletMD**
2. **GNNMD**
3. **DenoisinLD**
4. **NeuralMD**

### NeuralMD Binding Models

- `--NeuralMD_binding_model=NeuralMD_Binding01` for NeuralMD ODE
- `--NeuralMD_binding_model=NeuralMD_Binding02` or `--NeuralMD_binding_model=NeuralMD_Binding04` for NeuralMD SDE

### Checkpoints

We provide the optimal checkpoints and corresponding hyperparameters at [HuggingFace](https://huggingface.co/chao1224/NeuralMD/tree/main).

### Examples

Please check `examples/` for semi-flexible binding experiments.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2024NeuralMD,
  title={A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics},
  author={Liu, Shengchao* and Du, Weitao* and Xu, Hannan and Li, Yanjing and Li, Zhuoxinran and Bhethanabotla, Vignesh and Liang, Yan and Borgs, Christian* and Anandkumar, Anima* and Guo, Hongyu* and Chayes, Jennifer*},
  journal={arXiv preprint arXiv:2401.15122},
  year={2024}
}
```

---

## License

This project is for research purposes.

---

## Acknowledgments

This work was supported by various research institutions and funding agencies.
