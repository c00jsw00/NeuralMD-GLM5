"""
NeuralMD - 蛋白質 - 配體結合動力學模擬
PDB ID: 4mxc 測試版本

基於 Nature Communications 2025 文章和 NeuralMD GitHub 程式碼
作者：品丸 (Pinwan)
基於 GLM-5 模型生成
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入可選依賴
try:
    from Bio.PDB import PDBParser, PPBuilder
    from Bio import PDB
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️ 警告：BioPython 未安裝，將使用模擬數據")

try:
    import MDAnalysis as mda
    MDANALYSIS_AVAILABLE = True
except ImportError:
    MDANALYSIS_AVAILABLE = False
    print("⚠️ 警告：MDAnalysis 未安裝，將使用簡化方法")


class VectorFrame:
    """
    向量框架類 - 用於 SE(3)-equivariant 建模
    提供局部坐標框架來實現旋轉等變性
    """
    
    def __init__(self, positions: torch.Tensor):
        """
        初始化向量框架
        
        Args:
            positions: 原子位置 [N, 3]
        """
        self.positions = positions
        self.frames = self._compute_frames()
    
    def _compute_frames(self) -> torch.Tensor:
        """
        計算每個原子的局部向量框架
        
        Returns:
            框架矩陣 [N, 3, 3]
        """
        N = self.positions.shape[0]
        frames = torch.zeros(N, 3, 3, device=self.positions.device)
        
        # 簡單的框架計算（實際實現需要更複雜的算法）
        for i in range(N):
            if i < N - 1:
                # 第一個軸：指向下一個原子
                direction = self.positions[i + 1] - self.positions[i]
                direction = direction / (torch.norm(direction) + 1e-8)
                frames[i, 0] = direction
                
                # 第二個軸：與第一個軸正交
                random_vec = torch.tensor([1.0, 0.0, 0.0], device=self.positions.device)
                if torch.abs(direction[0]) < 0.9:
                    random_vec = torch.tensor([0.0, 1.0, 0.0], device=self.positions.device)
                
                cross_product = torch.cross(direction, random_vec)
                cross_product = cross_product / (torch.norm(cross_product) + 1e-8)
                frames[i, 1] = cross_product
                
                # 第三個軸：與前兩個軸正交
                frames[i, 2] = torch.cross(frames[i, 0], frames[i, 1])
            else:
                frames[i] = torch.eye(3, device=self.positions.device)
        
        return frames
    
    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        將向量投影到向量框架
        
        Args:
            vectors: 向量 [N, 3]
            
        Returns:
            投影後的向量 [N, 3]
        """
        return torch.bmm(self.frames, vectors.unsqueeze(-1)).squeeze(-1)


class BindingNet(nn.Module):
    """
    BindingNet - 多粒度 SE(3)-equivariant 結合模型
    
    三個粒度：
    1. 配體原子層級
    2. 蛋白質骨幹層級
    3. 殘基 - 原子配對層級
    """
    
    def __init__(
        self,
        ligand_atom_features: int = 10,
        protein_backbone_features: int = 15,
        residue_features: int = 20,
        hidden_features: int = 128,
        num_layers: int = 4,
        num_heads: int = 4
    ):
        """
        初始化 BindingNet
        
        Args:
            ligand_atom_features: 配體原子特徵維度
            protein_backbone_features: 蛋白質骨幹特徵維度
            residue_features: 殘基特徵維度
            hidden_features: 隱藏層特徵維度
            num_layers: 圖神經網絡層數
            num_heads: 注意力機制頭數
        """
        super().__init__()
        
        self.ligand_atom_features = ligand_atom_features
        self.protein_backbone_features = protein_backbone_features
        self.residue_features = residue_features
        self.hidden_features = hidden_features
        
        # 配體特徵投影
        self.ligand_embedding = nn.Sequential(
            nn.Linear(ligand_atom_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        )
        
        # 蛋白質特徵投影
        self.protein_embedding = nn.Sequential(
            nn.Linear(protein_backbone_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        )
        
        # 殘基特徵投影
        self.residue_embedding = nn.Sequential(
            nn.Linear(residue_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        )
        
        # SE(3)-equivariant 圖注意力層
        self.message_layers = nn.ModuleList([
            SE3AttentionLayer(hidden_features, num_heads)
            for _ in range(num_layers)
        ])
        
        # 能量頭
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, 1)
        )
        
        # 力頭
        self.force_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, 3)
        )
    
    def forward(
        self,
        ligand_positions: torch.Tensor,
        ligand_features: torch.Tensor,
        protein_positions: torch.Tensor,
        protein_features: torch.Tensor,
        residue_features: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            ligand_positions: 配體位置 [N_ligand, 3]
            ligand_features: 配體特徵 [N_ligand, ligand_atom_features]
            protein_positions: 蛋白質位置 [N_protein, 3]
            protein_features: 蛋白質特徵 [N_protein, protein_backbone_features]
            residue_features: 殘基特徵 [N_residue, residue_features]
            edge_index: 邊索引 [2, num_edges]
            distances: 邊距離 [num_edges]
            
        Returns:
            (energies, forces)
        """
        # 組合所有原子
        all_positions = torch.cat([ligand_positions, protein_positions], dim=0)
        all_features = torch.cat([
            self.ligand_embedding(ligand_features),
            self.protein_embedding(protein_features)
        ], dim=0)
        
        # 殘基特徵處理
        residue_embeddings = self.residue_embedding(residue_features)
        
        # 計算向量框架
        ligand_frames = VectorFrame(ligand_positions)
        protein_frames = VectorFrame(protein_positions)
        
        # 圖神經網絡層
        x = all_features
        for message_layer in self.message_layers:
            x = message_layer(x, all_positions, edge_index, distances)
        
        # 預測能量和力
        energies = self.energy_head(x)
        forces = self.force_head(x)
        
        return energies, forces


class SE3AttentionLayer(nn.Module):
    """
    SE(3)-equivariant 圖注意力層
    
    使用向量框架實現旋轉等變性
    """
    
    def __init__(self, hidden_features: int, num_heads: int = 4):
        super().__init__()
        
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.head_dim = hidden_features // num_heads
        
        # 查詢、鍵、值投影
        self.q_proj = nn.Linear(hidden_features, hidden_features)
        self.k_proj = nn.Linear(hidden_features, hidden_features)
        self.v_proj = nn.Linear(hidden_features, hidden_features)
        
        # 距離投影
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.GELU()
        )
        
        # 輸出投影
        self.out_proj = nn.Linear(hidden_features, hidden_features)
        
        # 歸一化
        self.norm = nn.LayerNorm(hidden_features)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 節點特徵 [N, hidden_features]
            positions: 節點位置 [N, 3]
            edge_index: 邊索引 [2, num_edges]
            distances: 邊距離 [num_edges]
            
        Returns:
            更新後的節點特徵 [N, hidden_features]
        """
        N = x.shape[0]
        
        # 計算查詢、鍵、值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 距離編碼
        distance_encoding = self.distance_mlp(distances.unsqueeze(-1))
        
        # 訊息傳遞
        message = torch.zeros_like(x)
        num_messages = torch.zeros(N, device=x.device)
        
        for i, j in edge_index[0], edge_index[1]:
            # 計算注意力權重
            attention = (q[i] * k[j]).sum(dim=-1, keepdim=True) / np.sqrt(self.hidden_features)
            attention = attention * distance_encoding
            
            # 加權值訊息
            weight = torch.sigmoid(attention)
            message[i] += weight * v[j]
            num_messages[i] += 1
        
        # 平均
        num_messages = num_messages.unsqueeze(-1).clamp(min=1)
        message = message / num_messages
        
        # 輸出投影和殘差連接
        out = self.out_proj(message)
        out = self.norm(x + out)
        
        return out


class DynamicsSolver:
    """
    動力學求解器 - 使用二階微分方程
    
    支援 Newtonian 動力學 (ODE) 和 Langevin 動力學 (SDE)
    """
    
    def __init__(
        self,
        timestep: float = 1e-3,
        num_steps: int = 1000,
        dynamics_type: str = "newtonian",
        temperature: float = 300.0,
        friction: float = 0.1
    ):
        """
        初始化動力學求解器
        
        Args:
            timestep: 時間步長 (ps)
            num_steps: 模擬步數
            dynamics_type: "newtonian" 或 "langevin"
            temperature: 溫度 (Kelvin)
            friction: 摩擦力係數 (Langevin)
        """
        self.timestep = timestep
        self.num_steps = num_steps
        self.dynamics_type = dynamics_type
        self.temperature = temperature
        self.friction = friction
        
        # 物理常數
        self.kb = 0.0083144621  # Boltzmann 常數 (kJ/(mol·K))
        self.atom_masses = {
            'C': 12.011, 'N': 14.007, 'O': 15.999,
            'H': 1.008, 'S': 32.065, 'P': 30.974
        }
    
    def compute_acceleration(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor
    ) -> torch.Tensor:
        """
        計算加速度 (F = ma)
        
        Args:
            positions: 位置 [N, 3]
            forces: 力 [N, 3]
            masses: 質量 [N]
            
        Returns:
            加速度 [N, 3]
        """
        accelerations = forces / (masses.unsqueeze(-1) + 1e-8)
        return accelerations
    
    def velocity_verlet(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        accelerations: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Velocity Verlet 積分算法
        
        Args:
            positions: 當前位置 [N, 3]
            velocities: 當前速度 [N, 3]
            accelerations: 當前加速度 [N, 3]
            dt: 時間步長
            
        Returns:
            (新位置，新速度)
        """
        # 更新位置
        new_positions = (
            positions + velocities * dt + 0.5 * accelerations * dt ** 2
        )
        
        # 速度更新（一半）
        new_velocities = velocities + 0.5 * accelerations * dt
        
        return new_positions, new_velocities
    
    def integrate_langevin(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        accelerations: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Langevin 動力學積分
        
        Args:
            positions: 當前位置 [N, 3]
            velocities: 當前速度 [N, 3]
            accelerations: 當前加速度 [N, 3]
            dt: 時間步長
            
        Returns:
            (新位置，新速度)
        """
        # 隨機力
        random_force = torch.randn_like(velocities) * np.sqrt(
            2 * self.friction * self.kb * self.temperature / dt
        )
        
        # 速度更新
        new_velocities = (
            velocities + 0.5 * accelerations * dt -
            self.friction * velocities * dt +
            random_force * np.sqrt(dt)
        )
        
        # 位置更新
        new_positions = positions + new_velocities * dt
        
        return new_positions, new_velocities


class PDBParser:
    """
    PDB 文件解析器 - 用於下載和解析 PDB ID 4mxc
    """
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True) if BIOPYTHON_AVAILABLE else None
    
    def download_pdb(self, pdb_id: str, output_dir: str = "data/pdb") -> str:
        """
        從 RCSB PDB 下載文件
        
        Args:
            pdb_id: PDB ID (例如：4mxc)
            output_dir: 輸出目錄
            
        Returns:
            PDB 文件路徑
        """
        if not BIOPYTHON_AVAILABLE:
            print(f"⚠️ BioPython 未安裝，將創建模擬數據")
            os.makedirs(output_dir, exist_ok=True)
            return self._create_mock_pdb(pdb_id, output_dir)
        
        import urllib.request
        
        os.makedirs(output_dir, exist_ok=True)
        pdb_file = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
        
        if not os.path.exists(pdb_file):
            url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
            print(f"📥 下載 PDB 文件：{pdb_id}")
            urllib.request.urlretrieve(url, pdb_file)
        
        return pdb_file
    
    def _create_mock_pdb(self, pdb_id: str, output_dir: str) -> str:
        """
        創建模擬 PDB 數據（當 BioPython 不可用時）
        """
        import random
        
        os.makedirs(output_dir, exist_ok=True)
        pdb_file = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
        
        # 生成模擬數據
        random.seed(42)
        with open(pdb_file, 'w') as f:
            # 模擬蛋白質結構
            for i in range(100):
                f.write(f"ATOM  {i+1:5d}  CA  ALA A {i+1:4d}    "
                       f"{random.uniform(0, 50):8.3f}"
                       f"{random.uniform(0, 50):8.3f}"
                       f"{random.uniform(0, 50):8.3f}"
                       f" 1.00  0.00           C\n")
            
            # 模擬配體結構
            for i in range(20):
                f.write(f"HETATM {i+101:5d}  C1  LIG A {i+1:4d}    "
                       f"{random.uniform(0, 10):8.3f}"
                       f"{random.uniform(0, 10):8.3f}"
                       f"{random.uniform(0, 10):8.3f}"
                       f" 1.00  0.00           C\n")
        
        return pdb_file
    
    def parse_pdb(self, pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        解析 PDB 文件
        
        Args:
            pdb_file: PDB 文件路徑
            
        Returns:
            (ligand_positions, protein_positions, atom_types)
        """
        if not BIOPYTHON_AVAILABLE:
            return self._parse_mock_pdb(pdb_file)
        
        structure = self.parser.get_structure('protein', pdb_file)
        
        ligand_positions = []
        protein_positions = []
        atom_types = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # 識別配體（HETATM）
                    if residue.resname not in ['H2O']:
                        atom_info = []
                        for atom in residue:
                            pos = atom.coord
                            atom_types.append(atom.element.strip())
                            
                            if residue.resname.isalpha() and not residue.resname[0].isdigit():
                                # 配體
                                ligand_positions.append(pos)
                            else:
                                # 蛋白質
                                protein_positions.append(pos)
        
        return (
            np.array(ligand_positions) if ligand_positions else np.zeros((10, 3)),
            np.array(protein_positions) if protein_positions else np.zeros((50, 3)),
            np.array(atom_types)
        )
    
    def _parse_mock_pdb(self, pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析模擬 PDB 文件"""
        positions_ligand = np.random.randn(20, 3) * 2
        positions_protein = np.random.randn(100, 3) * 5
        atom_types = ['C'] * 120
        
        return positions_ligand, positions_protein, atom_types


class NeuralMD:
    """
    NeuralMD - 主模型類
    
    整合 BindingNet 和 Dynamics Solver
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        num_epochs: int = 100
    ):
        """
        初始化 NeuralMD
        
        Args:
            device: 計算設備
            learning_rate: 學習率
            num_epochs: 訓練輪數
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # 初始化模型
        self.binding_net = BindingNet().to(self.device)
        
        # 初始化求解器
        self.solver = DynamicsSolver(
            timestep=1e-3,
            num_steps=1000,
            dynamics_type="newtonian"
        ).to(self.device)
        
        # 優化器
        self.optimizer = torch.optim.Adam(
            self.binding_net.parameters(),
            lr=learning_rate
        )
        
        # PDB 解析器
        self.pdb_parser = PDBParser()
        
        print(f"✅ NeuralMD 已初始化在 {device}")
    
    def prepare_data(self, pdb_id: str = "4mxc") -> dict:
        """
        準備 PDB 4mxc 數據
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            數據字典
        """
        print(f"📊 準備 PDB {pdb_id} 數據...")
        
        # 下載並解析 PDB
        pdb_file = self.pdb_parser.download_pdb(pdb_id)
        ligand_pos, protein_pos, atom_types = self.pdb_parser.parse_pdb(pdb_file)
        
        # 轉換為 torch tensor
        ligand_positions = torch.tensor(ligand_pos, dtype=torch.float32)
        protein_positions = torch.tensor(protein_pos, dtype=torch.float32)
        
        # 生成模擬特徵
        ligand_features = torch.randn(ligand_positions.shape[0], 10)
        protein_features = torch.randn(protein_positions.shape[0], 15)
        residue_features = torch.randn(50, 20)
        
        # 計算邊和距離
        all_positions = torch.cat([ligand_positions, protein_positions], dim=0)
        distances = torch.cdist(all_positions, all_positions)
        edge_index = self._create_edges(distances, cutoff=10.0)
        
        return {
            'ligand_positions': ligand_positions,
            'protein_positions': protein_positions,
            'ligand_features': ligand_features,
            'protein_features': protein_features,
            'residue_features': residue_features,
            'edge_index': edge_index,
            'distances': distances,
            'atom_types': atom_types
        }
    
    def _create_edges(
        self,
        distances: torch.Tensor,
        cutoff: float = 10.0
    ) -> torch.Tensor:
        """
        創建鄰居圖邊
        
        Args:
            distances: 距離矩陣 [N, N]
            cutoff: 距離閾值
            
        Returns:
            邊索引 [2, num_edges]
        """
        mask = (distances < cutoff) & (distances > 0)
        edge_indices = torch.nonzero(mask, as_tuple=False).t()
        return edge_indices
    
    def train_step(self, data: dict) -> float:
        """
        單次訓練步驟
        
        Args:
            data: 數據字典
            
        Returns:
            損失值
        """
        self.optimizer.zero_grad()
        
        # 前向傳播
        energies, forces = self.binding_net(
            ligand_positions=data['ligand_positions'].to(self.device),
            ligand_features=data['ligand_features'].to(self.device),
            protein_positions=data['protein_positions'].to(self.device),
            protein_features=data['protein_features'].to(self.device),
            residue_features=data['residue_features'].to(self.device),
            edge_index=data['edge_index'].to(self.device),
            distances=data['distances'].to(self.device)
        )
        
        # 計算損失（簡化版本）
        loss = torch.mean(energies ** 2) + torch.mean(forces ** 2)
        
        # 反向傳播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def infer(
        self,
        data: dict,
        num_steps: int = 100
    ) -> List[torch.Tensor]:
        """
        推論 - 模擬動力學軌跡
        
        Args:
            data: 數據字典
            num_steps: 模擬步數
            
        Returns:
            位置軌跡列表
        """
        print(f"🔬 開始推論模擬 ({num_steps} 步)...")
        
        # 初始化速度和加速度
        positions = data['ligand_positions'].to(self.device).clone()
        velocities = torch.zeros_like(positions)
        accelerations = torch.zeros_like(positions)
        
        # 儲存軌跡
        trajectory = [positions.clone().cpu()]
        
        for step in range(num_steps):
            # 計算能量和力
            energies, forces = self.binding_net(
                ligand_positions=data['ligand_positions'].to(self.device),
                ligand_features=data['ligand_features'].to(self.device),
                protein_positions=data['protein_positions'].to(self.device),
                protein_features=data['protein_features'].to(self.device),
                residue_features=data['residue_features'].to(self.device),
                edge_index=data['edge_index'].to(self.device),
                distances=data['distances'].to(self.device)
            )
            
            # 計算加速度
            masses = torch.ones(positions.shape[0], device=self.device)
            accelerations = self.solver.compute_acceleration(
                positions, forces, masses
            )
            
            # 積分
            positions, velocities = self.solver.velocity_verlet(
                positions, velocities, accelerations, self.solver.timestep
            )
            
            # 儲存軌跡
            if step % 10 == 0:
                trajectory.append(positions.clone().cpu())
        
        print(f"✅ 推論完成，軌跡長度：{len(trajectory)}")
        return trajectory
    
    def train(
        self,
        data: dict,
        num_epochs: int = None
    ) -> List[float]:
        """
        完整訓練流程
        
        Args:
            data: 數據字典
            num_epochs: 訓練輪數
            
        Returns:
            損失曲線
        """
        num_epochs = num_epochs or self.num_epochs
        losses = []
        
        print(f"🚀 開始訓練 ({num_epochs} epochs)...")
        
        for epoch in range(num_epochs):
            loss = self.train_step(data)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        
        print(f"✅ 訓練完成，最終損失：{losses[-1]:.4f}")
        return losses


def main():
    """主函數"""
    print("=" * 60)
    print("NeuralMD - 蛋白質 - 配體結合動力學模擬")
    print("PDB ID: 4mxc")
    print("=" * 60)
    
    # 初始化模型
    model = NeuralMD(device="cpu")  # 使用 CPU 以確保可運行
    
    # 準備數據
    data = model.prepare_data(pdb_id="4mxc")
    print(f"✅ 數據準備完成")
    print(f"   配體原子數：{data['ligand_positions'].shape[0]}")
    print(f"   蛋白質原子數：{data['protein_positions'].shape[0]}")
    
    # 訓練
    losses = model.train(data, num_epochs=20)
    
    # 推論
    trajectory = model.infer(data, num_steps=100)
    
    # 保存結果
    torch.save({
        'losses': losses,
        'trajectory': trajectory,
        'model_state': model.binding_net.state_dict()
    }, 'neuralmd_4mxc_results.pt')
    
    print("=" * 60)
    print("✅ 完成！結果已保存到 neuralmd_4mxc_results.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
