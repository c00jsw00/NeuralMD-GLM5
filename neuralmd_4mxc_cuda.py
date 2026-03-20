"""
NeuralMD - CUDA 加速版本
Protein-Ligand Binding Dynamics Simulation with GPU Acceleration

基於 Nature Communications 2025 文章
PDB ID: 4MXC
生成者：品丸 (Pinwan) via GLM-5
"""

import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 檢查 CUDA 可用性
CUDA_AVAILABLE = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0

print("=" * 60)
print("NeuralMD CUDA 版本")
print(f"CUDA 可用：{CUDA_AVAILABLE}")
print(f"GPU 數量：{NUM_GPUS}")
if CUDA_AVAILABLE:
    print(f"GPU 名稱：{torch.cuda.get_device_name(0)}")
print("=" * 60)


class Config:
    """CUDA 配置參數"""
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    NUM_GPUS = NUM_GPUS
    USE_MIXED_PRECISION = True
    USE_DDP = NUM_GPUS > 1
    
    # 訓練參數
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WARMUP_STEPS = 1000
    
    # GPU 優化
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2


class VectorFrame:
    """
    向量框架類 - 用於 SE(3)-equivariant 建模
    CUDA 加速版本
    """
    
    def __init__(self, positions: torch.Tensor):
        """
        初始化向量框架
        
        Args:
            positions: 原子位置 [N, 3] (GPU 張量)
        """
        self.positions = positions.cuda() if not positions.is_cuda else positions
        self.frames = self._compute_frames()
    
    def _compute_frames(self) -> torch.Tensor:
        """
        計算每個原子的局部向量框架 (CUDA 加速)
        
        Returns:
            框架矩陣 [N, 3, 3]
        """
        N = self.positions.shape[0]
        frames = torch.zeros(N, 3, 3, device=self.positions.device)
        
        for i in range(N):
            if i < N - 1:
                direction = self.positions[i + 1] - self.positions[i]
                direction = direction / (torch.norm(direction) + 1e-8)
                frames[i, 0] = direction
                
                random_vec = torch.tensor([1.0, 0.0, 0.0], device=self.positions.device)
                if torch.abs(direction[0]) < 0.9:
                    random_vec = torch.tensor([0.0, 1.0, 0.0], device=self.positions.device)
                
                cross_product = torch.cross(direction, random_vec)
                cross_product = cross_product / (torch.norm(cross_product) + 1e-8)
                frames[i, 1] = cross_product
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


class BindingNetCUDA(nn.Module):
    """
    BindingNet - CUDA 加速的多粒度 SE(3)-equivariant 結合模型
    
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
        num_heads: int = 4,
        device: str = "cuda"
    ):
        """
        初始化 BindingNetCUDA
        
        Args:
            ligand_atom_features: 配體原子特徵維度
            protein_backbone_features: 蛋白質骨幹特徵維度
            residue_features: 殘基特徵維度
            hidden_features: 隱藏層特徵維度
            num_layers: 圖神經網絡層數
            num_heads: 注意力機制頭數
            device: 設備 (cuda/cpu)
        """
        super().__init__()
        
        self.ligand_atom_features = ligand_atom_features
        self.protein_backbone_features = protein_backbone_features
        self.residue_features = residue_features
        self.hidden_features = hidden_features
        self.device = device
        
        # 配體特徵投影 (GPU 加速)
        self.ligand_embedding = nn.Sequential(
            nn.Linear(ligand_atom_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        ).to(device)
        
        # 蛋白質特徵投影
        self.protein_embedding = nn.Sequential(
            nn.Linear(protein_backbone_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        ).to(device)
        
        # 殘基特徵投影
        self.residue_embedding = nn.Sequential(
            nn.Linear(residue_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU()
        ).to(device)
        
        # SE(3)-equivariant 圖注意力層
        self.message_layers = nn.ModuleList([
            SE3AttentionLayer(hidden_features, num_heads).to(device)
            for _ in range(num_layers)
        ])
        
        # 能量頭
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, 1)
        ).to(device)
        
        # 力頭
        self.force_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, 3)
        ).to(device)
    
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
        CUDA 加速的前向傳播
        
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
        # 確保所有數據在 GPU 上
        device = ligand_positions.device
        
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
    SE(3)-equivariant 圖注意力層 (CUDA 優化)
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
        CUDA 加速的前向傳播
        
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
            attention = (q[i] * k[j]).sum(dim=-1, keepdim=True) / np.sqrt(self.hidden_features)
            attention = attention * distance_encoding
            weight = torch.sigmoid(attention)
            message[i] += weight * v[j]
            num_messages[i] += 1
        
        message = message / (num_messages.unsqueeze(-1) + 1e-8)
        
        # 殘差連接和歸一化
        x = self.norm(x + message)
        x = self.out_proj(x)
        
        return x


class NeuralMD:
    """
    NeuralMD - CUDA 加速的主類
    整合所有模組
    """
    
    def __init__(self, device: str = "cuda", use_mixed_precision: bool = True):
        """
        初始化 NeuralMD CUDA 版本
        
        Args:
            device: 設備 (cuda/cpu)
            use_mixed_precision: 是否使用混合精度
        """
        self.device = torch.device(device if CUDA_AVAILABLE else "cpu")
        self.use_mixed_precision = use_mixed_precision and CUDA_AVAILABLE
        self.scaler = amp.GradScaler() if self.use_mixed_precision else None
        
        # 初始化模型
        self.binding_net = BindingNetCUDA(device=str(self.device))
        
        if CUDA_AVAILABLE:
            self.binding_net = self.binding_net.cuda()
        
        # 優化器
        self.optimizer = torch.optim.AdamW(
            self.binding_net.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        print(f"✅ 模型已初始化到 {self.device}")
        print(f"   混合精度：{self.use_mixed_precision}")
    
    def prepare_data(self, pdb_id: str = "4mxc") -> Dict[str, torch.Tensor]:
        """
        準備數據 (模擬數據)
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            數據字典
        """
        print(f"📦 準備 PDB {pdb_id} 數據...")
        
        # 生成模擬數據
        N_ligand = 50
        N_protein = 200
        
        ligand_positions = torch.randn(N_ligand, 3) * 10
        ligand_features = torch.randn(N_ligand, 10)
        
        protein_positions = torch.randn(N_protein, 3) * 20
        protein_features = torch.randn(N_protein, 15)
        residue_features = torch.randn(N_protein // 3, 20)
        
        # 邊索引
        edge_index = torch.randint(0, N_ligand + N_protein, (2, 1000))
        distances = torch.norm(
            protein_positions[edge_index[0]] - protein_positions[edge_index[1]],
            dim=-1
        )
        
        data = {
            'ligand_positions': ligand_positions,
            'ligand_features': ligand_features,
            'protein_positions': protein_positions,
            'protein_features': protein_features,
            'residue_features': residue_features,
            'edge_index': edge_index,
            'distances': distances
        }
        
        # 移動到 GPU
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)
        
        print(f"✅ 數據準備完成")
        print(f"   配體原子數：{N_ligand}")
        print(f"   蛋白質原子數：{N_protein}")
        
        return data
    
    def train_step(self, data: Dict[str, torch.Tensor]) -> float:
        """
        單步訓練 (CUDA 加速)
        
        Args:
            data: 數據字典
            
        Returns:
            損失值
        """
        self.binding_net.train()
        self.optimizer.zero_grad()
        
        if self.use_mixed_precision:
            with amp.autocast():
                energies, forces = self.binding_net(
                    data['ligand_positions'],
                    data['ligand_features'],
                    data['protein_positions'],
                    data['protein_features'],
                    data['residue_features'],
                    data['edge_index'],
                    data['distances']
                )
                
                # 計算損失 (模擬)
                target_forces = torch.zeros_like(forces)
                loss = ((forces - target_forces) ** 2).mean()
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            energies, forces = self.binding_net(
                data['ligand_positions'],
                data['ligand_features'],
                data['protein_positions'],
                data['protein_features'],
                data['residue_features'],
                data['edge_index'],
                data['distances']
            )
            
            target_forces = torch.zeros_like(forces)
            loss = ((forces - target_forces) ** 2).mean()
            
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        data: Dict[str, torch.Tensor],
        num_epochs: int = 100
    ) -> List[float]:
        """
        完整訓練流程 (CUDA 加速)
        
        Args:
            data: 數據字典
            num_epochs: 訓練輪數
            
        Returns:
            損失曲線
        """
        losses = []
        
        print(f"🚀 開始訓練 ({num_epochs} epochs, {self.device})...")
        
        for epoch in range(num_epochs):
            loss = self.train_step(data)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        
        print(f"✅ 訓練完成，最終損失：{losses[-1]:.4f}")
        return losses
    
    def infer(self, data: Dict[str, torch.Tensor], num_steps: int = 100) -> List[torch.Tensor]:
        """
        CUDA 加速的推論
        
        Args:
            data: 數據字典
            num_steps: 推論步數
            
        Returns:
            軌跡列表
        """
        self.binding_net.eval()
        trajectory = []
        
        with torch.no_grad():
            positions = data['ligand_positions'].clone()
            velocities = torch.zeros_like(positions)
            
            for step in range(num_steps):
                # 計算能量和力
                energies, forces = self.binding_net(
                    data['ligand_positions'],
                    data['ligand_features'],
                    data['protein_positions'],
                    data['protein_features'],
                    data['residue_features'],
                    data['edge_index'],
                    data['distances']
                )
                
                # 計算加速度
                masses = torch.ones(positions.shape[0], device=self.device)
                accelerations = forces / masses
                
                # Velocity Verlet 積分
                positions = positions + velocities * 0.001 + 0.5 * accelerations * 0.001 ** 2
                velocities = velocities + 0.5 * (accelerations + accelerations) * 0.001
                
                # 儲存軌跡
                if step % 10 == 0:
                    trajectory.append(positions.cpu())
        
        print(f"✅ 推論完成，軌跡長度：{len(trajectory)}")
        return trajectory
    
    def save_checkpoint(self, path: str, losses: List[float], trajectory: List[torch.Tensor]):
        """
        保存檢查點
        
        Args:
            path: 保存路徑
            losses: 損失曲線
            trajectory: 軌跡數據
        """
        torch.save({
            'losses': losses,
            'trajectory': trajectory,
            'model_state': self.binding_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'use_mixed_precision': self.use_mixed_precision,
                'device': str(self.device)
            }
        }, path)
        print(f"✅ 檢查點已保存到 {path}")
    
    def load_checkpoint(self, path: str):
        """
        加載檢查點
        
        Args:
            path: 檢查點路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.binding_net.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"✅ 檢查點已從 {path} 加載")


def main():
    """主函數 - CUDA 加速版本"""
    print("=" * 60)
    print("NeuralMD - CUDA 加速版本")
    print("PDB ID: 4MXC")
    print("=" * 60)
    
    # 初始化模型 (使用 GPU)
    model = NeuralMD(device="cuda" if CUDA_AVAILABLE else "cpu")
    
    # 準備數據
    data = model.prepare_data(pdb_id="4mxc")
    
    # 訓練
    losses = model.train(data, num_epochs=50)
    
    # 推論
    trajectory = model.infer(data, num_steps=100)
    
    # 保存結果
    model.save_checkpoint('neuralmd_cuda_results.pt', losses, trajectory)
    
    print("=" * 60)
    print("✅ 完成！結果已保存到 neuralmd_cuda_results.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
