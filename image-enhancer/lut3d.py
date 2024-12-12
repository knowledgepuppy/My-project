import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import os

class LUT3D:
    def __init__(self, size: int = 33, device: Optional[torch.device] = None):
        """初始化3D LUT
        
        Args:
            size: LUT的大小 (size x size x size)
            device: 运行设备 (CPU/GPU)
        """
        self.size = size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建标准网格点
        self.grid = self._create_identity_lut()
        
    def _create_identity_lut(self) -> torch.Tensor:
        """创建恒等映射的3D LUT"""
        # 生成标准网格点坐标
        indices = torch.linspace(0, 1, self.size)
        r, g, b = torch.meshgrid(indices, indices, indices, indexing='ij')
        
        # 将网格点堆叠为 (size^3, 3) 的张量
        identity_lut = torch.stack([r, g, b], dim=-1).to(self.device)
        return identity_lut.reshape(-1, 3)
    
    def get_identity_lut(self) -> torch.Tensor:
        """获取恒等映射LUT"""
        return self.grid.clone()
    
    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """应用3D LUT到图像
        
        Args:
            image: 输入图像张量 (B, C, H, W)，值范围[0,1]
            
        Returns:
            处理后的图像张量 (B, C, H, W)
        """
        B, C, H, W = image.shape
        
        # 将图像重塑为 (B*H*W, C)
        image_flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # 计算在LUT中的索引位置
        scaled_input = image_flat * (self.size - 1)
        
        # 确保索引在有效范围内
        scaled_input = torch.clamp(scaled_input, 0, self.size - 1)
        
        lower = scaled_input.floor().long()
        upper = (lower + 1).clamp(max=self.size-1)
        slope = scaled_input - lower
        
        # 获取每个通道的索引
        lower_r, lower_g, lower_b = torch.chunk(lower, 3, dim=1)
        upper_r, upper_g, upper_b = torch.chunk(upper, 3, dim=1)
        slope_r, slope_g, slope_b = torch.chunk(slope, 3, dim=1)

        def get_lut_value(r, g, b):
            """获取LUT中的值
            
            Args:
                r, g, b: 索引张量
                
            Returns:
                对应位置的LUT值
            """
            # 计算在展平的LUT中的索引
            idx = (r * self.size * self.size + g * self.size + b).long()
            
            # 确保索引在有效范围内
            idx = torch.clamp(idx, 0, self.size**3 - 1)
            
            return torch.index_select(self.grid, 0, idx.squeeze(-1))

        # 获取8个相邻点的值
        c000 = get_lut_value(lower_r, lower_g, lower_b)
        c001 = get_lut_value(lower_r, lower_g, upper_b)
        c010 = get_lut_value(lower_r, upper_g, lower_b)
        c011 = get_lut_value(lower_r, upper_g, upper_b)
        c100 = get_lut_value(upper_r, lower_g, lower_b)
        c101 = get_lut_value(upper_r, lower_g, upper_b)
        c110 = get_lut_value(upper_r, upper_g, lower_b)
        c111 = get_lut_value(upper_r, upper_g, upper_b)
        
        # 三线性插值
        c00 = c000 * (1 - slope_b) + c001 * slope_b
        c01 = c010 * (1 - slope_b) + c011 * slope_b
        c10 = c100 * (1 - slope_b) + c101 * slope_b
        c11 = c110 * (1 - slope_b) + c111 * slope_b
        
        c0 = c00 * (1 - slope_g) + c01 * slope_g
        c1 = c10 * (1 - slope_g) + c11 * slope_g
        
        output = c0 * (1 - slope_r) + c1 * slope_r
        
        # 重塑回原始形状
        return output.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    def load_from_file(self, file_path: str) -> None:
        """从文件加载LUT数据
        
        目前支持.npy格式的LUT文件
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LUT文件不存在: {file_path}")
            
        if file_path.endswith('.npy'):
            lut_data = np.load(file_path)
            if lut_data.shape[-1] != 3:
                raise ValueError("LUT数据格式错误：最后一维必须是3（RGB）")
            self.grid = torch.from_numpy(lut_data).float().to(self.device)
        else:
            raise ValueError("不支持的文件格式")
    
    def save_to_file(self, file_path: str) -> None:
        """保存LUT数据到文件"""
        lut_data = self.grid.cpu().numpy()
        np.save(file_path, lut_data)
    
    def modify_lut(self, transform_func) -> None:
        """修改LUT数据
        
        Args:
            transform_func: 接收和返回torch.Tensor的变换函数
        """
        self.grid = transform_func(self.grid)
        
    def create_warm_lut(self, strength: float = 0.5) -> None:
        """创建暖色调LUT"""
        def warm_transform(x):
            # 增加红色分量，减少蓝色分量
            x = x.clone()
            x[:, 0] = torch.clamp(x[:, 0] + strength * 0.2, 0, 1)  # 红色
            x[:, 2] = torch.clamp(x[:, 2] - strength * 0.1, 0, 1)  # 蓝色
            return x
        self.modify_lut(warm_transform)
    
    def create_cool_lut(self, strength: float = 0.5) -> None:
        """创建冷色调LUT"""
        def cool_transform(x):
            # 增加蓝色分量，减少红色分量
            x = x.clone()
            x[:, 2] = torch.clamp(x[:, 2] + strength * 0.2, 0, 1)  # 蓝色
            x[:, 0] = torch.clamp(x[:, 0] - strength * 0.1, 0, 1)  # 红色
            return x
        self.modify_lut(cool_transform) 