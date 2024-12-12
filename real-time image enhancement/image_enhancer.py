import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import time
from collections import deque
import tkinter as tk
from tkinter import ttk
import os
from lut3d import LUT3D

class ImageEnhancer:
    def __init__(self):
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化增强参数
        self.brightness = 1.0
        self.contrast = 1.0
        self.sharpness = 1.0
        self.saturation = 1.0
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 初始化3D LUT相关属性
        self.lut = LUT3D(size=33, device=self.device)
        self.current_lut_name = 'identity'
        self.lut_strength = 1.0
        self.available_luts = self._load_preset_luts()
        
        # 场景识别相关
        self.auto_mode = False
        self.scene_presets = self._init_scene_presets()
        
        # 参数历史记录
        self.history = deque(maxlen=20)
        self.history_position = -1
        self._save_history()  # 保存初始状态
        
        # GUI相关
        self.gui = None
        self.parameter_widgets = {}

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """将OpenCV图像帧转换为PyTorch张量
        
        Args:
            frame: BGR格式的numpy数组图像
            
        Returns:
            处理后的图像张量 (1, 3, H, W)，值范围[0,1]
        """
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为float32并归一化到[0,1]
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        
        # 转换为PyTorch张量并调整维度顺序
        frame_tensor = torch.from_numpy(frame_norm).to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor

    def postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """将PyTorch张量转换回OpenCV图像格式
        
        Args:
            frame_tensor: 值范围[0,1]的图像张量 (1, 3, H, W)
            
        Returns:
            BGR格式的numpy数组图像
        """
        # 转换回numpy数组并调整维度顺序
        frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 转换值范围到[0,255]并转为uint8类型
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # RGB转BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
        
    def _init_scene_presets(self) -> Dict[str, Dict[str, float]]:
        """初始化不同场景的预设参数"""
        return {
            'portrait': {
                'brightness': 1.1,
                'contrast': 1.2,
                'saturation': 1.1,
                'sharpness': 1.3,
                'lut_strength': 0.8
            },
            'landscape': {
                'brightness': 1.0,
                'contrast': 1.3,
                'saturation': 1.4,
                'sharpness': 1.5,
                'lut_strength': 0.9
            },
            'night': {
                'brightness': 1.4,
                'contrast': 1.1,
                'saturation': 0.9,
                'sharpness': 0.8,
                'lut_strength': 0.7
            }
        }
    
    def _load_preset_luts(self) -> dict:
        """加载预设的LUT"""
        presets = {
            'identity': self.lut.get_identity_lut(),
        }
        
        # 添加预设的暖色和冷色LUT
        warm_lut = self.lut.get_identity_lut()
        self.lut.create_warm_lut(0.5)
        presets['warm'] = warm_lut
        
        cool_lut = self.lut.get_identity_lut()
        self.lut.create_cool_lut(0.5)
        presets['cool'] = cool_lut
        
        return presets
    
    def analyze_scene(self, frame_tensor: torch.Tensor) -> str:
        """基于简单规则分析场景类型"""
        # 计算基本图像统计信息
        brightness = torch.mean(frame_tensor).item()
        saturation = torch.std(frame_tensor).item()
        
        # 简单场景判断规则
        if brightness < 0.3:
            return 'night'
        elif saturation > 0.5:
            return 'landscape'
        else:
            return 'portrait'
    
    def get_current_params(self) -> Dict[str, float]:
        """获取当前参数状态"""
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'sharpness': self.sharpness,
            'lut_strength': self.lut_strength
        }
    
    def apply_params(self, params: Dict[str, float]) -> None:
        """应用参数设置"""
        self.brightness = params['brightness']
        self.contrast = params['contrast']
        self.saturation = params['saturation']
        self.sharpness = params['sharpness']
        self.lut_strength = params['lut_strength']
        
        # 更新GUI显示（如果存在）
        if self.gui is not None:
            self._update_gui_values()
    
    def _save_history(self) -> None:
        """保存当前参数状态到历史记录"""
        current_params = self.get_current_params()
        
        # 如果在历史中间位置进行了新的修改，删除后面的记录
        if self.history_position < len(self.history) - 1:
            while len(self.history) > self.history_position + 1:
                self.history.pop()
        
        self.history.append(current_params)
        self.history_position = len(self.history) - 1
    
    def undo(self) -> None:
        """撤销操作"""
        if self.history_position > 0:
            self.history_position -= 1
            self.apply_params(self.history[self.history_position])
    
    def redo(self) -> None:
        """重做操作"""
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self.apply_params(self.history[self.history_position])
    
    def init_gui(self) -> None:
        """初始化图形界面"""
        self.gui = tk.Tk()
        self.gui.title("图像增强控制面板")
        
        # 创建参数控制区
        self._create_parameter_controls()
        
        # 创建按钮区
        self._create_control_buttons()
        
        # 更新GUI显示
        self._update_gui_values()
    
    def _create_parameter_controls(self) -> None:
        """创建参数控制滑块"""
        params_frame = ttk.LabelFrame(self.gui, text="参数调整")
        params_frame.pack(padx=5, pady=5, fill="x")
        
        parameters = {
            'brightness': ('亮度', 0.0, 2.0),
            'contrast': ('对比度', 0.0, 2.0),
            'saturation': ('饱和度', 0.0, 2.0),
            'sharpness': ('锐化', 0.0, 2.0),
            'lut_strength': ('LUT强度', 0.0, 1.0)
        }
        
        for param, (label, min_val, max_val) in parameters.items():
            frame = ttk.Frame(params_frame)
            frame.pack(fill="x", padx=5, pady=2)
            
            ttk.Label(frame, text=label).pack(side="left")
            
            var = tk.DoubleVar(value=getattr(self, param))
            slider = ttk.Scale(frame, from_=min_val, to=max_val,
                             variable=var, orient="horizontal")
            slider.pack(side="right", fill="x", expand=True)
            
            self.parameter_widgets[param] = (var, slider)
            var.trace_add("write", lambda *args, p=param: self._on_parameter_change(p))
    
    def _create_control_buttons(self) -> None:
        """创建控制按钮"""
        button_frame = ttk.Frame(self.gui)
        button_frame.pack(padx=5, pady=5, fill="x")
        
        ttk.Button(button_frame, text="撤销", command=self.undo).pack(side="left", padx=2)
        ttk.Button(button_frame, text="重做", command=self.redo).pack(side="left", padx=2)
        
        auto_var = tk.BooleanVar(value=self.auto_mode)
        ttk.Checkbutton(button_frame, text="自动模式", 
                       variable=auto_var,
                       command=lambda: setattr(self, 'auto_mode', auto_var.get())
                       ).pack(side="right", padx=2)
    
    def _update_gui_values(self) -> None:
        """更新GUI显示的参数值"""
        if self.gui is None:
            return
            
        for param, (var, _) in self.parameter_widgets.items():
            var.set(getattr(self, param))
    
    def _on_parameter_change(self, parameter: str) -> None:
        """参数变化回调"""
        if not self.auto_mode:
            value = self.parameter_widgets[parameter][0].get()
            setattr(self, parameter, value)
            self._save_history()
    
    def enhance_frame(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """增强图像帧"""
        if self.auto_mode:
            # 分析场景并应用相应的预设
            scene_type = self.analyze_scene(frame_tensor)
            preset = self.scene_presets[scene_type]
            old_params = self.get_current_params()
            self.apply_params(preset)
            enhanced = self._enhance_frame_internal(frame_tensor)
            self.apply_params(old_params)
            return enhanced
        else:
            return self._enhance_frame_internal(frame_tensor)
    
    def _enhance_frame_internal(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """内部图像增强处理
        
        Args:
            frame_tensor: 输入图像张量 (B, C, H, W)，值范围[0,1]
            
        Returns:
            处理后的图像张量 (B, C, H, W)，值范围[0,1]
        """
        # 确保输入张量在[0,1]范围内
        frame_tensor = torch.clamp(frame_tensor, 0, 1)
        
        # 亮度调整
        enhanced = frame_tensor * self.brightness
        
        # 对比度调整
        mean = torch.mean(enhanced)
        enhanced = (enhanced - mean) * self.contrast + mean
        
        # 饱和度调整
        gray = torch.mean(enhanced, dim=1, keepdim=True)
        enhanced = torch.lerp(gray, enhanced, self.saturation)
        
        # 锐化处理
        if self.sharpness > 1.0:
            laplacian = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32).to(self.device)
            laplacian = laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sharp = F.conv2d(enhanced, laplacian, padding=1, groups=3)
            enhanced = enhanced + (self.sharpness - 1) * sharp
        
        # 应用3D LUT
        if self.lut_strength > 0:
            try:
                lut_enhanced = self.lut.apply(enhanced)
                enhanced = torch.lerp(enhanced, lut_enhanced, self.lut_strength)
            except Exception as e:
                print(f"应用LUT时出错: {e}")
        
        # 确保输出在[0,1]范围内
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced

    def run(self):
        # 初始化GUI
        self.init_gui()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                start_time = time.time()
                
                frame_tensor = self.preprocess_frame(frame)
                enhanced_tensor = self.enhance_frame(frame_tensor)
                enhanced_frame = self.postprocess_frame(enhanced_tensor)
                
                fps = 1.0 / (time.time() - start_time)
                
                # 显示信息
                cv2.putText(enhanced_frame, 
                           f'LUT: {self.current_lut_name} ({self.lut_strength:.2f})',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(enhanced_frame,
                           f'Mode: {"Auto" if self.auto_mode else "Manual"}',
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Enhanced', enhanced_frame)
                cv2.imshow('Original', frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('['):
                    self.set_lut_strength(self.lut_strength - 0.1)
                elif key == ord(']'):
                    self.set_lut_strength(self.lut_strength + 0.1)
                elif key == ord('z'):  # 撤销
                    self.undo()
                elif key == ord('y'):  # 重做
                    self.redo()
                
                # 更新GUI
                self.gui.update()
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.gui is not None:
                self.gui.destroy()

if __name__ == '__main__':
    enhancer = ImageEnhancer()
    enhancer.run()