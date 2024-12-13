import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        attention = self.sigmoid(x_out)
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # 初始特征提取
        self.init_conv = ConvBlock(3, 64)
        
        # 编码器
        self.encoder = nn.Sequential(
            ConvBlock(64, 128, 4, 2, 1),    # 降采样
            ConvBlock(128, 256, 4, 2, 1),   # 降采样
        )
        
        # 注意力模块
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(256)
        
        # 残差块
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(256) for _ in range(6)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 上采样
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 上采样
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, 3, 1, 1),              # 最终输出层
            nn.Tanh()
        )
           
    def enhance_colors(self, img):
        """增强图像颜色"""
        # 转换为HSV空间
        img_hsv = torch.zeros_like(img)
        img_hsv[:, 0] = 0.5 * (torch.sin(3.14159 * img[:, 0]) + 1)  # 调整色调
        img_hsv[:, 1] = torch.clamp(1.2 * img[:, 1], 0, 1)          # 增加饱和度
        img_hsv[:, 2] = torch.clamp(1.1 * img[:, 2], 0, 1)          # 提高亮度
        return img_hsv
    
    def denoise(self, img, kernel_size=3, sigma=1.0):
        """去噪处理"""
        # 实现简单的高斯滤波
        padding = kernel_size // 2
        gaussian_kernel = torch.zeros((kernel_size, kernel_size), device=img.device)
        
        # 创建坐标网格
        center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                x0 = x - center
                y0 = y - center
                # 将计算转换为张量操作
                exponent = torch.tensor(-(x0**2 + y0**2)/(2*sigma**2), device=img.device)
                gaussian_kernel[x, y] = torch.exp(exponent)
        
        # 归一化核
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # 对每个通道进行滤波
        denoised = torch.zeros_like(img)
        for c in range(img.shape[1]):
            denoised[:, c:c+1] = torch.nn.functional.conv2d(
                img[:, c:c+1],
                gaussian_kernel.expand(1, 1, -1, -1),
                padding=padding
            )
        return denoised
    
    def forward(self, x):
        # 初始特征提取
        x = self.init_conv(x)
        
        # 编码
        feat = self.encoder(x)
        
        # 注意力处理
        feat = self.spatial_attention(feat)
        feat = self.channel_attention(feat)
        
        # 残差处理
        feat = self.res_blocks(feat)
        
        # 解码
        out = self.decoder(feat)
        
        # 后处理增强
        out = self.denoise(out)                # 去噪
        out = self.enhance_colors(out)         # 颜色增强
        out = torch.tanh(out)                  # 确保输出在[-1,1]范围内
        
        return out