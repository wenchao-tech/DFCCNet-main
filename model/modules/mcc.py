import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 自定义小波变换层（Haar小波分解低频提取）
class HaarWaveletLowpass(Function):
    @staticmethod
    def forward(ctx, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)
        # Haar小波分解
        ll = F.avg_pool2d(x, kernel_size=2, stride=2)  # 低频分量
        ctx.save_for_backward(x)
        return ll.reshape(B, C, H//2, W//2)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        B, C, H, W = grad_output.shape
        grad_input = F.interpolate(grad_output, scale_factor=2, mode='nearest')
        return grad_input.reshape(B, C, H*2, W*2)

# 动态化路由
class HDR(nn.Module):
    def __init__(self, in_channels, groups=8):
        super().__init__()
        self.groups = groups
        self.base_channels = in_channels
        
        # 多尺度特征提取
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(2**i, stride=2**i),
                nn.Conv2d(in_channels, in_channels//self.groups, 3, padding=1)
            ) for i in range(1, 4)
        ])
        
        # 动态路由权重生成
        route_in_channels = in_channels + 3*(in_channels//self.groups)
        self.route = nn.Sequential(
            nn.Conv2d(route_in_channels, route_in_channels//2, 1),
            nn.GELU(),
            nn.Conv2d(route_in_channels//2, self.groups*3, 1),
            nn.Softmax(dim=1)
        )
        
        # 修正后的投影层（关键修复点）
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),  # 输入通道修正为in_channels
                nn.BatchNorm2d(in_channels)
            ) for _ in range(3)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 多尺度特征生成
        features = [x] + [m(x) for m in self.scales]
        
        # 特征对齐处理
        aligned_features = []
        for i, f in enumerate(features):
            if i == 0:
                aligned = f
            else:
                aligned = F.interpolate(f, size=(H,W), mode='bilinear')
                if aligned.size(1) != C//self.groups:
                    aligned = F.pad(aligned, (0,0,0,0,0,C//self.groups-aligned.size(1)))
            aligned_features.append(aligned)
        
        # 路由权重计算
        route_weight = self.route(torch.cat(aligned_features, dim=1))
        route_weight = route_weight.view(B, 3, self.groups, H, W)
        
        # 安全融合（维度验证）
        out = x.clone()
        for i in range(3):
            # 获取加权特征 [B, C//G, H, W]
            weighted_feat = aligned_features[i+1]
            
            # 获取路由权重 [B, G, H, W]
            current_weight = route_weight[:, i]
            
            # 维度扩展与融合
            weighted = weighted_feat.unsqueeze(1) * current_weight.unsqueeze(2)
            weighted = weighted.view(B, -1, H, W)  # [B, C, H, W]
            
            # 通道投影（输入通道已修正为C）
            out += self.proj[i](weighted)
            
        return out

class MCCWithWaveletSA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.wavelet_lowpass = HaarWaveletLowpass()
        
        # # 自注意力机制（简化版Transformer）
        # self.self_attn = nn.MultiheadAttention(
        #     embed_dim=in_channels, 
        #     num_heads=4,
        #     dropout=0.1,
        #     batch_first=True
        # )
        
        self.self_attn = HDR(in_channels, groups=8)
        
        # 低频先验融合
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),
            nn.GELU()
        )
        
        # 局部细化
        self.local_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )

    def forward(self, x):
        # 小波低频提取
        x_low = self.wavelet_lowpass.apply(x)  # [B, C, H/2, W/2]
        
        # 自注意力处理（空间-通道联合）
        B, C, H, W = x.shape
        # x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
        # attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)  # [B, HW, C]
        # attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        attn_out = self.self_attn(x)  # 直接处理4D张量
        
        # 低频先验融合
        x_low_up = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([attn_out, x_low_up], dim=1))
        
        # 局部细化 + 残差
        out = fused + self.local_refine(fused)
        return out