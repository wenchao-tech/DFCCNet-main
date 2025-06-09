from .deconv import DEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from .fusion import PixelAttention,ChannelAttention,SpatialAttention
class HaarDWT(nn.Module):
    def __init__(self):
        super(HaarDWT, self).__init__()
        haar_kernels = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],   # LL
            [[-0.5, -0.5], [0.5, 0.5]], # LH
            [[-0.5, 0.5], [-0.5, 0.5]], # HL
            [[0.5, -0.5], [-0.5, 0.5]]  # HH
        ])  # shape: [4, 2, 2]

        # Add in_channel dimension
        haar_kernels = haar_kernels.unsqueeze(1)  # shape: [4, 1, 2, 2]
        self.register_buffer('weight', haar_kernels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight, stride=2)
        out = out.view(B, C * 4, H // 2, W // 2)
        return out


class WRABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(WRABlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

        self.wavelet_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.dwt = HaarDWT()
        self.reduce_wave = nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1)
    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)


        with torch.no_grad():
            wave_feat = self.dwt(x)            
            wave_feat = F.interpolate(wave_feat, size=res.shape[2:], mode='bilinear', align_corners=False)
            wave_feat = self.reduce_wave(wave_feat)

        fused_feat = torch.cat([res, wave_feat], dim=1)
        fused_feat = self.wavelet_fusion(fused_feat)

        cattn = self.ca(fused_feat)
        sattn = self.sa(fused_feat)
        pattn1 = sattn + cattn
        pattn2 = self.pa(fused_feat, pattn1)

        out = fused_feat * pattn2
        out = out + x
        return out
    
class WRBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(WRBlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        self.wavelet_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.dwt = HaarDWT()
        self.reduce_wave = nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1)
    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        
        
        with torch.no_grad():  
            wave_feat = self.dwt(x)  
            wave_feat = F.interpolate(wave_feat, size=res.shape[2:], mode='bilinear', align_corners=False)
            wave_feat = self.reduce_wave(wave_feat)

        fused = torch.cat([res, wave_feat], dim=1)  
        fused = self.wavelet_fusion(fused)          

        out = self.conv2(fused)
        out = out + x
        return out
