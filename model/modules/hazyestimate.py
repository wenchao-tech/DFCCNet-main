import torch.nn as nn
import torch.nn.functional as F
import torch
class FourierBranch(nn.Module):
    def __init__(self):
        super(FourierBranch, self).__init__()
    
    def forward(self, x):
        # x: [B, C, H, W]，取值范围[0,1]
        x_fft = torch.fft.fft2(x, norm='ortho')  # 复数Tensor
        mag = torch.abs(x_fft)  # [B, C, H, W]幅值谱
        mag_avg = mag.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        mag_log = torch.log1p(mag_avg)           # 对数变换，压缩动态范围
        return mag_log  # [B, 1, H, W]

# ---------------------------
# 2. Haar 小波变换分支：利用固定权重实现二维 Haar 小波变换
# ---------------------------
class HaarWaveletTransform(nn.Module):
    def __init__(self, in_channels=3):
        super(HaarWaveletTransform, self).__init__()
        # 定义 4 个 Haar 滤波器
        kernel_approx = torch.tensor([[0.5, 0.5],
                                      [0.5, 0.5]], dtype=torch.float32)
        kernel_horiz = torch.tensor([[0.5, 0.5],
                                     [-0.5, -0.5]], dtype=torch.float32)
        kernel_vert = torch.tensor([[0.5, -0.5],
                                    [0.5, -0.5]], dtype=torch.float32)
        kernel_diag = torch.tensor([[0.5, -0.5],
                                    [-0.5, 0.5]], dtype=torch.float32)
        kernels = torch.stack([kernel_approx, kernel_horiz, kernel_vert, kernel_diag], dim=0)  # (4, 2, 2)
        kernels = kernels.unsqueeze(1)  # (4, 1, 2, 2)
        weight = kernels.repeat(in_channels, 1, 1, 1)  # (in_channels*4, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.groups = in_channels

    def forward(self, x):
        # x: [B, in_channels, H, W]
        out = F.conv2d(x, self.weight, bias=None, stride=2, groups=self.groups)  # [B, in_channels*4, H/2, W/2]
        out_upsampled = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out_upsampled  # [B, in_channels*4, H, W]

# ---------------------------
# 3. 融合网络：结合傅里叶、小波特征并加入注意力分支
# ---------------------------
class FourierWaveletHazeEstimatorWithAttention(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16, out_channels=1):
        super(FourierWaveletHazeEstimatorWithAttention, self).__init__()
        # 傅里叶分支
        self.fourier = FourierBranch()
        # 小波分支：输入 in_channels 得到 in_channels*4 通道（3*4=12）
        self.haar = HaarWaveletTransform(in_channels=in_channels)
        # 融合后总通道数：1 (傅里叶) + 12 (小波) = 13
        # 注意力分支：从 13 通道特征生成一个注意力图（单通道，值在0~1之间）
        self.attention_conv = nn.Sequential(
            nn.Conv2d(13, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 融合卷积分支：利用注意力增强后的特征生成雾霭密度图
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(13, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )
        self.dark_down1 = nn.Sequential(nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1),
                                   )
        self.dark_down2 = nn.Sequential(nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1),
                                  )
        
    def forward(self, x):
        B, C, H, W = x.shape

    # Step 1: 对输入进行 padding（保证 H, W 为偶数）
        pad_H = H % 2
        pad_W = W % 2
        if pad_H != 0 or pad_W != 0:
            x = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')  # pad: (left, right, top, bottom)

        # x: [B, 3, H, W]
        fourier_feat = self.fourier(x)  # [B, 1, H, W]
        haar_feat = self.haar(x)        # [B, 12, H, W]
        fused = torch.cat([fourier_feat, haar_feat], dim=1)  # [B, 13, H, W]
        # 注意力图计算
        att_map = self.attention_conv(fused)  # [B, 1, H, W]，数值在 [0,1]
        # 利用注意力对 fused 特征进行加权
        attended = fused * att_map
        density = self.fusion_conv(attended)  # [B, 1, H, W]
        density1 = self.dark_down1(density)
        density2 = self.dark_down2(density1)
        return density, density1,density2,att_map