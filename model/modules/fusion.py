from torch import nn
import torch
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
    
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
# # 细节增强
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)*x
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        x = x * y
        return x
    
class multi_ca(nn.Module):
    def __init__(self, dim, bins = [1,2,3,6]):
        super(multi_ca, self).__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AvgPool2d(bin),
                nn.Conv2d(dim, dim, kernel_size=1),
                CALayer (dim),
            ))
        self.features = nn.ModuleList(self.features)
        self.cov2 = nn.Conv2d(dim * 5, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return self.cov2(torch.cat(out, 1))

class MSAFF(nn.Module):
    def __init__(self, dim):
        super(MSAFF, self).__init__()
        self.multi_ca = multi_ca(dim)

        self.SA1 = SpatialAttention(1)
        self.SA3 = SpatialAttention(3)
        self.SA5 = SpatialAttention(5)
        self.SA7 = SpatialAttention(7)
        self.cov3 = nn.Conv2d(dim*4, dim, kernel_size=3, padding=1)
        self.t = torch.nn.Tanh()
    def forward(self, x):
        # x = changeshape(x)
        input_x = x
        x = self.multi_ca(x)

        xx1 = self.SA1(x)
        xx2 = self.SA3(x)
        xx3 = self.SA5(x)
        xx4 = self.SA7(x)
        x = torch.cat((xx1, xx2, xx3, xx4), dim=1)
        x =  self.cov3(x)
        x = self.t(x) + input_x
        # x = changeshape3(x)
        return x

#特征融合
class LFM(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(LFM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.t = nn.Sequential(
            nn.Conv2d(channel*2, channel//reduction, kernel_size=to_2tuple(3), padding=1, bias=bias),
            # True表示直接修改原始张量
            nn.ReLU(inplace=False),
            nn.Conv2d(channel//reduction, channel*2, kernel_size=to_2tuple(3), padding=1,bias=bias),
            nn.Sigmoid()
        )
    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        x = in_feats
        pa = self.t(x)
        j = torch.mul(pa, in_feats)
        
        return j
    

class MSLFBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super(MSLFBlock, self).__init__()
        self.msaff = MSAFF(dim)                  # 多尺度空间注意力
        self.multi_ca = multi_ca(dim)           # 金字塔通道注意力
        self.pa = PixelAttention(dim)            # 像素注意力
        self.fusion = LFM(dim, reduction)        # 可学习特征融合
        self.conv_out = nn.Conv2d(dim*2, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y                          # 初始融合
        sa_feat = self.msaff(initial)            # 多尺度空间注意力
        ca_feat = self.multi_ca(initial)         # 多尺度通道注意力
        attn_feat = sa_feat + ca_feat            # 注意力融合

        pa_feat = self.sigmoid(self.pa(initial, attn_feat))  # 像素注意力引导图

        fused = pa_feat * x + (1 - pa_feat) * y  # 按像素融合
        fused = self.fusion([initial, fused])    # 进一步用CFF融合
        output = self.conv_out(fused)            # 卷积变换输出
        return output