import torch.nn as nn
import torch
from .modules import WRABlock, WRBlock, MSLFBlock,MCCWithWaveletSA,FourierWaveletHazeEstimatorWithAttention

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
 
class ColorResidualGate(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ColorResidualGate, self).__init__()
        self.color_correct = MCCWithWaveletSA(in_channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        corrected = self.color_correct(x)  
        residual = corrected - x           
        gate_weight = self.gate(x)         
        output = x + residual * gate_weight
        return output

class DFCCNet(nn.Module):
    def __init__(self, base_dim=32):
        super(DFCCNet, self).__init__()
        
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = WRBlock(default_conv, base_dim, 3)
        self.down_level1_block2 = WRBlock(default_conv, base_dim, 3)
        self.down_level1_block3 = WRBlock(default_conv, base_dim, 3)
        self.down_level1_block4 = WRBlock(default_conv, base_dim, 3)
        self.up_level1_block1 = WRBlock(default_conv, base_dim, 3)
        self.up_level1_block2 = WRBlock(default_conv, base_dim, 3)
        self.up_level1_block3 = WRBlock(default_conv, base_dim, 3)
        self.up_level1_block4 = WRBlock(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = WRBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = WRBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = WRBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = WRBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = WRBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = WRBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = WRBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = WRBlock(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block2 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block3 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block4 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block5 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block6 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block7 = WRABlock(default_conv, base_dim * 4, 3)
        self.level3_block8 = WRABlock(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = MSLFBlock(base_dim * 4, reduction=8)
        self.mix2 = MSLFBlock(base_dim * 2, reduction=4)
        self.mix3 = MSLFBlock(base_dim, reduction=2)
        self.hazyestimate = FourierWaveletHazeEstimatorWithAttention(3,16,base_dim)
        self.crg_color1 = ColorResidualGate(base_dim)
        self.crg_color2 = ColorResidualGate(base_dim * 4,reduction=16)

        self.fuse_color = nn.Conv2d(base_dim*2, base_dim, kernel_size=1)
        self.fuse_color1 = nn.Conv2d(base_dim*8, base_dim*4, kernel_size=1)
    def forward(self, x):
        
        x_down1 = self.down1(x)
        color_corrected = self.crg_color1(x_down1)  

        fused_color = torch.cat([x_down1, color_corrected], dim=1)
        x_down1 = self.fuse_color(fused_color)  
        
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)


        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)


        x_down3 = self.down3(x_down2_init)
        ccrd = self.crg_color2(x_down3)
        fused_color3 = torch.cat([x_down3, ccrd], dim=1)  
        x_down3 = self.fuse_color1(fused_color3)

        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)

        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        # print('x_up1.shape',x_up1.shape)

        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        # print('x_up2.shape',x_up2.shape)

        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)

        x_level1_mix = self.mix3(x_down1, x_up2)
        
        out = self.up3(x_level1_mix)

        return out
