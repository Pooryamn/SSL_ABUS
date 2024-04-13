import torch
import torch.nn as nn

# Double convolutional block
class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm2d -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# DownSampling block
class Down(nn.Module):
    """
    Max Pooling -> DoubleConv
    """
    
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        
        self.pool = nn.MaxPool3d(2, stride=2)
        self.conv = DoubleConv(in_ch=in_ch, out_ch=out_ch)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        
        return x

# UpsSampling Block
class Up(nn.Module):
    """
    (Conv2dTranspose -> concatenate -> DoubleConv) * 2
    """
    
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input Cropp (if necessary)
        # to be complited if there is any errors
        print(f'Debug: X1{x1.size()} X2:{x2.size()}')
        # Concatenate feature maps
        x = torch.cat([x2, x1], dim=1)
        
        x = self.conv(x)
        
        return x

# U-Net Model
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512], bilinear=False):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(in_ch, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        self.up1 = Up(features[3], features[2], bilinear=bilinear)
        self.up2 = Up(features[2], features[1], bilinear=bilinear)
        self.up3 = Up(features[1], features[0], bilinear=bilinear)
        
        self.outc = nn.Conv3d(features[0], out_ch, kernel_size=1)
        
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        
        return logits