import torch
import torch.nn as nn

from model.Attention_modules import Attention_block

# Double convolutional block
class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm2d -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
    Upsample -> Conv -> Batch norm -> ReLU
    """
    
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)

        return x

class Attention_Unet(nn.Module):
    """
    Attention Unet
    paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256,512]):
        super(Attention_Unet, self).__init__()

        self.Maxpool1 = nn.MaxPool3d(2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(2, stride=2)

        self.Conv1 = DoubleConv(in_ch, features[0])
        self.Conv2 = DoubleConv(features[0], features[1])
        self.Conv3 = DoubleConv(features[1], features[2])
        self.Conv4 = DoubleConv(features[2], features[3])
        self.Conv5 = DoubleConv(features[3], features[4])

        self.Up5 = Up(features[4], features[3])
        self.Att5 = Attention_block(F_g= features[3], F_l=features[3], F_int=features[2])
        self.Up_conv5 = DoubleConv(features[4], features[3])

        self.Up4 = Up(features[3], features[2])
        self.Att4 = Attention_block(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv4 = DoubleConv(features[3], features[2])

        self.Up3 = Up(features[2], features[1])
        self.Att3 = Attention_block(F_g=features[1], F_l=features[1],F_int=features[0])
        self.Up_conv3 = DoubleConv(features[2], features[1])

        self.Up2 = Up(features[1], features[0])
        self.Att2 = Attention_block(F_g=features[0], F_l=features[0], F_int=32)
        self.Up_conv2 = DoubleConv(features[1], features[0])

        self.Conv = nn.Conv3d(features[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        
        e1 = self.Conv1(x)
    
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
     
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
      
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        d5 = self.Up5(e5) 
        x4 = self.Att5(g=d5, x=e4)
        d5 = self.padding_func(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = self.padding_func(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = self.padding_func(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = self.padding_func(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        #out = self.activation(out)

        return out
    
    def padding_func(self, x, d):
        if (x.shape != d.shape):
            # if needed:
            if (x.shape[2] != d.shape[2]):
                d_slice = d[:,:,-1,:,:]
                d = torch.cat((d, d_slice[:,:,None,:,:]), 2)
        
            if (x.shape[3] != d.shape[3]):
                d_slice = d[:,:,:,-1,:]
                d = torch.cat((d, d_slice[:,:,:,None,:]), 3)

        return d