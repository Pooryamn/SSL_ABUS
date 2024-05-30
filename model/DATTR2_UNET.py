import torch
import torch.nn as nn

from model.R2UNET import RRCNN_Block
from model.R2UNET import Up
from model.Attention_modules import Attention_block
from model.Attention_modules import CBAM

class DoubleATTR2U_Net(nn.Module):
    """
    Residual Recurrent Unet with base Attention Blobks
    """
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256,512], t=2):
        super(DoubleATTR2U_Net, self).__init__()

        self.Maxpool1 = nn.MaxPool3d(2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(2, stride=2)

        self.RRCNN1 = RRCNN_Block(in_ch, features[0], t=t)
        self.CBAM1 = CBAM(features[0])

        self.RRCNN2 = RRCNN_Block(features[0], features[1], t=t)
        self.CBAM2 = CBAM(features[1])

        self.RRCNN3 = RRCNN_Block(features[1], features[2], t=t)
        self.CBAM3 = CBAM(features[2])

        self.RRCNN4 = RRCNN_Block(features[2], features[3], t=t)
        self.CBAM4 = CBAM(features[3])

        self.RRCNN5 = RRCNN_Block(features[3], features[4], t=t)
        self.CBAM5 = CBAM(features[4])

        self.Up5 = Up(features[4], features[3])
        self.Att5 = Attention_block(F_g=features[3], F_l=features[3], F_int=features[2])
        self.Up_RRCNN5 = RRCNN_Block(features[4], features[3], t=t)

        self.Up4 = Up(features[3], features[2])
        self.Att4 = Attention_block(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_RRCNN4 = RRCNN_Block(features[3], features[2], t=t)

        self.Up3 = Up(features[2], features[1])
        self.Att3 = Attention_block(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_RRCNN3 = RRCNN_Block(features[2], features[1], t=t)

        self.Up2 = Up(features[1], features[0])
        self.Att2 = Attention_block(F_g=features[0], F_l=features[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_Block(features[1], features[0], t=t)

        self.Conv = nn.Conv3d(features[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        e1 = self.RRCNN1(x)
        Att_e1 = self.CBAM1(e1)

        e2 = self.Maxpool1(Att_e1)
        e2 = self.RRCNN2(e2)
        Att_e2 = self.CBAM2(e2)

        e3 = self.Maxpool2(Att_e2)
        e3 = self.RRCNN3(e3)
        Att_e3 = self.CBAM3(e3)

        e4 = self.Maxpool3(Att_e3)
        e4 = self.RRCNN4(e4)
        Att_e4 = self.CBAM4(e4)

        e5 = self.Maxpool4(Att_e4)
        e5 = self.RRCNN5(e5)
        Att_e5 = self.CBAM5(e5)

        d5 = self.Up5(Att_e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = self.padding_func(e4, d5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = self.padding_func(e3, d4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = self.padding_func(e2, d3)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = self.padding_func(e1, d2)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

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
