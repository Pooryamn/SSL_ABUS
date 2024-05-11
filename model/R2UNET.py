import torch
import torch.nn as nn

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

class Recurrent_block(nn.Module):
    """
    Recurrent block for R2U_Net
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch

        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        for i in range(self.t):
            if (i == 0):
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        
        return x1

class RRCNN_Block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_Block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )

        self.Conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)

        out = x1 + x2
        return out

class R2U_Net(nn.Module):
    """
    R2U-Net model
    paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256,512], t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool1 = nn.MaxPool3d(2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_Block(in_ch, features[0], t=t)
        self.RRCNN2 = RRCNN_Block(features[0], features[1], t=t)
        self.RRCNN3 = RRCNN_Block(features[1], features[2], t=t)
        self.RRCNN4 = RRCNN_Block(features[2], features[3], t=t)
        self.RRCNN5 = RRCNN_Block(features[3], features[4], t=t)

        self.Up5 = Up(features[4], features[3])
        self.Up_RRCNN5 = RRCNN_Block(features[4], features[3], t=t)

        self.Up4 = Up(features[3], features[2])
        self.Up_RRCNN4 = RRCNN_Block(features[3], features[2], t=t)

        self.Up3 = Up(features[2], features[1])
        self.Up_RRCNN3 = RRCNN_Block(features[2], features[1], t=t)

        self.Up2 = Up(features[1], features[0])
        self.Up_RRCNN2 = RRCNN_Block(features[1], features[0], t=t)

        self.Conv = nn.Conv3d(features[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

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