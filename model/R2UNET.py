import torch
import torch.nn as nn

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