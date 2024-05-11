import torch
import torch.nn as nn

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