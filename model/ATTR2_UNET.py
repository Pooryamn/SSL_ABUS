import torch
import torch.nn as nn

from R2UNET import RRCNN_Block
from R2UNET import Up
from ATT_UNET import Attention_block

class ATTR2U_Net(nn.Module):
    """
    Residual Recurrent Unet with base Attention Blobks
    """
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256,512, t=2]):
        super(ATTR2U_Net, self).__init__()

        self.Maxpool1 = nn.MaxPool3d(2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(2, stride=2)

        self.RRCNN1 = RRCNN_Block(in_ch, features[0], t=t)
        self.RRCNN2 = RRCNN_Block(features[0], features[1], t=t)
        self.RRCNN3 = RRCNN_Block(features[1], features[2], t=t)
        self.RRCNN4 = RRCNN_Block(features[2], features[3], t=t)
        self.RRCNN5 = RRCNN_Block(features[3], features[4], t=t)

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
        
