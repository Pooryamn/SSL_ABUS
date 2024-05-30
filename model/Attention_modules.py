import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc(avg_out)
        
        max_out = self.max_pool(x)
        max_out = self.fc(max_out)

        out = avg_out + max_out
        out = x * self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=2, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Module
    paper: https://doi.org/10.1016/j.ultrasmedbio.2020.06.015
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
        
        # if needed:
        if (g.shape[2] != x.shape[2]):
            g_slice = g[:,:,-1,:,:]
            g = torch.cat((g, g_slice[:,:,None,:,:]), 2)
        
        if (g.shape[3] != x.shape[3]):
            g_slice = g[:,:,:,-1,:]
            g = torch.cat((g, g_slice[:,:,:,None,:]), 3)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi

        return out


class GCN(nn.Module):
    def __init__(self, in_ch):
        super(GCN, self).__init__()

        self.conv_11 = nn.Conv3d(in_ch, 64, kernel_size=(7,7,1), padding=(3,3,0))
        self.conv_12 = nn.Conv3d(64, 64, kernel_size=(1,1,7), padding=(0,0,3))
        
        self.conv21 = nn.Conv3d(in_ch, 64, kernel_size=(1,1,7), padding=(0,0,3))
        self.conv22 = nn.Conv3d(64, 64, kernel_size=(7,7,1), padding=(3,3,0))

    def forward(self,x):
        out1 = self.conv_11(x)
        out1 = self.conv_12(out1)

        out2 = self.conv21(x)
        out2 = self.conv22(out2)

        return out1 + out2

class BR(nn.Module):
    def __init__(self, in_ch):
        super(BR, self).__init__()
        
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, stride=1)
        
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv3d(in_ch, 64, kernel_size=3, padding=1, stride=1)
    
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.prelu(x1)
        x1 = self.conv2(x1)

        return x + x1




class Attention_block_v2(nn.Module):
    def __init__(self, in_ch):
        super(Attention_block_v2, self).__init__()

        self.convg = nn.Conv3d(in_ch, 128, kernel_size=1, stride=1, padding=0)
        self.convx = nn.Conv3d(in_ch, 128, kernel_size=1, stride=1, padding=0)
        
        self.prelux = nn.PReLU()
        self.prelug = nn.PReLU()
        
        self.gcn = GCN(128)
        self.br = BR(64)

    def forward(self, g, x):
        g = self.convg(g)
        g = self.prelug(g)

        x = self.convx(x)
        x = self.prelux(x)

        out = self.gcn(g + x)

        out = self.br(out)

        return out

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Attention_block_v21(nn.Module):
    def __init__(self, in_ch):
        super(Attention_block_v21, self).__init__()

        self.convg = nn.Conv3d(in_ch, 128, kernel_size=1, stride=1, padding=0)
        self.convx = nn.Conv3d(in_ch, 128, kernel_size=1, stride=1, padding=0)

        self.prelux = nn.PReLU()
        self.prelug = nn.PReLU()

        self.dwsc1 = depthwise_separable_conv(128, 128)
        self.prelu1= nn.PReLU()

        self.dwsc2 = depthwise_separable_conv(128, 128)
        self.prelu2= nn.PReLU()

        self.conv = nn.Conv3d(128, 128, kernel_size=1,  stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        
        x1 = self.convx(x)
        x1 = self.prelux(x1)

        g = self.convg(g)
        g = self.prelug(g)

        out = self.dwsc1(x1 + g)
        out = self.prelu1(out)

        out = self.dwsc2(out)
        out = self.prelu2(out)

        out = x * out

        return out