import torch
import torch.nn as nn
from model.Attention_modules import SpatialAttention

class Detection_model(nn.Module):
    """
    detection model
    """
    def __init__(self, in_ch, out_ch, features = [16, 32, 64], threshold=0.5):
        super(Detection_model, self).__init__()

        self.Threshold = threshold

        self.SA = SpatialAttention(kernel_size=5)
        
        self.Maxpool1 = nn.MaxPool3d((3,2,1), stride=(3,2,1))
        self.Maxpool2 = nn.MaxPool3d((2,2,1), stride=(2,2,1))
        self.Maxpool3 = nn.MaxPool3d((2,2,1), stride=(2,2,1))
        self.Maxpool4 = nn.MaxPool3d((2,2,1), stride=(2,2,1))

        self.Conv1 = nn.Conv3d(1, features[0], kernel_size=(7,7,1), padding=(0,0,0))
        self.Conv2 = nn.Conv3d(features[0], features[1], kernel_size=(5,5,1), padding=(0,0,0))
        self.Conv3 = nn.Conv3d(features[1], features[2], kernel_size=(5,5,1), padding=(0,0,0))
        self.Conv4 = nn.Conv3d(features[2], features[1], kernel_size=(3,3,1), padding=(0,0,0))
        self.Conv5 = nn.Conv3d(features[1], features[0], kernel_size=(3,3,1), padding=(0,0,0))
        self.Conv6 = nn.Conv3d(features[0], 1, kernel_size=(7,7,1), padding=(0,0,0))

        self.BN1 = nn.BatchNorm3d(features[0])
        self.BN2 = nn.BatchNorm3d(features[1])
        self.BN3 = nn.BatchNorm3d(features[2])
        self.BN4 = nn.BatchNorm3d(features[1])
        self.BN5 = nn.BatchNorm3d(features[0])

        self.Relu = nn.ReLU(inplace=True)

        self.Flatt = nn.Flatten()
        #self.FC = nn.Linear(1568, 32)
        self.Activation = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.Maxpool1(x)
        
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Maxpool2(x)
        
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.Relu(x)
        x = self.SA(x)
        x = self.Maxpool3(x)
        
        x = self.Conv3(x)
        x = self.BN3(x)
        x = self.Relu(x)
        x = self.Maxpool4(x)
        
        x = self.Conv4(x)
        x = self.BN4(x)
        x = self.Relu(x)

        x = self.Conv5(x)
        x = self.BN5(x)
        x = self.Relu(x)

        x = self.Conv6(x)
        x = self.Relu(x)
        
        x = self.Flatt(x)

        #x = self.FC(x)

        x = self.Activation(x)

        x = (x > self.Threshold) * 1.0

        return x