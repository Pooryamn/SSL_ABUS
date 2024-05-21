import torch
import torch.nn as nn
import torch.nn.functional as F

class Detection_model(nn.Module):
    """
    detection model
    """
    def __init__(self, in_ch, out_ch, features = [4,8]):
        super(Detection_model, self).__init__()
        
        self.Maxpool1 = nn.MaxPool3d((4,4,1), stride=(4,4,1))
        self.Maxpool2 = nn.MaxPool3d((2,2,1), stride=(2,2,1))
        self.Maxpool3 = nn.MaxPool3d((2,2,1), stride=(2,2,1))
        self.Maxpool4 = nn.MaxPool3d((2,2,1), stride=(2,2,1))
        self.Maxpool5 = nn.MaxPool3d((2,2,1), stride=(2,2,1))

        self.Conv1 = nn.Conv3d(1, features[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv2 = nn.Conv3d(features[0], features[1], kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv3 = nn.Conv3d(features[1], features[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv4 = nn.Conv3d(features[0], 1, kernel_size=(3,3,3), padding=(1,1,1))

        self.BN1 = nn.BatchNorm3d(features[0])
        self.BN2 = nn.BatchNorm3d(features[1])
        self.BN3 = nn.BatchNorm3d(features[0])
        self.BN4 = nn.BatchNorm3d(1)

        self.Relu = nn.ReLU(inplace=True)

        self.Flatt = nn.Flatten()
        self.FC = nn.Linear(480, 32)
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
        x = self.Maxpool3(x)
        
        x = self.Conv3(x)
        x = self.BN3(x)
        x = self.Relu(x)
        x = self.Maxpool4(x)
        
        x = self.Conv4(x)
        x = self.BN4(x)
        x = self.Relu(x)
        x = self.Maxpool5(x)
        
        x = self.Flatt(x)

        x = self.FC(x)

        x = self.Activation(x)

        x = (x > 0.5) * 1.0

        kernel = torch.tensor([1, 1, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        #opening
        erosion = (F.conv1d(x.long(), kernel.long(), padding=1) == 3) * 1.0
        output = (F.conv1d(erosion.long(), kernel.long(), padding=1) > 0) * 1.0

        return output
