import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        Dice_BCE = BCE + dice_loss
        return Dice_BCE
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    

class FocalLoss(nn.Module):
    """
    Source paper:  https://arxiv.org/abs/1708.02002
    """
    def __init__(self, ALPHA = 0.8, GAMMA = 2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = ALPHA
        self.gamma = GAMMA

    def forward(self, inputs, targets, smooth=1):    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        #focal_loss.requires_grad = True
                       
        return focal_loss

class DualSSIMLoss(nn.Module):
    def __init__(self, ALPHA, BETA):
        super(DualSSIMLoss, self).__init__()

        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel = False, kernel_size=5,data_range=1.0)
        
        self.alpha = ALPHA
        self.beta = BETA
    
    def forward(self, predictions, inputs, targets):

        # calculate SSIM between target and prediction
        SSIM_pred_target = self.ssim(targets, predictions)
        Loss1 = 1 - SSIM_pred_target

        # calculate SSIM between Input and prediction
        SSIM_pred_input = self.ssim(inputs, predictions)
        Loss2 = 1 - SSIM_pred_input

        Loss = (self.alpha * Loss1) + (self.beta * Loss2)

        return Loss, SSIM_pred_target

        
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel = False, kernel_size=5,data_range=1.0)
    
    def forward(self, predictions, targets):

        # calculate SSIM between target and prediction
        SSIM_pred_target = self.ssim(targets, predictions)
        Loss = 1 - SSIM_pred_target

        return Loss, SSIM_pred_target

class TverskyLoss(nn.Module):
    """
    Source paper: https://arxiv.org/abs/1706.05721
    """
    def __init__(self, ALPHA, BETA, weight=None, size_average=True, smooth=1):
        super(TverskyLoss, self).__init__()

        self.alpha = ALPHA
        self.beta = BETA
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # TP, FP, FN
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + (self.alpha * FP) + (self.beta * FN) + self.smooth)
        Tversky.requires_grad = True

        return 1 - Tversky

class Detection_loss(nn.Module):
    def __init__(self, ALPHA, BETA):
        super(Detection_loss, self).__init__()     

        self.alpha = ALPHA
        self.beta = BETA
    
    def forward(self, predictions, targets):

        focal = FocalLoss()
        
        FOCAL = focal(predictions[:,:,0].float(), targets[:,0,:,0].float())

        mse = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        MSE = mse(predictions[:,:,1:], targets[:,0,:,1:])

        Loss = (self.alpha * FOCAL) + (self.beta * MSE)

        return Loss