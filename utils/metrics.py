import torch
import numpy as np

def dice_score(predictions, targets):
    """
    Calculate the Dice score between predicted and ground truth masks

    Args:
        predictions: predicted masks
        targets: Ground truth masks

    Returns:
        Average Dice score across all elements in the batch
    """
    smooth = 1.0

    # flatten predictions and targets
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Thresholding
    predictions = (predictions > 0.5).float()
    
    # Calculate Intersection and Union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()

    # # Dice score formula
    dice = (2 * intersection + smooth) / (union + smooth)

    return dice

def PSNR(data, noisy_data):

    MSE = np.mean((data - noisy_data) ** 2)

    if (MSE == 0):
        # no noise is present in the signal
        return 100

    max_pixel = 1.0

    psnr = 20 * np.log10(max_pixel / (np.sqrt(MSE)))

    return psnr 

def Sensitivity(inputs, targets):
        # Debug
        print('===== DEBUG =====')
        print(f'inputs shape 1: {inputs.size()}')
        print(f'targets shape 1: {targets.size()}')

        print(f'inputs: {inputs}')
        print(f'targets: {targets}')

        inputs = inputs.view(-1)
        targets = targets.view(-1)


        print(f'inputs shape 2: {inputs.size()}')
        print(f'targets shape 2: {targets.size()}')

        # TP, FP, FN
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        print(f'TP: {TP}')
        print(f'FP: {FP}')
        print(f'FN: {FN}')

        sensitivity = TP / (TP + FN)

        print(f'sensitivity{sensitivity}')
        exit()

        return sensitivity, FP