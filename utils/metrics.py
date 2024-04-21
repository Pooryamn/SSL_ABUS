import torch

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