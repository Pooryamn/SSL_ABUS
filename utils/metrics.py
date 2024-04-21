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

    # Calculate Intersection and Union
    # intersection = (predictions * targets).sum()
    # union = predictions.sum() + targets.sum()

    # # Dice score formula
    # dice = (2 * intersection + smooth) / (union + smooth)

    predictions = np.array([1 if x > 0.5 else 0.0 for x in predictions])
    targets     = np.array([1 if x > 0.5 else 0.0 for x in targets])

    dice = np.sum(predictions[targets == 1.0]) * 2.0 / (np.sum(predictions) + np.sum(targets))
    
    return dice