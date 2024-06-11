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

def Classification_results(inputs, targets, smooth=0.001):

        batches = inputs.shape[0]

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # TP, FP, FN
        TP = (inputs * targets).sum()
        TN = ((1 - targets) * (1 - inputs)).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Precision = TP / (TP + FP + smooth)
        Recall = TP / (TP + FN + smooth)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall + smooth)

        return Precision, Recall, F1, Accuracy, FP / batches

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def Detection_results(predictions, targets, iou_threshold, smooth=0.001):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for B in range(targets.shape[0]):
        for i in range(targets.shape[2]):
            if(predictions[B,i,0] == 1 and targets[B,0,i,0] == 1):
                # chech IOU
                IOU = calculate_iou(predictions[B, i, 1:], targets[B,0, i, 1:])
                if(IOU >= iou_threshold):
                    TP += 1
            elif(predictions[B,i,0] == 0 and targets[B,0,i,0] == 1):
                FN += 1
            elif(predictions[B,i,0] == 1 and targets[B,0,i,0] == 0):
                FP += 1
            else:
                TN += 1

    TP = TP / (targets.shape[0] * targets.shape[2])
    TN = TN / (targets.shape[0] * targets.shape[2])
    FP = FP / (targets.shape[0] * targets.shape[2])
    FN = FN / (targets.shape[0] * targets.shape[2])

    Precision = TP / (TP + FP + smooth)
    Recall = TP / (TP + FN + smooth)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall + smooth)

    return Precision, Recall, F1, Accuracy, FP