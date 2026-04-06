import torch
from torch import Tensor

def dice_coeff(prediction, target, epsilon= 1e-7):
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float()

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

def dice_loss(prediction, target):
    return 1 - dice_coeff(prediction, target)