import torch

def iou_score(pred, mask, smooth=1e-6):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred *mask).sum()
    union = pred.sum() + mask.sum() - intersection

    iou = (intersection + smooth ) / (union +smooth)

    return iou