import torch

def precision_recall(pred, mask, smooth=1e-6):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * mask).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)

    return precision, recall