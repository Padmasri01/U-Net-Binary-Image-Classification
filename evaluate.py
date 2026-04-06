
import torch
from tqdm import tqdm

from utils.dice_score import  dice_coeff, dice_loss
from utils.iou_score import iou_score
from utils.precision_recall import precision_recall

@torch.inference_mode()
def evaluate(model, test_dataloader, criterion, device):
    model.eval()

    test_running_loss = 0
    test_running_dc = 0
    test_running_iou = 0
    test_running_precision = 0
    test_running_recall = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = (mask > 0).float()

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_loss(y_pred, mask)

            pred = (torch.sigmoid(y_pred) > 0.5).float()

            iou = iou_score(pred, mask)
            p, r = precision_recall(pred, mask)

            test_running_loss += loss.item()
            test_running_dc += dc.item()
            test_running_iou       += iou.item()
            test_running_precision += p.item()
            test_running_recall    += r.item()


        test_loss = test_running_loss / (idx + 1)
        test_dc = test_running_dc / (idx + 1)
        test_iou       = test_running_iou       / (idx + 1)
        test_precision = test_running_precision / (idx + 1)
        test_recall    = test_running_recall    / (idx + 1)

        return test_loss, test_dc, test_iou, test_precision, test_recall
