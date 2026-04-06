import os
import time
import logging
import torch
import torch.nn as nn
from torch import optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from utils.data_loading import CarvanaDataset
from utils.dice_score import dice_loss, dice_coeff
from utils.precision_recall import precision_recall
from utils.iou_score import iou_score
from unet import UNet
from evaluate import evaluate
from utils.utils import train_and_val_loss, train_and_val_diceloss,visualize_batch

torch.set_num_threads(4)

# ================= LOGGING =================
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

rootpath = "C:/Users/padma-inc5710/task//task2/data"

# ================= TRAIN FUNCTION =================
def train_model(
        model,
        optimizer,
        device,
        epochs: int = 5,
        batch_size: int =4,
        start_epoch: int = 1
):
    model.to(device)

    #create dataset
    dataset = CarvanaDataset(rootpath)

    #split into train/test/val
    indices = list(range(len(dataset)))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size= 0.2,
        random_state = 42
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=42
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)


    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    print("\n========== DATASET SPLIT ==========")
    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")
    print("===================================\n")

    num_workers = os.cpu_count() // 2  
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set,batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test_set,batch_size=batch_size, shuffle = False)

    #set up loss function
    criterion = nn.BCEWithLogitsLoss()

    print("========== TRAINING START ==========\n")

    #begin training
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_dcs, val_dcs = [], []

    for epoch in range(start_epoch, epochs+1):

        model.train()

        train_loss, train_dc = 0,0

        for idx, (img,mask) in enumerate(tqdm(train_loader, desc=f'Epoch{epoch}/{epochs} - Training')):
            img = img.float().to(device)
            mask = (mask>0).float().to(device)

            y_pred = model(img)
            
            dc = dice_loss(torch.sigmoid(y_pred), mask)
            bce = criterion(y_pred, mask)
            loss = dc + bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dc += dice_loss(y_pred, mask).item()

        train_loss /= (idx + 1)
        train_dc /= (idx + 1)

        #Validation
        model.eval()

        val_loss, val_dc, val_iou = 0, 0, 0
        val_precision, val_recall = 0, 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_loader , desc=f"Epoch {epoch}/{epochs} - Validation")):
                img = img.float().to(device)
                mask = (mask > 0).float().to(device)

                y_pred = model(img)

                bce = criterion(y_pred, mask)
                dc = dice_loss(torch.sigmoid(y_pred), mask)
                loss = bce + dc

                pred = (torch.sigmoid(y_pred) > 0.5).float()

                val_loss += loss.item()
                val_dc += dice_loss(y_pred, mask).item()
                val_iou += iou_score(pred, mask).item()

                p, r = precision_recall(pred, mask)
                val_precision += p.item()
                val_recall += r.item()
                if idx==0:
                    visualize_batch(img, mask, pred, epoch)


            val_loss /= (idx + 1)
            val_dc /= (idx + 1)
            val_iou /= (idx + 1)
            val_precision /= (idx + 1)
            val_recall /= (idx + 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dcs.append(train_dc)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dc:.4f}, IoU: {val_iou:.4f}")
        print("-" * 30)

        logger.info(
            f"Epoch {epoch} | Train Loss {train_loss:.4f}, Dice {train_dc:.4f} | "
            f"Val Loss {val_loss:.4f}, Dice {val_dc:.4f}, IoU {val_iou:.4f}, "
            f"Precision {val_precision:.4f}, Recall {val_recall:.4f}"
        )

        # ========= CHECKPOINT (FIXED) =========
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, 'last_checkpoint.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"Best model saved at epoch {epoch}")


    print("\n========== TEST EVALUATION ==========")

    test_loss, test_dc, test_iou, test_precision, test_recall = evaluate(model = model,
                                  test_dataloader=test_loader,
                                  criterion = criterion,
                                  device =device)
    
    print("=" * 40)
    print(f'Test Loss      : {test_loss:.4f}')
    print(f'Test Dice      : {test_dc:.4f}')
    print(f'Test IoU       : {test_iou:.4f}')
    print(f'Test Precision : {test_precision:.4f}')
    print(f'Test Recall    : {test_recall:.4f}')
    logger.info(
        f"TEST | Loss {test_loss:.4f}, Dice {test_dc:.4f}, "
        f"IoU {test_iou:.4f}, Precision {test_precision:.4f}, Recall {test_recall:.4f}"
    )
    print("=" * 40)

    train_and_val_loss(range(1, epochs+1), train_losses, val_losses)
    train_and_val_diceloss(range(1, epochs+1), train_dcs, val_dcs)

    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_channels=3, n_classes=1).to(device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-8,
        momentum=0.9,
    )
    
    start_epoch = 1

    if os.path.exists('last_checkpoint.pth'):
        print('Loading checkpoint...')
        checkpoint = torch.load('last_checkpoint.pth', map_location = device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')

    start_time = time.time()
    print('start time:',start_time)

    train_model(model,optimizer,device=device, start_epoch = start_epoch)

    end_time = time.time()- start_time

    print("\n========== EXECUTION TIME ==========")
    print(f"Total seconds : {end_time:.2f}")
    print(f"Total minutes : {(end_time/60):.2f}")
    print("====================================")
    