import os
import random
import torch
import matplotlib.pyplot as plt

from utils.data_loading import CarvanaDataset
from utils.dice_score import dice_coeff
from predict import load_model, predict_image

model_path = './my_checkpoint_5.pth'

def visualize_predictions(n=10):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device)

    dataset = CarvanaDataset("C:/Users/padma-inc5710/task/task2/data")

    for _ in range(n):

        idx = random.randint(0, len(dataset) - 1)

        img, mask = dataset[idx]
        mask = (mask > 0).float()

        pred_mask = predict_image(model, img, device)

        dice = dice_coeff(pred_mask, mask)

        print(
            f"Image: {os.path.basename(dataset.images[idx])} | Dice: {round(float(dice),5)}"
        )

        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Original")

        plt.subplot(132)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")

        plt.subplot(133)
        plt.imshow(mask.squeeze(), cmap="gray")
        plt.title("Ground Truth")

        plt.show()


if __name__ == "__main__":
    visualize_predictions(10)
