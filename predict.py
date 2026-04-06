import torch
import torchvision.transforms as transforms
from unet import UNet
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred_mask = model(img)

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    pred_mask = pred_mask.squeeze().cpu()

    return pred_mask

def save_prediction(image, pred_mask, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(pred_mask.numpy(), cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction to {save_path}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = 'C:\\Users\\padma-inc5710\\task\\task2\\best_model.pth'
    model = load_model(model_path, device)

   
    test_dir  = 'C:\\Users\\padma-inc5710\\task\\task2\\data'   # folder with input images
   
    image_files = []

    for root, dirs, files in os.walk(test_dir):
        if 'masks' in root:
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))

    # 🔥 random 10
    num_images = min(10, len(image_files))
    image_files = random.sample(image_files, num_images)
    save_dir  = 'C:/Users/padma-inc5710/task/task2/predictions'        # folder to save results
    os.makedirs(save_dir, exist_ok=True)

    # supported image extensions
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

    # image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_ext)]

    if len(image_files) == 0:
        print("No images found in test directory.")
    else:
        print(f"Found {len(image_files)} images. Running predictions...\n")

    for image_path in image_files:
        # image_path = os.path.join(test_dir, filename)
        # load image
        image = Image.open(image_path).convert('RGB')

        # predict
        pred_mask = predict_image(model, image, device)

         # extract only file name
        base_name = os.path.basename(image_path)

        # save side-by-side result
        save_path = os.path.join(save_dir, f"pred_{base_name}")
        save_prediction(image, pred_mask, save_path)

    print(f"\nAll predictions saved to: {save_dir}")

       

