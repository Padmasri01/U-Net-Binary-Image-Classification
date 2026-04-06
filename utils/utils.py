import os
import matplotlib.pyplot as plt


def train_and_val_loss(epochs_list, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_list, train_losses, label='Training Loss')
    ax.plot(epochs_list, val_losses,   label='Validation Loss')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig('loss_curve.png')
    plt.show()
    plt.close(fig)


def train_and_val_diceloss(epochs_list, train_dcs, val_dcs):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_list, train_dcs, label='Training Dice')
    ax.plot(epochs_list, val_dcs,   label='Validation Dice')
    ax.set_title('Dice Coefficient over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig('dice_curve.png')
    plt.show()
    plt.close(fig)


def visualize_batch(images, masks, preds, epoch, num_images=2):
    os.makedirs('outputs', exist_ok=True)

    images = images.cpu()
    masks  = masks.cpu()
    preds  = preds.cpu()

    for i in range(min(num_images, images.size(0))):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        axes[0].imshow(images[i].permute(1, 2, 0))
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(masks[i].squeeze(), cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(preds[i].squeeze(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        fig.tight_layout()
        fig.savefig(f'outputs/epoch_{epoch}_img_{i}.png')
        plt.close(fig)