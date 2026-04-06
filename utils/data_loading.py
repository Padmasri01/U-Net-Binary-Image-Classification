import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import InterpolationMode

class CarvanaDataset(Dataset):
    def __init__(self, root_path, limit = None):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([root_path + "/imgs/" + i for i in os.listdir(root_path + "/imgs/")])[:self.limit]
        self.masks = sorted([root_path + "/masks/" + i for i in os.listdir(root_path + "/masks/")])[:self.limit]

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation= InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256,256), interpolation = InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return img, mask
    
    def __len__(self):
        return min(len(self.images), self.limit)