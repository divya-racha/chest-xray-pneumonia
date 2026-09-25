"""Dataset utilities for the chest X-ray pneumonia classifier.

Expected layout (Kaggle `paultimothymooney/chest-xray-pneumonia` format):

    data/chest_xray/
        train/NORMAL/*.jpeg
        train/PNEUMONIA/*.jpeg
        val/NORMAL/*.jpeg
        val/PNEUMONIA/*.jpeg
        test/NORMAL/*.jpeg
        test/PNEUMONIA/*.jpeg
"""
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            # Horizontal flips are fine; vertical flips are NOT anatomically
            # meaningful for chest X-rays.
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples = []
        self.transform = transform or get_transforms(train=(split == "train"))
        split_dir = os.path.join(root, split)
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(split_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {split_dir}. "
                "See data/README.md for download instructions."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
