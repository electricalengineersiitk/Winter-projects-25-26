## cifar-10 data loading with standard augmentation
## downloads the dataset if needed, applies random flips and crops for training, normalises everything

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


# precomputed mean and std for each RGB channel of cifar-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_transforms(): # train gets augmentation (random crop + flip), test just gets normalised
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_data_loaders(batch_size=None, num_workers=2): # returns train and test loaders ready to feed into the model
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_transform, test_transform = get_transforms()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Data] CIFAR-10 loaded  |  Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")
    print(f"[Data] Batch size: {batch_size}  |  Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"[Data] Sample batch shape: {images.shape}, Labels: {labels[:10]}")
