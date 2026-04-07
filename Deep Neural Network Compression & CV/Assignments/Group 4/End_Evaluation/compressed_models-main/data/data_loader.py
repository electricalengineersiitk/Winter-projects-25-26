"""
CIFAR-10 Data Loader with AlexNet Preprocessing

Implements data augmentation and preprocessing pipeline for training AlexNet on CIFAR-10.
- Resize images from 32x32 to 224x224 (AlexNet input size)
- Apply data augmentation (random crop, horizontal flip)
- Normalize with CIFAR-10 statistics
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

def get_train_transforms():
    """Training transforms with data augmentation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR_MEAN, std=config.CIFAR_STD)
    ])

def get_test_transforms():
    """Test transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR_MEAN, std=config.CIFAR_STD)
    ])

def CIFAR10_loader(path=None, batch_size=None):
    """
    Create CIFAR-10 data loaders for training and testing.
    
    Args:
        path: Path to download/load CIFAR-10 dataset
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    if path is None:
        path = config.DATA_PATH
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_dataset = datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=get_train_transforms()
    )
    
    test_dataset = datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=get_test_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader