# data/data_loader.py — CIFAR-10 dataset and dataloader setup

import torchvision
import torchvision.transforms as transforms
import torch


def get_dataloaders(data_dir, batch_size_train, batch_size_test, num_workers):
    """
    Returns (trainloader, testloader) for CIFAR-10.
    Downloads dataset to data_dir if not already present.
    """

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train,
        shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test,
        shuffle=False, num_workers=num_workers
    )

    print(f'Dataset ready  |  Train: {len(trainset)}  |  Test: {len(testset)}')
    return trainloader, testloader
