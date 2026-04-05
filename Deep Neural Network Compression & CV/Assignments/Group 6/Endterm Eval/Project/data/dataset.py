import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
def build_mnist_dataloaders(batch_size:int):
    transform = transforms.Compose([
        transforms.ToTensor(),   
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader