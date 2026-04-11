import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def MNIST_loader(path, batch_size=64):
    """
    Downloads and loads the MNIST dataset.
    Returns the train_loader and test_loader.
    """
    # 1. Define how we want to alter the raw images before using them
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts images to PyTorch Tensors (0-255 pixels become 0.0-1.0)
        transforms.Normalize((0.1307,), (0.3081,)) # Standard mathematical normalization for MNIST
    ])

    # 2. Download/Load the Training Set
    train_dataset = datasets.MNIST(
        root=path, 
        train=True, 
        download=True, # Will download into the 'data' folder if it's not there yet
        transform=transform
    )
    
    # 3. Download/Load the Testing Set
    test_dataset = datasets.MNIST(
        root=path, 
        train=False, 
        download=True, 
        transform=transform
    )

    # 4. Wrap the datasets in DataLoaders
    # shuffle=True for training prevents the model from learning the order of the images
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # We don't need to shuffle the test set
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded MNIST: {len(train_dataset)} training images, {len(test_dataset)} testing images.")
    
    return train_loader, test_loader