import torch
import torchvision
import torchvision.transforms as transforms
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

def get_dataloaders():
    print(f"Preparing datasets in: {DATA_DIR}")
    
    # 1. Define transformations for the training data
    # We use random cropping and flipping to make the model more robust
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Standard normalization values for the CIFAR-10 dataset
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2. Define transformations for the testing data
    # No random augmentations here, just formatting
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Download and load the datasets
    # We wrap DATA_DIR in str() because your config uses a pathlib Path object
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return trainloader, testloader

# Quick test block to ensure data flows correctly
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\n✅ Data Loader working perfectly!")
    print(f"Batch image shape: {images.shape} (Batch Size, Channels, Height, Width)")
    print(f"Batch label shape: {labels.shape}")