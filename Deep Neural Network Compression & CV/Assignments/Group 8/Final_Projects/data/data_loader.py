
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor(),])

        self.dataset = datasets.MNIST(
            root="./data/MNIST",
            train=train,
            download=True,
            transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Feature extraction Step
        feature = img.view(-1)
        return feature, label


def get_data_loaders(batch_size=128):
    train_dataset = MNISTFeatureDataset(train=True)
    test_dataset = MNISTFeatureDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader