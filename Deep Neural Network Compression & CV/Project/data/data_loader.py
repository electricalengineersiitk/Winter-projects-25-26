from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def MNIST_loader(path):
    training_data = datasets.MNIST(path + '/MNIST/',
                                 download=True,
                                 train=True,transform=ToTensor())
    test_data = datasets.MNIST(path + '/MNIST/',
                                    download=True,
                                    train=False,transform=ToTensor())
    train_dataloader = DataLoader(training_data , batch_size = 32 , shuffle = True)
    test_dataloader = DataLoader(test_data , batch_size = 32 , shuffle = True)

    return train_dataloader,test_dataloader


def CIFAR10_loader(path):
    training_data = datasets.MNIST(path + '/CIFAR10/',
                                 download=True,
                                 train=True,transform=ToTensor())
    test_data = datasets.MNIST(path + '/CIFAR10/',
                                    download=True,
                                    train=False,transform=ToTensor())
    train_dataloader = DataLoader(training_data , batch_size = 32 , shuffle = True)
    test_dataloader = DataLoader(test_data , batch_size = 32 , shuffle = True)

    return train_dataloader,test_dataloader
