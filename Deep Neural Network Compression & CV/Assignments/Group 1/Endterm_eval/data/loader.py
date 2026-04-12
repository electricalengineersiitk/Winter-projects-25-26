# data loading and feature extraction
# we dont train a CNN from scratch, instead we use a pre-trained ResNet-18 as a feature extractor
# it gives us 512-dim feature vectors for each image which then go into our MLP
# this way we can focus on compressing the MLP without worrying about conv layers

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms,models

def load_dataset(dataset_name='cifar10', data_dir='./data_cache', image_size=224): # <-- Add image_size here
    
    transform=transforms.Compose([
        transforms.Resize((image_size, image_size)), # <-- Update this line
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    transform_gray=transforms.Compose([
        transforms.Resize((image_size, image_size)), # <-- Update this line too
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    if dataset_name=='cifar10':
        train_data=datasets.CIFAR10(root=data_dir,train=True,
                                    download=True,transform=transform)
        test_data=datasets.CIFAR10(root=data_dir,train=False,
                                   download=True,transform=transform)
        num_classes=10
    elif dataset_name=='cifar100':
        train_data=datasets.CIFAR100(root=data_dir,train=True,
                                     download=True,transform=transform)
        test_data=datasets.CIFAR100(root=data_dir,train=False,
                                    download=True,transform=transform)
        num_classes=100
    elif dataset_name=='mnist':
        train_data=datasets.MNIST(root=data_dir,train=True,
                                  download=True,transform=transform_gray)
        test_data=datasets.MNIST(root=data_dir,train=False,
                                 download=True,transform=transform_gray)
        num_classes=10
    elif dataset_name=='fashion_mnist':
        train_data=datasets.FashionMNIST(root=data_dir,train=True,
                                         download=True,transform=transform_gray)
        test_data=datasets.FashionMNIST(root=data_dir,train=False,
                                        download=True,transform=transform_gray)
        num_classes=10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Dataset: {dataset_name}")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples:     {len(test_data)}")
    print(f"  Number of classes: {num_classes}")

    return train_data,test_data,num_classes


def extract_features(dataset,batch_size=64,device=None):
    # extract features using pre-trained ResNet-18
    # we chop off the final classification layer and use the 512-dim avgpool output
    # no training happens here - the backbone is completely frozen
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device=torch.device(device)

    # load pretrained resnet18 and strip off the final FC layer
    resnet=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor=nn.Sequential(*list(resnet.children())[:-1]) # everything except last FC
    feature_extractor=feature_extractor.to(device)
    feature_extractor.eval() # freeze - we dont want to update these weights

    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,
                      num_workers=2,pin_memory=True)

    all_features=[]
    all_labels=[]

    print("Extracting features...")
    with torch.no_grad(): # no gradients needed since we're not training
        for i,(images,labels) in enumerate(loader):
            images=images.to(device)
            feats=feature_extractor(images) # forward pass through frozen resnet
            feats=feats.squeeze(-1).squeeze(-1) # reshape from (batch,512,1,1) to (batch,512)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

            if (i+1)%50==0:
                print(f"  Processed {(i+1)*batch_size}/{len(dataset)} samples")

    features=np.concatenate(all_features,axis=0)
    labels=np.concatenate(all_labels,axis=0)

    print(f"Feature extraction complete.")
    print(f"  Feature shape: {features.shape}")
    print(f"  Labels shape:  {labels.shape}")

    return features,labels


def prepare_dataloaders(train_features,train_labels,
                        test_features,test_labels,
                        batch_size=64):
    # wrap numpy arrays into pytorch DataLoaders for training
    train_X=torch.tensor(train_features,dtype=torch.float32)
    train_y=torch.tensor(train_labels,dtype=torch.long)
    test_X=torch.tensor(test_features,dtype=torch.float32)
    test_y=torch.tensor(test_labels,dtype=torch.long)

    train_dataset=TensorDataset(train_X,train_y)
    test_dataset=TensorDataset(test_X,test_y)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,
                            shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,
                           shuffle=False)

    return train_loader,test_loader
