# MLP model definition
# this is the model that gets compressed - trained on extracted feature vectors not raw images
# architecture: Input(512) -> FC(256) -> ReLU -> Dropout -> FC(128) -> ReLU -> Dropout -> FC(64) -> ReLU -> Dropout -> FC(num_classes)
# similar concept to assignment 6 where we built an MLP from scratch for the donut classification
# but here we use nn.Module since we need proper layer support for the compression steps later

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    # multi-layer perceptron for classification
    # unlike assignment 6 where we did raw torch.matmul, here we use nn.Linear
    # because we need to swap layers later for compression (CompressedLinear)
    def __init__(self,input_dim=512,hidden_layers=None,
                 num_classes=10,dropout=0.3):
        super(MLP,self).__init__()

        if hidden_layers is None:
            hidden_layers=[256,128,64]

        layers=[]
        prev_dim=input_dim

        # build hidden layers one by one (like stacking layers in assignment 6 but using Sequential)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim,h_dim))
            layers.append(nn.ReLU()) # using relu like in assignment 6
            layers.append(nn.Dropout(dropout)) # dropout to prevent overfitting
            prev_dim=h_dim

        # output layer - no activation here because CrossEntropyLoss includes softmax
        layers.append(nn.Linear(prev_dim,num_classes))

        self.network=nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)


def create_mlp(input_dim=512,hidden_layers=None,
               num_classes=10,dropout=0.3):
    # factory function to create the MLP and print a summary
    model=MLP(input_dim,hidden_layers,num_classes,dropout)

    # count parameters (similar to counting weights in assignment 6)
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters()
                         if p.requires_grad)

    print("\n--- MLP Architecture ---")
    print(model)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# training utilities

def train_epoch(model,loader,criterion,optimizer,device):
    # train for one epoch, return avg loss and accuracy
    model.train()
    total_loss=0.0
    correct=0
    total=0

    for features,labels in loader:
        features,labels=features.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(features) # forward pass
        loss=criterion(outputs,labels) # compute loss (cross entropy)
        loss.backward() # backprop - figures out gradients

        # FIX: Zero out gradients of pruned weights before optimizer step.
        # Without this, Adam's momentum and weight decay will revive pruned weights.
        for module in model.modules():
            if hasattr(module, 'mask') and hasattr(module, 'weight'):
                if module.weight.grad is not None:
                    module.weight.grad *= module.mask

        optimizer.step() # update weights

        total_loss+=loss.item()*features.size(0)
        _,predicted=outputs.max(1) # whichever class has highest score
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()

    avg_loss=total_loss/total
    accuracy=100.0*correct/total
    return avg_loss,accuracy


def evaluate(model,loader,criterion,device):
    # evaluate model on test/val set, returns avg loss and accuracy
    model.eval()
    total_loss=0.0
    correct=0
    total=0

    with torch.no_grad(): # no need to track gradients during evaluation
        for features,labels in loader:
            features,labels=features.to(device),labels.to(device)
            outputs=model(features)
            loss=criterion(outputs,labels)

            total_loss+=loss.item()*features.size(0)
            _,predicted=outputs.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()

    avg_loss=total_loss/total
    accuracy=100.0*correct/total
    return avg_loss,accuracy


def train_model(model,train_loader,test_loader,epochs,lr,
                weight_decay=1e-4,device=None,verbose=True):
    # full training loop similar to assignment 6 but using Adam instead of manual SGD
    # we use Adam here because it converges faster and handles learning rate adaptation
    # (in assignment 6 we did manual SGD: p-=lr*p.grad, but Adam is better for this use case)
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=model.to(device)
    criterion=nn.CrossEntropyLoss() # for multi-class classification
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,
                               weight_decay=weight_decay) # weight_decay = L2 regularization

    history={
        'train_loss':[],'train_acc':[],
        'test_loss':[],'test_acc':[]
    }

    for epoch in range(1,epochs+1):
        train_loss,train_acc=train_epoch(model,train_loader,
                                         criterion,optimizer,device)
        test_loss,test_acc=evaluate(model,test_loader,
                                    criterion,device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # print every 5 epochs (like how we printed every 100 in assignment 6)
        if verbose and (epoch%5==0 or epoch==1):
            print(f"Epoch [{epoch:3d}/{epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
                  f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

    return history