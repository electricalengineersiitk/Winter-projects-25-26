# utils/training.py — Training and fine-tuning loops

import torch
import torch.nn as nn


def train_model(model, trainloader, optimizer, scheduler, loss_fn, num_epochs, device):
    """
    Full training loop with LR scheduling.
    Prints loss every 200 batches.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch: {epoch+1:3d}, Batch: {i+1:5d}, '
                      f'Loss: {running_loss/200:.3f}')
                running_loss = 0.0

        scheduler.step()

    print('Finished Training')
    return model


def fine_tune(model, trainloader, loss_fn, device, lr, momentum,
              weight_decay, fine_tune_epochs):
    """
    Fine-tunes a pruned model with a lower learning rate.
    Masks are kept fixed — only surviving weights update.
    """
    ft_optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay
    )

    print('Starting fine-tuning...')
    for epoch in range(fine_tune_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            ft_optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            ft_optimizer.step()
            running_loss += loss.item()

        print(f'  Epoch {epoch+1:2d}/{fine_tune_epochs}  '
              f'loss: {running_loss/len(trainloader):.4f}')

    print('Fine-tuning complete.')
    return model
