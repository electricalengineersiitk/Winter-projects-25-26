import torch
import torch.nn as nn
import torch.optim as optim

def train_and_eval(model, train_loader, test_loader, device, epochs=1, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img, lbp, canny, shape, color, targets in train_loader:
            img, lbp, canny = img.to(device), lbp.to(device), canny.to(device)
            shape, color, targets = shape.to(device), color.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(img, lbp, canny, shape, color)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'mask') and module.mask is not None:
                        module.weight.data.mul_(module.mask)
            
            running_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")
            
    return evaluate_only(model, test_loader, device)

def evaluate_only(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, lbp, canny, shape, color, targets in test_loader:
            img, lbp, canny = img.to(device), lbp.to(device), canny.to(device)
            shape, color, targets = shape.to(device), color.to(device), targets.to(device)
            
            outputs = model(img, lbp, canny, shape, color)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    acc = 100 * correct / total
    print(f"--- Evaluation Accuracy: {acc:.2f}% ---")
    return acc
