import torch
from torch import nn, optim

def train_model(model, train_loader, epochs, learning_rate, device):
    
    model.to(device)
    model.train() 

   
    criterion = nn.CrossEntropyLoss()
    
  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting Training for {epochs} Epochs")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
           
            images, labels = images.to(device), labels.to(device)

            
            optimizer.zero_grad()

            
            outputs = model(images)
            loss = criterion(outputs, labels)

            
            loss.backward()

            
            optimizer.step()

            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

def evaluate(model, test_loader, device) -> float:
    
    model.to(device)
    model.eval() 
    
    correct = 0
    total = 0
    

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100. * correct / total
    print(f"==> Test Accuracy: {accuracy:.2f}%")
    
    return accuracy