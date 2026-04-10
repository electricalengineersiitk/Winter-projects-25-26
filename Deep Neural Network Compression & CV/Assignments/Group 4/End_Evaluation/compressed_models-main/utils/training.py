import torch
from torch import nn
import torchvision.models as models
from .test_eval import evaluate

def train_and_eval(model, train_loader, test_loader, device, epochs=5):
    """
    Trains the MLP model on features extracted by a frozen CNN.
    """
    print("Loading frozen feature extractor (AlexNet)...")
    # Load AlexNet features and freeze them
    feature_extractor = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Extract Features (NO GRADIENTS)
            with torch.no_grad():
                features = feature_extractor(images)
                features = torch.flatten(features, 1) # Shape: [batch_size, 9216]
                
            # 2. Train MLP
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {running_loss/len(train_loader):.4f}")
        
    # Evaluate at the end of training
    return evaluate(model, test_loader, feature_extractor, device)

def train_model(*args, **kwargs):
    # Fallback alias if main.py explicitly calls train_model instead of train_and_eval
    return train_and_eval(*args, **kwargs)