import torch

def evaluate(model, test_loader, feature_extractor, device):
    """
    Evaluates the MLP on the extracted features of the test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Extract features first (frozen)
            features = feature_extractor(images)
            features = torch.flatten(features, 1)
            
            # Pass through MLP
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy