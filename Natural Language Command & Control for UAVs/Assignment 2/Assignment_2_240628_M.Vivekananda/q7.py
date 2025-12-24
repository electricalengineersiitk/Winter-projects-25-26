import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init(self):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(10, 5) 
        self.layer2 = nn.Linear(5, 1)  
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.sigmoid(x)
        return x