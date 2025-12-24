import torch 
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier,self).__init()
        self.layer1=nn.Linear(10,5)
        self.layer2=nn.Linear(5,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.sigmoid(x)
        return x
    
def train_step(model,inputs,targets,optimizer,criterion):
    predictions=model(inputs)
    loss=criterion(predictions,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


model= SimpleClassifier()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
criterion=nn.BCELoss()

input_1=torch.rand(1,10)
target_1=torch.tensor([[1.0]])
loss_val=train_step(model,input_1,target_1,optimizer,criterion)
print(f"Current step loss : {loss_val}")