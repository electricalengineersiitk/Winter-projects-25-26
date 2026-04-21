#TRAINING MODEL

import numpy as np  #Importing required libraries
import torch
import torch.nn as nn
import copy
import os

DATA_PATH = "tasks.npz" #Dataset which was created earlier

META_LR = 0.1  #Defining parameters
INNER_LR = 0.01
INNER_STEPS = 5
EPOCHS = 50

os.makedirs("results", exist_ok=True) #Creating folder to store results


# MODEL

class Net(nn.Module): #Defining neural network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x): #Forward pass
        return self.net(x).squeeze()


data = np.load(DATA_PATH, allow_pickle=True) #Loading data
train_tasks = data["train"]

model = Net()
loss_fn = nn.MSELoss() #Mean squared Error

loss_history = [] #Stores loss per epoch


# TRAINING LOOP (Using REPTILE model)

for epoch in range(EPOCHS):
    total_loss = 0 #Tracking loss of each epoch

    for task in train_tasks:
        Xs = torch.tensor(task["X_support"], dtype=torch.float32)
        Ys = torch.tensor(task["Y_support"], dtype=torch.float32)

        temp_model = copy.deepcopy(model) #Getting copy of model for each task

        # inner loop
        for _ in range(INNER_STEPS):
            preds = temp_model(Xs)
            loss = loss_fn(preds, Ys) #Loss is difference between predicted and actual outputs

            temp_model.zero_grad() #Computing gradients
            loss.backward()

            for p in temp_model.parameters():
                p.data -= INNER_LR * p.grad #Updating temp model parameters

        total_loss += loss.item() #Tracking training loss

        # reptile update
        for p, tp in zip(model.parameters(), temp_model.parameters()):
            p.data += META_LR * (tp.data - p.data)

    avg_loss = total_loss / len(train_tasks)
    loss_history.append(avg_loss) #Computing avg loss

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

#saving results
torch.save(model.state_dict(), "reptile_model.pth")
np.save("loss_history/loss.npy", loss_history)

