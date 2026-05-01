import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

model = nn.Linear(4, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.4f}")

weights = model.weight.data.numpy() * 256
biases = model.bias.data.numpy() * 256

with open("weights.mem", "w") as fw, open("biases.mem", "w") as fb:
    for i in range(8):
        if i < 3:
            for w in weights[i]:
                hex_val = format(int(w) & 0xFFFF, '04x')
                fw.write(f"{hex_val}\n")

            hex_b = format(int(biases[i]) & 0xFFFF, '04x')
            fb.write(f"{hex_b}\n")
        else:
            for _ in range(4): fw.write("0000\n")
            fb.write("0000\n")

