import numpy as np
import matplotlib.pyplot as plt

loss = np.load("results/train_loss.npy")
plt.figure()
plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.savefig("results/plot_loss.png")
plt.show()

print(" plot_loss.png saved!")