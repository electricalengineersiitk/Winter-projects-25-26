import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/test_data.npz")

maml = data["m_errs"]
baseline = data["b_errs"]

maml_avg = np.mean(maml)
baseline_avg = np.mean(baseline)

labels = ["Meta-learning", "Baseline"]
values = [maml_avg, baseline_avg]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Error")
plt.title("Meta-learning vs Baseline")

plt.savefig("results/plot_comparison.png")
plt.show()

print("plot_comparison.png saved!")