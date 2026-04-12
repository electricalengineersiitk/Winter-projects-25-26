# Meta-Learning for Wireless Systems: Channel Estimation

Part 1 — What did you build?
I built a neural network that leverages Model-Agnostic Meta-Learning (MAML) for the task of wireless Channel Estimation. Instead of training from scratch, the model is meta-trained across varying wireless environments so it can rapidly adapt to new, unseen environments (predicting the true channel from pilot signals) using only a few labeled examples.



Part 2 — How to set it up
To set up this project locally, run the following commands:

```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo/end_term
pip install -r requirements.txt
```


Part 3 — How to generate data

```bash
python generate_data.py
```
This script generates synthetic wireless channel estimation tasks using NumPy. It creates 100 training tasks and 20 testing tasks by randomly varying the Signal-to-Noise Ratio (SNR) and the number of multipath components. The formatted support and query sets are saved locally into a single wireless_dataset.npz file.


Part 4 — How to train
```bash
python train.py
```
This script runs the MAML meta-training loop for 500 iterations. It uses a meta-batch size of 4, an inner learning rate of 0.01, and 5 adaptation shots (inner steps). It also trains a baseline model from scratch for 200 steps on the test tasks. The trained MAML weights are saved to meta_model.pth.


| Parameter | Value |
| :--- | :--- |
| **Network Architecture** | 3 layers × 64 neurons |
| **Outer Optimizer (Meta)** | Adam (lr = 0.001) |
| **Inner Optimizer (Task)** | SGD (lr = 0.01) |
| **Inner Loop Steps** | 5 steps |
| **Support Shots** | 10 shots |
| **Tasks per Meta-Batch** | 4 tasks |
| **Meta-Iterations** | 500 iterations |



Part 5 — How to test
```bash
python test.py
```
This script evaluates the trained MAML and Reptile models against the baseline on the 20 unseen test tasks. It prints the average 5-shot and 20-shot mean squared error (converted to dB) to the console, and generates a visual comparison curve saved as results/plot_comparison.png.


Part 6 — The table below compares the performance of MAML and Reptile against a basic model trained from scratch on unseen test tasks. (Note: Lower dB indicates better performance/lower error).


| Method | 5-shot Error | 20-shot Error |
| :--- | :--- | :--- |
| **Basic model (scratch)** | -11.55 dB | -17.05 dB |
| **MAML** | **-14.56 dB** | **-16.92 dB** |
| **Reptile** | -12.63 dB | -14.48 dB |


Conclusions:
As shown by the 5-shot error, both meta-learning methods (MAML and Reptile) adapt faster to new wireless environments with very limited data compared to a network initialized from random weights. MAML shows the strongest early adaptation advantage, beating the baseline by a full 3 dB at the 5-shot mark!

