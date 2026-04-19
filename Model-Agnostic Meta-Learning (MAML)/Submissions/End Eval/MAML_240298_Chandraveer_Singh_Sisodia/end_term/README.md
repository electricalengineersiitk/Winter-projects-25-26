
# Meta-Learning for Wireless Systems: Channel Estimation



### Part 1 — What did you build?

I built a 3-layer neural network designed to adapt quickly to new wireless environments using few-shot learning. I implemented both **MAML** and **Reptile** meta-learning algorithms to estimate the channel coefficient $H$ from limited pilot signals. The model learns a "shared initialization" that allows it to generalize efficiently across diverse fading conditions with minimal data.



### Part 2 — How to set it up

Use these commands to clone the repository and install dependencies:



```bash

git clone https://github.com/cssisodia24/Winter-projects-25-26.git
cd Winter-projects-25-26/end_term
pip install -r requirements.txt

```



### Part 3 — How to generate data

```bash

python generate_data.py

```

This script creates synthetic wireless tasks by simulating different SNR levels and channel gains using NumPy.It generates 100 training tasks and 20 test tasks, each containing a 10-sample support set for adaptation and a 50-sample query set for evaluation.



### Part 4 — How to train

```bash

python train.py

```

Model details:

3 layers with 64 neurons each
Optimizer: Adam (inner + outer loop)
Training iterations: 2500
Outer learning rate: 0.001
Inner learning rate: 0.001
Meta-batch size: 5
Inner adaptation steps: 5



### Part 5 — How to test

```bash

python test.py

```

This script:

Evaluates on 20 unseen tasks
Performs 5 adaptation steps per task
Outputs Average Mean Squared Error (MSE) comparison



### Part 6 — Your results

Comparison of the average Mean Squared Error (MSE) over all 20 test tasks (Lower error is better) :



| Method                     | Average Error (MSE) |
| -------------------------- | ------------------- |
| Basic model (from scratch) | 0.996030            |
| Reptile Model              | 1.563954            |
| MAML Model (Bonus)         | 0.656250            |
