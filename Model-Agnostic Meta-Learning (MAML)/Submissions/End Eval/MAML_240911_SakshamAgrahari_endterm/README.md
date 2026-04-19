
# Meta-Learning for Wireless Channel Estimation

## Part 1 — What did you build?

In this project, I implemented a meta-learning model using the Reptile algorithm for wireless channel estimation. The model is trained across multiple simulated wireless environments so that it can quickly adapt to a new unseen environment using only a few samples. Instead of training from scratch every time, the model learns a good initialization that enables fast adaptation.

## Part 2 — How to set it up

pip install -r requirements.txt

This command installs all the required libraries such as numpy, torch, and matplotlib which are used for data generation, model training, and plotting.

## Part 3 — How to generate data

python generate_data.py

This script creates synthetic wireless channel estimation tasks. It generates 1000 training tasks and 100 test tasks, where each task represents a different environment. The data is stored in .npz format inside the data folder, and each task contains a support set (15 samples) and a query set (50 samples).

## Part 4 — How to train

python train.py

This script trains the meta-learning model using the Reptile algorithm across multiple tasks. The inner learning rate is 0.01 and the outer step size is 0.5. The model is trained on 1000 tasks, and each task uses 15 support samples (shots) for adaptation. During training, the model learns a good initialization so that it can adapt quickly to new tasks.

## Part 5 — How to test

python test.py

This script evaluates the trained meta-learning model on new unseen tasks. It adapts the model using the support set with 5 gradient steps and then evaluates it on the query set. A baseline model is also trained from scratch using multiple steps for comparison. The script prints the average error of the meta-learning model and the average error of the baseline model, allowing direct comparison of their performance.

## Part 6 —  results

| Method                      | Error |
|---------------------------|-------|
| Basic model (from scratch) | 2.10  |
| Meta-learning (Reptile)    | 1.00  |

The meta-learning model achieves lower error compared to the basic model. This shows that the model has learned to adapt quickly to new environments using only a few samples. This confirms that meta-learning improves performance over standard training from scratch.

## 

The training loss initially decreases and then stabilizes, which is expected in meta-learning due to the variation in tasks. The comparison plot clearly shows that the meta-learning model performs better than the baseline model. All plots are saved inside the results folder.