# Meta-Learning for Wireless Signal Processing

## Part 1 - What did you build?

I built a meta-learning system for wireless signal processing using the Reptile algorithm. The model learns to quickly adapt to new wireless channel conditions with minimal data, implementing few-shot learning for communication systems where the model must rapidly adjust to different channel environments.

## Part 2 - How to set it up

```bash
git clone https://github.com/Samu-c/Winter-projects-25-26
cd Winter-projects-25-26/Model-Agnostic Meta-Learning (MAML)/Submissions/End Eval/MAML_240535_SamruddhiKedari/end_term
pip install -r requirements.txt
```

## Part 3 - How to generate data

```bash
python generate_data.py
```

This script creates synthetic wireless communication tasks with varying channel conditions and signal-to-noise ratios (0-20 dB). It generates 100 training tasks and 20 test tasks, each containing support sets (10 samples for adaptation) and query sets (60 samples for evaluation), saved in NumPy format as `tasks.npz`.

## Part 4 - How to train

```bash
python train.py
```

Key settings:
- Meta learning rate: 0.1
- Inner learning rate: 0.01
- Number of training tasks: 100
- Number of shots: 5-shot learning (5 inner adaptation steps)
- Training epochs: 50
- Feature dimension: 10, Channel dimension: 10

## Part 5 - How to test

```bash
python test.py
```

The script prints average MSE errors for both the Reptile model and baseline model after 5 adaptation steps, comparing performance across different adaptation steps (1, 2, 3, 4, 5). It also generates two plots: training loss curve and Reptile vs baseline comparison, saved in the `results/` folder.

## Part 6 - Your results

| Method | Avg Error | 
|--------|-------------|
| Basic model (from scratch) | 8.9476 | 
| Your Reptile model | 5.4390 | 



Lower error indicates better performance for MSE. The Reptile model shows approximately **39% improvement** over the baseline model at the given adaptation setting, demonstrating the effectiveness of meta-learning for rapid adaptation in wireless channel estimation tasks.


## Additional Details

### Model Architecture
- Neural network with 2 hidden layers (10->40->1)
- ReLU activation
- MSE loss function

### Task Distribution
- SNR range: 0-20 dB (uniformly distributed)
- Channel vectors: Random Gaussian distribution
- Signal vectors: 10-dimensional features

### Outputs
- `tasks.npz`: Generated dataset
- `reptile_model.pth`: Trained model weights
- `loss_history/loss.npy`: Training loss progression
- `results/plot_loss.png`: Loss curve visualization
- `results/plot_comparison.png`: Model comparison chart
