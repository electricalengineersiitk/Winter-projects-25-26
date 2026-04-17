# End-Term Project: Channel Estimation with Reptile and MAML

## Part 1 - What Did You Build?
I built a few-shot channel estimation system using the Reptile meta-learning algorithm. For the bonus comparison, I also trained a MAML-style meta-learner and compared it with Reptile and a scratch baseline on unseen wireless environments. Each model adapts from a small support set of pilot/channel pairs and then predicts the channel vector on the query set.

## Part 2 - How to Set It Up
```bash
git clone https://github.com/anirudh3810/channel-estimation-reptile-maml.git
cd channel-estimation-reptile-maml/end_term
pip install -r requirements.txt
```

If you already cloned the repository, just move into the `end_term/` folder and install the requirements.

## Part 3 - How to Generate Data
```bash
python generate_data.py
```

This script creates a synthetic channel estimation dataset with 100 meta-training tasks and 20 unseen test tasks. Each task stores a support pool of 20 labelled pilot observations plus a query set of 100 samples, and everything is saved into `data/channel_estimation_tasks.npz` with train-set normalization statistics.

## Part 4 - How to Train
```bash
python train.py
```

Default training uses Reptile and MAML for 250 meta-iterations with a meta-batch size of 8 tasks, 5 inner adaptation steps, inner learning rate `0.01`, Reptile meta learning rate `0.12`, MAML meta learning rate `0.003`, and a 3-layer MLP with 64 hidden units per layer. The training script saves the learned initializations to `results/reptile_model.npz` and `results/maml_model.npz`, writes the training loss curve to `results/plot_loss.png`, and writes the bonus comparison plot to `results/plot_comparison.png`.

## Part 5 - How to Test
```bash
python test.py
```

The test script adapts the Reptile and MAML initializations for 5 gradient steps on each unseen task and prints the average query NMSE over all 20 test tasks. It also trains a baseline network from scratch for 200 steps on the same support set, prints all averages, and saves the MAML-vs-Reptile-vs-baseline comparison plot to `results/plot_comparison.png`.

## Part 6 - Results
The table below contains the values from the latest run in this workspace after `python generate_data.py`, `python train.py`, and `python test.py`.

| Method | 5-shot Error (NMSE dB) | 10-shot Error (NMSE dB) | 20-shot Error (NMSE dB) |
| --- | ---: | ---: | ---: |
| Baseline model (from scratch) | 0.89 | -1.14 | -2.54 |
| Reptile meta-learned model | -3.63 | -4.22 | -4.89 |
| MAML meta-learned model | -4.63 | -5.42 | -5.66 |

Lower NMSE is better. In this run, both meta-learned models outperformed the scratch baseline at every support size, and the bonus plot compares MAML and Reptile side by side.
