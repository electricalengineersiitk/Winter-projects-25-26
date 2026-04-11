# deep_compression/
# │
# ├── main.py                # Entry point (wires everything together)
# │
# ├── config.py              # Hyperparameters & settings (optional but recommended)
# │
# ├── data/
# │   ├── __init__.py
# │   └── cifar.py           # Dataset + dataloaders only
# │
# ├── models/
# │   ├── __init__.py
# │   └── alexnet.py         # Model architecture only
# │
# ├── compression/
# │   ├── __init__.py
# │   ├── layers.py          # modified_linear, modified_conv2d
# │   ├── prune.py           # pruning logic
# │   ├── quantize.py        # quantization logic
# │   └── huffman.py         # huffman coding
# │
# ├── training/
# │   ├── __init__.py
# │   ├── train.py           # training loop
# │   └── eval.py            # evaluation logic
# │
# ├── utils/
# │   ├── __init__.py
# │   └── device.py          # device helpers, seeds, logging
# │
# └── requirements.txt