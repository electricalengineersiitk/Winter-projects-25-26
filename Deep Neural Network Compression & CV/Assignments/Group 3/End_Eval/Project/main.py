## deep compression pipeline — end-to-end execution
## trains a vgg model on cifar-10, then runs all 3 compression stages:
## 1. prune 90% of weights by magnitude, then fine-tune
## 2. quantize remaining weights via k-means clustering, then fine-tune centroids
## 3. apply huffman coding (offline, lossless)
## finally reports compression ratio, accuracy, and per-layer stats

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import config
from data.data_loader import get_data_loaders
from models.vgg_cifar import vgg_cifar10
from compression.pruning import prune_model, finetune_pruned_model, apply_pruning_masks
from compression.quantization import quantize_model, finetune_centroids, quantize_and_finetune
from compression.huffman import huffman_encode_model
from utils.metrics import (
    evaluate_accuracy, count_parameters, compute_sparsity,
    model_size_bytes, compressed_size_bytes, print_compression_summary
)


def set_seed(seed): # locks down all random number generators for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(): # picks gpu if available, otherwise cpu
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] Using CPU")
    return device


def train_baseline(model, train_loader, test_loader, device,
                   epochs=None, lr=None): # trains the vgg model from scratch on cifar-10
    if epochs is None:
        epochs = config.NUM_EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE

    print(f"\n{'='*60}")
    print(f"  PHASE 0: TRAINING BASELINE MODEL")
    print(f"{'='*60}")
    print(f"  Architecture: VGG-11 for CIFAR-10")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {config.BATCH_SIZE}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_SCHEDULE_MILESTONES,
        gamma=config.LR_GAMMA
    )

    best_acc = 0.0
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % config.PRINT_EVERY == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}]  "
                      f"Loss: {running_loss/(batch_idx+1):.4f}  "
                      f"Acc: {100.*correct/total:.2f}%")

        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  "
              f"Train: {train_acc:.2f}%  Test: {test_acc:.2f}%  LR: {current_lr}")

        # save best model checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(config.COMPRESSED_DIR, "baseline_best.pth")
            os.makedirs(config.COMPRESSED_DIR, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  * New best: {best_acc:.2f}% -> saved to {save_path}")

    print(f"\n[Baseline] Training complete. Best accuracy: {best_acc:.2f}%")
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description="Deep Compression Pipeline")
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip baseline training, load saved model')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--prune-sparsity', type=float, default=None,
                        help='Override pruning sparsity (e.g. 0.90)')
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()

    print(f"\n{'#'*60}")
    print(f"#  DEEP COMPRESSION PIPELINE")
    print(f"#  Paper: Han et al., ICLR 2016")
    print(f"#  Dataset: {config.DATASET}  |  Seed: {config.SEED}")
    print(f"{'#'*60}")

    train_loader, test_loader = get_data_loaders()

    model = vgg_cifar10(num_classes=config.NUM_CLASSES).to(device)
    original_size = model_size_bytes(model)
    total_params, _, _ = count_parameters(model)
    print(f"\n[Model] VGG-CIFAR  |  Parameters: {total_params:,}  |  "
          f"Size: {original_size/1024/1024:.2f} MB")

    # train or load baseline
    baseline_path = os.path.join(config.COMPRESSED_DIR, "baseline_best.pth")

    if args.skip_training and os.path.exists(baseline_path):
        print(f"\n[Baseline] Loading pre-trained model from {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=device))
    else:
        epochs = args.epochs if args.epochs else config.NUM_EPOCHS
        model, _ = train_baseline(model, train_loader, test_loader, device,
                                  epochs=epochs)
        if os.path.exists(baseline_path):
            model.load_state_dict(torch.load(baseline_path, map_location=device))

    original_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\n[Baseline] Accuracy before compression: {original_acc:.2f}%")

    original_size = model_size_bytes(model)

    # stage 1: pruning
    start = time.time()

    sparsity = args.prune_sparsity if args.prune_sparsity else config.PRUNE_SPARSITY
    model, masks = prune_model(model, sparsity=sparsity)

    pruned_acc_before = evaluate_accuracy(model, test_loader, device)
    print(f"\n[Stage 1] Accuracy after pruning (before fine-tune): {pruned_acc_before:.2f}%")
    print(f"[Stage 1] Accuracy drop: {original_acc - pruned_acc_before:.2f}%")

    model = finetune_pruned_model(
        model, masks, train_loader, test_loader, device,
        evaluate_fn=evaluate_accuracy
    )

    pruned_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\n[Stage 1] Accuracy after pruning + fine-tune: {pruned_acc:.2f}%")
    print(f"[Stage 1] Accuracy drop from baseline: {original_acc - pruned_acc:.2f}%")
    print(f"[Stage 1] Sparsity: {compute_sparsity(model)*100:.1f}%")
    print(f"[Stage 1] Time: {time.time()-start:.1f}s")

    pruned_path = os.path.join(config.COMPRESSED_DIR, "pruned_model.pth")
    torch.save({'state_dict': model.state_dict(), 'masks': masks}, pruned_path)

    # stage 2: quantization
    start = time.time()

    model, codebooks = quantize_and_finetune(
        model, masks, train_loader, test_loader, device,
        evaluate_fn=evaluate_accuracy
    )

    quantized_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\n[Stage 2] Accuracy after quantization + fine-tune: {quantized_acc:.2f}%")
    print(f"[Stage 2] Accuracy drop from baseline: {original_acc - quantized_acc:.2f}%")
    print(f"[Stage 2] Time: {time.time()-start:.1f}s")

    quant_path = os.path.join(config.COMPRESSED_DIR, "quantized_model.pth")
    torch.save({
        'state_dict': model.state_dict(),
        'masks': masks,
        'codebooks': codebooks,
    }, quant_path)

    # stage 3: huffman coding
    start = time.time()

    huffman_results, total_compressed_bits = huffman_encode_model(codebooks)

    print(f"\n[Stage 3] Time: {time.time()-start:.1f}s")

    # final summary
    final_acc = evaluate_accuracy(model, test_loader, device)

    print_compression_summary(
        model,
        original_size_bytes=original_size,
        codebooks=codebooks,
        huffman_results=huffman_results,
        original_acc=original_acc,
        compressed_acc=final_acc,
    )

    _, breakdown = compressed_size_bytes(model, codebooks, huffman_results)
    compression_ratio = (original_size / 1024 / 1024) / breakdown['total_mb'] if breakdown['total_mb'] > 0 else 0

    print(f"+---------------------------------------------------------+")
    print(f"|  DEEP COMPRESSION - FINAL RESULTS                       |")
    print(f"+---------------------------------------------------------+")
    print(f"|  Original size:      {original_size/1024/1024:8.2f} MB                     |")
    print(f"|  Compressed size:    {breakdown['total_mb']:8.2f} MB                     |")
    print(f"|  Compression ratio:  {compression_ratio:8.1f}x                      |")
    print(f"|  Original accuracy:  {original_acc:8.2f}%                     |")
    print(f"|  Final accuracy:     {final_acc:8.2f}%                     |")
    print(f"|  Accuracy drop:      {original_acc - final_acc:8.2f}%                     |")
    print(f"+---------------------------------------------------------+")

    # save results to file
    final_path = os.path.join(config.COMPRESSED_DIR, "compression_results.txt")
    with open(final_path, 'w') as f:
        f.write(f"Deep Compression Results\n")
        f.write(f"========================\n")
        f.write(f"Original size:     {original_size/1024/1024:.2f} MB\n")
        f.write(f"Compressed size:   {breakdown['total_mb']:.2f} MB\n")
        f.write(f"Compression ratio: {compression_ratio:.1f}x\n")
        f.write(f"Original accuracy: {original_acc:.2f}%\n")
        f.write(f"Final accuracy:    {final_acc:.2f}%\n")
        f.write(f"Accuracy drop:     {original_acc - final_acc:.2f}%\n")

    print(f"\n[Done] Results saved to {final_path}")
    print(f"[Done] Models saved in {config.COMPRESSED_DIR}/")


if __name__ == "__main__":
    main()
