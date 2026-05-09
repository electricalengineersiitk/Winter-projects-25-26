import torch
import torch.optim as optim
import copy

from config import *
from models.model import get_model
from data.data_loader import get_dataloaders
from utils.train import train_model, test_model
from utils.metrics import check_sparsity
from compression.prune import l1_prune
from compression.rewind import rewind_weights
from compression.quantize import quantize_model
from compression.huffman import save_compressed_model, compute_compression_stats


def main():
    print(f"Running on: {DEVICE}")

    model = get_model(DEVICE)
    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    initial_state_dict = copy.deepcopy(model.state_dict())

    masks = {
        name: torch.ones_like(param).to(DEVICE)
        for name, param in model.named_parameters()
        if 'weight' in name
    }

    for cycle in range(PRUNING_CYCLES):
        print(f"\n===== Cycle {cycle+1}/{PRUNING_CYCLES} =====")

        optimizer = optim.Adam(model.parameters(), lr=LR)

        print("[Training]")
        train_model(model, trainloader, optimizer, masks,
                    EPOCHS_PER_CYCLE, DEVICE)

        test_model(model, testloader, DEVICE)

        print("[Pruning]")
        masks = l1_prune(model, masks, PRUNE_RATE)
        check_sparsity(masks)

        print("[Rewinding]")
        rewind_weights(model, initial_state_dict, masks)

    print("\nFinal Refinement")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(model, trainloader, optimizer, masks, 10, DEVICE)

    test_model(model, testloader, DEVICE)
    check_sparsity(masks)

    print("\n--- STAGE 2: K-MEANS QUANTIZATION ---")
    quantized_state_dict = quantize_model(model, masks, bits=3)

    print("\n--- STAGE 3: HUFFMAN CODING & SERIALIZATION ---")
    save_path = "compressed_models/vgg_compressed.npz"
    save_compressed_model(quantized_state_dict, save_path)

    compute_compression_stats(model, save_path)

    from deployment.load_model import load_compressed_model, run_inference

    deployment_model = load_compressed_model("compressed_models/vgg_compressed.npz")
    run_inference(deployment_model)

if __name__ == "__main__":
    main()