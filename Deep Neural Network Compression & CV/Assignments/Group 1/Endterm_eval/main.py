# deep compression pipeline - main script
# end-to-end pipeline implementing the deep compression paper (Han et al., 2016):
#   1. data loading & feature extraction (using pretrained resnet18)
#   2. baseline MLP training (similar idea to assignment 6 but with nn.Module)
#   3. pruning via magnitude thresholding (zero out small weights)
#   4. fine-tuning (recover accuracy after pruning)
#   5. weight quantization using k-means (like assignment 5 but on neural net weights)
#   6. model serialization (save compressed in .npz)
#   7. huffman encoding (lossless compression of quantized indices)
#   8. compression summary & comparisons
#
# usage: python main.py --dataset cifar10 --epochs 30 --sparsity 0.6

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import copy

# add parent directory to path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.loader import load_dataset,extract_features,prepare_dataloaders
from models.mlp import MLP,create_mlp,train_model,evaluate
from compression.linear import CompressedLinear,replace_linear_with_compressed
from compression.pruning import apply_pruning,compute_sparsity
from compression.quantization import apply_quantization
from compression.huffman import (huffman_encode,compute_compression_ratio,
                                  compute_entropy)
from utils.serialization import save_compressed_npz,save_original_npz
from utils.visualization import (plot_weight_distribution,plot_sparsity,
                                  plot_training_curves,plot_compression_summary)
from torch.utils.data import DataLoader
from models.cnn import create_cnn
from compression.conv2d import CompressedConv2d,replace_conv2d_with_compressed

def main():
    print("="*60)
    print("Deep Compression Pipeline")
    print("="*60)

    # parse arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='cifar10')
    parser.add_argument('--epochs',type=int,default=EPOCHS_BASELINE)
    parser.add_argument('--finetune_epochs',type=int,default=EPOCHS_FINETUNE)
    parser.add_argument('--sparsity',type=float,default=PRUNE_SPARSITY)
    parser.add_argument('--clusters',type=int,default=NUM_CLUSTERS)
    parser.add_argument('--lr',type=float,default=LEARNING_RATE)
    parser.add_argument('--batch_size',type=int,default=BATCH_SIZE)
    parser.add_argument('--skip_extraction',action='store_true',
                        help='Skip feature extraction (use cached features)')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                    help='Choose architecture (default: cnn)')
    args=parser.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    os.makedirs(MODEL_DIR,exist_ok=True)
    cache_dir=os.path.join(BASE_DIR,'data_cache')
    feature_cache=os.path.join(cache_dir,f'{args.dataset}_features.npz')

    # ===========================================================
    # TASK 1: DATA LOADING & FEATURE EXTRACTION
    # ===========================================================

    print("\n"+"="*60)
    print("Task 1: Data Loading & Feature Extraction")
    print("="*60)

    if args.model == 'mlp':
        train_data, test_data, num_classes = load_dataset(args.dataset, data_dir=cache_dir, image_size=224)

        if args.skip_extraction and os.path.exists(feature_cache):
            print(f"Loading cached features from {feature_cache}")
            cached=np.load(feature_cache)
            train_features=cached['train_features']
            train_labels=cached['train_labels']
            test_features=cached['test_features']
            test_labels=cached['test_labels']
            num_classes=int(cached['num_classes'])
        else:
            train_features,train_labels=extract_features(
                train_data,batch_size=args.batch_size,device=device)
            test_features,test_labels=extract_features(
                test_data,batch_size=args.batch_size,device=device)

            os.makedirs(cache_dir,exist_ok=True)
            np.savez(feature_cache,
                     train_features=train_features, train_labels=train_labels,
                     test_features=test_features, test_labels=test_labels,
                     num_classes=num_classes)
            print(f"Features cached to {feature_cache}")

        # FIX: Move feature_dim and prepare_dataloaders INSIDE the mlp block!
        feature_dim = train_features.shape[1]
        print(f"\nFeature dimension: {feature_dim}")
        
        train_loader, test_loader = prepare_dataloaders(
            train_features, train_labels,
            test_features, test_labels,
            batch_size=args.batch_size)
    else:
        train_data, test_data, num_classes = load_dataset(args.dataset, data_dir=cache_dir, image_size=32)
        
        print("Using raw images for CNN training (Skipping ResNet feature extraction).")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)            

    print(f"Number of classes: {num_classes}")


    # ===========================================================
    # TASK 2: BASELINE TRAINING
    # ===========================================================
    print("\n"+"="*60)
    print("Task 2: Baseline Training")
    print("="*60)

    if args.model == 'mlp':
        model=create_mlp(input_dim=feature_dim,
                         hidden_layers=MLP_HIDDEN_LAYERS,
                         num_classes=num_classes,
                         dropout=MLP_DROPOUT)
    else:
        model = create_cnn(num_classes=num_classes)
    # count how many parameters we start with
    baseline_params=sum(p.numel() for p in model.parameters())
    print(f"\nBaseline parameter count: {baseline_params:,}")

    # train the baseline model
    print("\nTraining baseline model...")
    history_baseline=train_model(model,train_loader,test_loader,
                                 epochs=args.epochs,lr=args.lr,
                                 weight_decay=WEIGHT_DECAY,device=device)

    criterion=nn.CrossEntropyLoss()
    baseline_loss,baseline_acc=evaluate(model,test_loader,
                                        criterion,device)
    print(f"\nBaseline Test Accuracy: {baseline_acc:.2f}%")

    # save baseline for size comparison later
    baseline_path=os.path.join(MODEL_DIR,'baseline_model.npz')
    baseline_size=save_original_npz(model,baseline_path)

    # plot the training curves (like the loss/accuracy plots in assignment 6)
    plot_training_curves(history_baseline,title="Baseline Training",
                         save_path=os.path.join(MODEL_DIR,'baseline_training.png'))
    plot_weight_distribution(model,title="Baseline Weight Distribution",
                             save_path=os.path.join(MODEL_DIR,'baseline_weights.png'))

    # ===========================================================
    # TASK 3: PRUNING
    # ===========================================================
    print("\n"+"="*60)
    print("Task 3: Pruning (Magnitude-based)")
    print("="*60)

    # first swap nn.Linear for CompressedLinear (adds mask support)
    model = replace_linear_with_compressed(model)
    if args.model == 'cnn':
        model = replace_conv2d_with_compressed(model)
        
    model = model.to(device)
    
    # apply magnitude-based pruning globally
    model=apply_pruning(model,target_sparsity=args.sparsity)

    # check accuracy right after pruning (before fine-tuning)
    pruned_loss,pruned_acc_before=evaluate(model,test_loader,
                                           criterion,device)
    print(f"\nAccuracy after pruning (before fine-tuning): "
          f"{pruned_acc_before:.2f}%")

    # visualize how sparse each layer is
    sparsity_dict=compute_sparsity(model)
    plot_sparsity(sparsity_dict,
                  save_path=os.path.join(MODEL_DIR,'sparsity.png'))

    # ===========================================================
    # TASK 3 (cont): FINE-TUNING AFTER PRUNING
    # ===========================================================
    # fine-tune with reduced learning rate to recover accuracy
    print("\nFine-tuning pruned model...")
    history_finetune=train_model(model,train_loader,test_loader,
                                 epochs=args.finetune_epochs,
                                 lr=args.lr*0.1, # lower lr for fine-tuning
                                 weight_decay=WEIGHT_DECAY,
                                 device=device)

    pruned_loss,pruned_acc=evaluate(model,test_loader,
                                    criterion,device)
    print(f"\nAccuracy after fine-tuning: {pruned_acc:.2f}%")

    plot_weight_distribution(model,title="Pruned Weight Distribution",
                             save_path=os.path.join(MODEL_DIR,'pruned_weights.png'))

    # save pruned model
    pruned_path=os.path.join(MODEL_DIR,'pruned_model.npz')
    pruned_size=save_compressed_npz(model,pruned_path)

    # ===========================================================
    # TASK 4: QUANTIZATION
    # ===========================================================
    print("\n"+"="*60)
    print("Task 4: Weight Quantization (k-means)")
    print("="*60)

    # this is similar to assignment 5 kmeans_matrix() but applied to neural net weights
    quantization_info=apply_quantization(model,num_clusters=args.clusters)

    quant_loss,quant_acc=evaluate(model,test_loader,criterion,device)
    print(f"\nAccuracy after quantization: {quant_acc:.2f}%")

    plot_weight_distribution(model,title="Quantized Weight Distribution",
                             save_path=os.path.join(MODEL_DIR,'quantized_weights.png'))

    # save quantized model
    quant_path=os.path.join(MODEL_DIR,'quantized_model.npz')
    quant_size=save_compressed_npz(model,quant_path,
                                   quantization_info=quantization_info)

    # ===========================================================
    # TASK 7: HUFFMAN ENCODING
    # ===========================================================
    print("\n"+"="*60)
    print("Task 7: Huffman Encoding")
    print("="*60)

    huffman_results={}
    total_original_bits=0
    total_compressed_bits=0

    for name in quantization_info:
        q_info=quantization_info[name]
        indices=q_info['indices']
        # get the mask for this layer to only encode active (non-pruned) indices
        # get the mask for this layer to only encode active (non-pruned) indices
        for mod_name,mod in model.named_modules():
            # FIX: Check for BOTH CompressedLinear and CompressedConv2d
            if mod_name==name and isinstance(mod, (CompressedLinear, CompressedConv2d)):
                mask_flat=mod.mask.data.cpu().numpy().flatten()
                active_indices=indices[mask_flat>0]
                break

        if len(active_indices)==0:
            continue

        encoded_bits,codebook,freq_dict=huffman_encode(active_indices)
        entropy=compute_entropy(freq_dict)
        ratio,stats=compute_compression_ratio(
            active_indices,encoded_bits,
            original_bits_per_element=32)

        total_original_bits+=stats['original_size_bits']
        total_compressed_bits+=stats['compressed_size_bits']

        huffman_results[name]={
            'encoded_bits':encoded_bits,
            'codebook':codebook,
            'entropy':entropy,
            'stats':stats
        }

        print(f"\n  Layer '{name}':")
        print(f"    Symbols: {len(active_indices):,}")
        print(f"    Unique symbols: {len(freq_dict)}")
        print(f"    Shannon entropy: {entropy:.3f} bits") # theoretical minimum bits per symbol
        print(f"    Avg code length: {stats['avg_bits_per_symbol']:.3f} bits")
        print(f"    Original size: {stats['original_size_bytes']:,} bytes")
        print(f"    Compressed size: {stats['compressed_size_bytes']:,} bytes")
        print(f"    Compression ratio: {ratio:.2f}x")

    if total_compressed_bits>0:
        overall_ratio=total_original_bits/total_compressed_bits
        print(f"\n  Overall Huffman compression ratio: {overall_ratio:.2f}x")

    # save final model with all compression info
    final_path=os.path.join(MODEL_DIR,'final_compressed.npz')
    final_size=save_compressed_npz(model,final_path,
                                    quantization_info=quantization_info,
                                    huffman_info=huffman_results)

    # ===========================================================
    # COMPRESSION SUMMARY
    # ===========================================================
    print("\n"+"="*60)
    print("Compression Summary")
    print("="*60)

    sizes={
        'Baseline':baseline_size,
        'Pruned':pruned_size,
        'Quantized':quant_size,
        'Huffman':final_size
    }

    accuracies={
        'Baseline':baseline_acc,
        'Pruned':pruned_acc,
        'Quantized':quant_acc,
        'Huffman':quant_acc # huffman is lossless so accuracy stays the same
    }

    print(f"\n{'Stage':<15} {'Size (KB)':>10} {'Accuracy':>10} {'Ratio':>10}")
    print("-"*45)
    for stage in sizes:
        ratio=baseline_size/sizes[stage]
        print(f"{stage:<15} {sizes[stage]/1024:>10.1f} "
              f"{accuracies[stage]:>9.2f}% {ratio:>9.2f}x")

    # final plot showing everything together
    plot_compression_summary(sizes,accuracies,
                             save_path=os.path.join(MODEL_DIR,
                                                     'compression_summary.png'))

    print("\n"+"="*60)
    print("Pipeline complete. Compressed models saved in:",MODEL_DIR)
    print("="*60)


if __name__=='__main__':
    main()
