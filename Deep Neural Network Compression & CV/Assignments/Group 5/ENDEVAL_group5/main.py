# main.py — Entry point for the compression pipeline
#
# Two modes:
#   python main.py --mode compress   → loads trained weights, runs full pipeline,
#                                      saves compressed archive
#   python main.py --mode evaluate   → loads compressed archive, reports accuracy
#
# Pre-trained weights must be placed at the path in config.py (WEIGHTS_PATH)
# before running compress mode.

import argparse
import json
import os
import torch
import torch.nn as nn

import config
from models       import miniVGG
from data         import get_dataloaders
from utils        import evaluate, print_compression_report, load_model, load_compressed_model
from utils.training import fine_tune
from compression  import (apply_pruning, make_pruning_permanent, check_sparsity,
                           quantize_model, get_quant_stats,
                           huffman_encode_codebooks)


def run_compress():
    print(f'\nDevice: {config.DEVICE}')
    os.makedirs('compressed_models', exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    trainloader, testloader = get_dataloaders(
        config.DATA_DIR, config.BATCH_SIZE_TRAIN,
        config.BATCH_SIZE_TEST, config.NUM_WORKERS
    )

    # ── Load baseline model ───────────────────────────────────────────────────
    model = load_model(config.WEIGHTS_PATH, config.DEVICE)

    orig_kb      = sum(p.numel() * p.element_size()
                       for p in model.parameters()) / 1024
    baseline_acc = evaluate(model, testloader, config.DEVICE)
    print(f'\nBaseline accuracy : {baseline_acc*100:.2f}%')
    print(f'Baseline size     : {orig_kb:.2f} KB')

    # ── Stage 1: Pruning ──────────────────────────────────────────────────────
    print('\n── STAGE 1: L1 Unstructured Pruning ──')
    apply_pruning(model, config.PRUNING_AMOUNT)
    print('\nSparsity after pruning:')
    check_sparsity(model)

    loss_fn = nn.CrossEntropyLoss()
    fine_tune(model, trainloader, loss_fn, config.DEVICE,
              config.FINE_TUNE_LR, config.MOMENTUM,
              config.WEIGHT_DECAY, config.FINE_TUNE_EPOCHS)

    make_pruning_permanent(model)
    pruned_acc = evaluate(model, testloader, config.DEVICE)
    print(f'\nPruned accuracy   : {pruned_acc*100:.2f}%  '
          f'(drop: {(baseline_acc-pruned_acc)*100:.2f}%)')

    torch.save(model.state_dict(), config.PRUNED_PATH)
    print(f'Saved → {config.PRUNED_PATH}')

    # ── Stage 2: K-Means Quantization ────────────────────────────────────────
    print('\n── STAGE 2: K-Means Quantization ──')
    q_model, codebooks = quantize_model(
        model, config.N_CLUSTERS_CONV, config.N_CLUSTERS_FC
    )
    quant_acc   = evaluate(q_model, testloader, config.DEVICE)
    quant_stats = get_quant_stats(codebooks)
    print(f'Quantized accuracy : {quant_acc*100:.2f}%  '
          f'(drop from baseline: {(baseline_acc-quant_acc)*100:.2f}%)')
    print(f'Compression ratio  : {quant_stats["overall_ratio"]:.2f}x')
    print(f'Avg bits/weight    : {quant_stats["avg_bits_weight"]:.3f}')

    torch.save(q_model.state_dict(), config.QUANTIZED_PATH)
    print(f'Saved → {config.QUANTIZED_PATH}')

    # ── Stage 3: Huffman Coding ───────────────────────────────────────────────
    print('\n── STAGE 3: Huffman Encoding ──')
    encoded_data, huff_stats = huffman_encode_codebooks(codebooks)
    print(f'Size before Huffman : {huff_stats["size_before_huff_kb"]:.2f} KB')
    print(f'Size after  Huffman : {huff_stats["size_after_huff_kb"]:.2f} KB')
    print(f'Huffman gain        : {huff_stats["huffman_ratio"]:.4f}x')

    import pickle
    with open(config.ARCHIVE_PATH, 'wb') as f:
        pickle.dump({'encoded_data': encoded_data,
                     'huffman_stats': huff_stats}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    archive_kb    = os.path.getsize(config.ARCHIVE_PATH) / 1024
    overall_ratio = orig_kb / max(archive_kb, 1e-6)
    print(f'Archive saved → {config.ARCHIVE_PATH}  ({archive_kb:.2f} KB)')

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        'baseline_accuracy':   round(float(baseline_acc), 4),
        'pruned_accuracy':     round(float(pruned_acc), 4),
        'quantized_accuracy':  round(float(quant_acc), 4),
        'accuracy_drop':       round(float(baseline_acc - quant_acc), 4),
        'pruning_sparsity':    config.PRUNING_AMOUNT,
        'n_clusters_conv':     config.N_CLUSTERS_CONV,
        'n_clusters_fc':       config.N_CLUSTERS_FC,
        'original_size_kb':    round(orig_kb, 2),
        'after_huffman_kb':    round(archive_kb, 2),
        'overall_ratio':       round(overall_ratio, 3),
        'quantization_ratio':  round(quant_stats['overall_ratio'], 3),
        'huffman_gain':        round(huff_stats['huffman_ratio'], 4),
        'avg_bits_per_weight': round(quant_stats['avg_bits_weight'], 3),
    }
    with open(config.RESULTS_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results JSON → {config.RESULTS_JSON_PATH}')

    print(f'\n=== PIPELINE SUMMARY ===')
    print(f'  Baseline   : {baseline_acc*100:.2f}%  |  {orig_kb:.2f} KB')
    print(f'  After prune: {pruned_acc*100:.2f}%')
    print(f'  After quant: {quant_acc*100:.2f}%')
    print(f'  Final size : {archive_kb:.2f} KB  ({overall_ratio:.2f}x compression)')
    print(f'  Acc drop   : {(baseline_acc - quant_acc)*100:.2f}%')


def run_evaluate():
    print(f'\nDevice: {config.DEVICE}')

    _, testloader = get_dataloaders(
        config.DATA_DIR, config.BATCH_SIZE_TRAIN,
        config.BATCH_SIZE_TEST, config.NUM_WORKERS
    )

    model = load_compressed_model(
        config.ARCHIVE_PATH, config.QUANTIZED_PATH, config.DEVICE
    )
    acc = evaluate(model, testloader, config.DEVICE)
    print(f'\nCompressed model accuracy : {acc*100:.2f}%')

    if os.path.exists(config.RESULTS_JSON_PATH):
        with open(config.RESULTS_JSON_PATH) as f:
            r = json.load(f)
        print(f'Baseline accuracy         : {r["baseline_accuracy"]*100:.2f}%')
        print(f'Total accuracy drop       : '
              f'{(r["baseline_accuracy"] - acc)*100:.2f}%')
        print(f'Overall compression ratio : {r["overall_ratio"]:.2f}x')
        print(f'Final archive size        : {r["after_huffman_kb"]:.2f} KB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='miniVGG Compression Pipeline'
    )
    parser.add_argument(
        '--mode', choices=['compress', 'evaluate'], required=True,
        help='"compress" runs the full pipeline from trained weights. '
             '"evaluate" loads the compressed archive and reports accuracy.'
    )
    args = parser.parse_args()

    if args.mode == 'compress':
        run_compress()
    elif args.mode == 'evaluate':
        run_evaluate()
