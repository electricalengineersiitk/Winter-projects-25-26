

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loading  import load_model_from_npz
from .training import evaluate


# ──────────────────────────────────────────────────────────────────────────────
# test_compressed_model
# ──────────────────────────────────────────────────────────────────────────────

def test_compressed_model(
    model:       nn.Module,
    npz_path:    str | os.PathLike,
    test_loader: DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    verbose:     bool = True,
) -> Dict[str, float]:
    
    npz_path = Path(npz_path)

    # ── Step 1: Decode ────────────────────────────────────────────────────────
    t0 = time.time()
    model = load_model_from_npz(model, npz_path, device)
    decode_time = time.time() - t0

    if verbose:
        print(f"[Test] Decoded in {decode_time:.3f}s")

    # ── Step 2: Evaluate ──────────────────────────────────────────────────────
    t1 = time.time()
    metrics = evaluate(model, test_loader, criterion, device, verbose=verbose)
    inference_time = time.time() - t1

    compressed_kb = npz_path.stat().st_size / 1024

    results = {
        **metrics,
        "decode_time_s":      round(decode_time,      4),
        "inference_time_s":   round(inference_time,   4),
        "compressed_size_kb": round(compressed_kb,    2),
    }

    if verbose:
        _print_box(
            title="COMPRESSED MODEL — TEST RESULTS",
            rows=[
                ("Compressed file",  f"{npz_path.name}  ({compressed_kb:.1f} KB)"),
                ("Decode time",      f"{decode_time:.3f}s"),
                ("Inference time",   f"{inference_time:.3f}s"),
                ("Test loss",        f"{metrics['loss']:.4f}"),
                ("Top-1 accuracy",   f"{metrics['accuracy']:.2f}%"),
                ("Top-5 accuracy",   f"{metrics['top5_accuracy']:.2f}%"),
            ],
        )

    return results


def compare_models(
    original_model: nn.Module,
    npz_path:       str | os.PathLike,
    model_factory:  Callable[[], nn.Module],
    test_loader:    DataLoader,
    criterion:      nn.Module,
    device:         torch.device,
) -> Dict:
    
    npz_path = Path(npz_path)

    print("\n[Compare] ── Evaluating ORIGINAL model ──────────────────────")
    orig_metrics = evaluate(original_model, test_loader, criterion, device)

    print("\n[Compare] ── Evaluating COMPRESSED model ────────────────────")
    fresh_model  = model_factory()
    comp_metrics = test_compressed_model(
        fresh_model, npz_path, test_loader, criterion, device, verbose=False
    )

    acc_drop  = orig_metrics["accuracy"]      - comp_metrics["accuracy"]
    top5_drop = orig_metrics["top5_accuracy"] - comp_metrics["top5_accuracy"]

    # Rough in-memory size of original model (parameters + buffers)
    orig_bytes = sum(
        p.numel() * p.element_size()
        for p in list(original_model.parameters()) + list(original_model.buffers())
    )
    orig_kb = orig_bytes / 1024
    comp_kb = npz_path.stat().st_size / 1024

    _print_table(
        headers=["Metric", "Original", "Compressed"],
        rows=[
            ("Loss",                f"{orig_metrics['loss']:.4f}",
                                    f"{comp_metrics['loss']:.4f}"),
            ("Top-1 accuracy (%)",  f"{orig_metrics['accuracy']:.2f}",
                                    f"{comp_metrics['accuracy']:.2f}"),
            ("Top-5 accuracy (%)",  f"{orig_metrics['top5_accuracy']:.2f}",
                                    f"{comp_metrics['top5_accuracy']:.2f}"),
            ("Size (KB)",           f"{orig_kb:.1f}  (in-memory)",
                                    f"{comp_kb:.1f}  (on disk)"),
        ],
        footer_rows=[
            ("Top-1 accuracy drop", f"{acc_drop:+.4f}%"),
            ("Top-5 accuracy drop", f"{top5_drop:+.4f}%"),
            ("Size reduction",      f"{orig_kb / max(comp_kb, 0.001):.2f}×"),
        ],
    )

    return {
        "original":           orig_metrics,
        "compressed":         comp_metrics,
        "accuracy_drop":      acc_drop,
        "top5_accuracy_drop": top5_drop,
        "size_reduction_kb":  orig_kb - comp_kb,
    }



def benchmark_compression(
    original_model: nn.Module,
    npz_path:       str | os.PathLike,
    model_factory:  Callable[[], nn.Module],
    device:         torch.device,
    n_warmup:       int   = 5,
    n_runs:         int   = 20,
    batch_size:     int   = 32,
    input_shape:    tuple = (3, 224, 224),
) -> Dict:
    
    npz_path = Path(npz_path)

    def _model_size_mb(m: nn.Module) -> float:
        total = sum(
            t.numel() * t.element_size()
            for t in list(m.parameters()) + list(m.buffers())
        )
        return total / (1024 ** 2)

    def _latency_ms(m: nn.Module) -> float:
        m.eval().to(device)
        dummy = torch.randn(batch_size, *input_shape, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                m(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                m(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000  

    orig_size_mb  = _model_size_mb(original_model)
    comp_size_kb  = npz_path.stat().st_size / 1024


    print("[Benchmark] Reconstructing compressed model for latency test …")
    comp_model = model_factory()
    comp_model  = load_model_from_npz(comp_model, npz_path, device)

    print("[Benchmark] Measuring original model latency …")
    orig_lat_ms = _latency_ms(original_model)

    print("[Benchmark] Measuring compressed model latency …")
    comp_lat_ms = _latency_ms(comp_model)

    ratio = (orig_size_mb * 1024) / max(comp_size_kb, 0.001)

    results = {
        "original_size_mb":      round(orig_size_mb,          3),
        "compressed_size_kb":    round(comp_size_kb,          2),
        "size_reduction_ratio":  round(ratio,                 4),
        "original_latency_ms":   round(orig_lat_ms,           3),
        "compressed_latency_ms": round(comp_lat_ms,           3),
        "latency_overhead_ms":   round(comp_lat_ms - orig_lat_ms, 3),
        "batch_size":            batch_size,
        "n_runs":                n_runs,
    }

    _print_box(
        title="BENCHMARK SUMMARY",
        rows=[
            ("Original model size (in-memory)", f"{orig_size_mb:.2f} MB"),
            ("Compressed file size (on disk)",  f"{comp_size_kb:.1f} KB"),
            ("Size reduction ratio",            f"{ratio:.2f}×"),
            ("Original latency",                f"{orig_lat_ms:.2f} ms / batch"),
            ("Compressed latency",              f"{comp_lat_ms:.2f} ms / batch"),
            ("Latency overhead",                f"{comp_lat_ms - orig_lat_ms:+.2f} ms"),
        ],
    )
    return results

def _print_box(title: str, rows: list) -> None:
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    for label, value in rows:
        print(f"  {label:<32} {value}")
    print("=" * width + "\n")


def _print_table(headers: list, rows: list, footer_rows: Optional[list] = None) -> None:
    from typing import Optional
    col_w = [28, 20, 20]
    sep   = "─" * (sum(col_w) + 6)
    print("\n" + sep)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
    if footer_rows:
        print(sep)
        for label, value in footer_rows:
            print(f"  {label:<28}  {value}")
    print(sep + "\n")
