from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _unpack_batch(batch, device):
    if isinstance(batch, dict):
        inputs  = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
    else:
        inputs, targets = batch
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    return inputs, targets


def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    scheduler:    Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler:       Optional[torch.cuda.amp.GradScaler]             = None,
    log_interval: int                                              = 50,
    epoch_num:    Optional[int]                                    = None,
) -> Dict[str, float]:

    model.train()
    model.to(device)

    running_loss = 0.0
    correct      = 0
    total        = 0
    t0           = time.time()
    prefix       = f"Epoch {epoch_num} | " if epoch_num is not None else ""

    for batch_idx, batch in enumerate(loader):
        inputs, targets = _unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs            = inputs.size(0)
        running_loss += loss.item() * bs
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(targets).sum().item()
        total        += bs

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / total
            acc      = 100.0 * correct / total
            lr       = optimizer.param_groups[0]["lr"]
            print(
                f"  {prefix}Batch [{batch_idx + 1:>4}/{len(loader)}]  "
                f"loss={avg_loss:.4f}  acc={acc:.2f}%  lr={lr:.2e}"
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = 100.0 * correct  / max(total, 1)
    elapsed    = time.time() - t0

    print(
        f"[Train] {prefix}loss={epoch_loss:.4f}  "
        f"acc={epoch_acc:.2f}%  time={elapsed:.1f}s"
    )
    return {
        "loss":         epoch_loss,
        "accuracy":     epoch_acc,
        "epoch_time_s": elapsed,
    }


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    verbose:   bool = True,
) -> Dict[str, float]:
    
    model.eval()
    model.to(device)

    running_loss  = 0.0
    correct_top1  = 0
    correct_top5  = 0
    total         = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = _unpack_batch(batch, device)

            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            bs            = inputs.size(0)
            running_loss += loss.item() * bs
            total        += bs

            # Top-1
            _, pred1 = outputs.max(1)
            correct_top1 += pred1.eq(targets).sum().item()

            # Top-5 (safe when num_classes < 5)
            k = min(5, outputs.size(1))
            _, pred_k   = outputs.topk(k, dim=1, largest=True, sorted=True)
            targets_exp = targets.view(-1, 1).expand_as(pred_k)
            correct_top5 += pred_k.eq(targets_exp).any(dim=1).sum().item()

    avg_loss = running_loss / max(total, 1)
    top1_acc = 100.0 * correct_top1 / max(total, 1)
    top5_acc = 100.0 * correct_top5 / max(total, 1)

    if verbose:
        print(
            f"[Eval]  loss={avg_loss:.4f}  "
            f"top1={top1_acc:.2f}%  top5={top5_acc:.2f}%"
        )
    return {
        "loss":          avg_loss,
        "accuracy":      top1_acc,
        "top5_accuracy": top5_acc,
    }

def train_and_eval(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 10,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,       
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,     
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr =1e-6,
        # verbose=True,
    )

    criterion_eval = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch_num=epoch,
        )
        eval_results = evaluate(model, test_loader, criterion_eval, device)

        scheduler.step(eval_results['loss'])



train_model = train_one_epoch