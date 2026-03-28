#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd

from gnn_eea.config import load_config
from gnn_eea.data import load_elliptic_dataset
from gnn_eea.train import train_gnn


def score(metrics: dict) -> float:
    return (
        float(metrics.get("pr_auc", 0.0))
        + 0.60 * float(metrics.get("macro_f1", 0.0))
        + 0.20 * float(metrics.get("illicit_recall", 0.0))
        - 0.20 * float(metrics.get("false_positive_rate", 0.0))
    )


def build_space(model_name: str) -> list[dict]:
    if model_name == "gcn":
        space = {
            "hidden_dim": [128, 192, 256],
            "dropout": [0.35, 0.45, 0.55],
            "learning_rate": [5e-4, 1e-3, 1.5e-3],
            "weight_decay": [1e-4, 5e-4],
            "use_focal_loss": [True, False],
            "focal_gamma": [1.5, 2.0],
            "heads": [4],
        }
    elif model_name == "graphsage":
        space = {
            "hidden_dim": [128, 192, 256],
            "dropout": [0.25, 0.35, 0.45],
            "learning_rate": [5e-4, 1e-3],
            "weight_decay": [1e-4, 5e-4],
            "use_focal_loss": [True, False],
            "focal_gamma": [1.5, 2.0],
            "heads": [4],
        }
    else:
        space = {
            "hidden_dim": [64, 96, 128],
            "dropout": [0.30, 0.40, 0.50],
            "learning_rate": [5e-4, 1e-3, 1.5e-3],
            "weight_decay": [1e-4, 5e-4],
            "use_focal_loss": [True, False],
            "focal_gamma": [1.5, 2.0],
            "heads": [4, 6],
        }

    keys = list(space.keys())
    combos = []
    for values in product(*[space[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune GNN hyperparameters for the project.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-mode", choices=["all", "local"], default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    use_local = args.feature_mode == "local"
    loaded = load_elliptic_dataset(
        data_dir=cfg.data.data_dir,
        use_local_features_only=use_local,
        standardize=cfg.data.standardize,
        make_undirected_graph=cfg.data.make_undirected,
        train_time_max=cfg.data.train_time_max,
        val_time_max=cfg.data.val_time_max,
    )
    data = loaded.data

    out_rows = []
    best_overrides: dict[str, dict] = {}

    for model_name in cfg.models:
        candidates = build_space(model_name)[: args.max_trials]
        best = None
        for i, cand in enumerate(candidates, 1):
            res = train_gnn(
                model_name=model_name,
                data=data,
                hidden_dim=int(cand["hidden_dim"]),
                dropout=float(cand["dropout"]),
                heads=int(cand["heads"]),
                learning_rate=float(cand["learning_rate"]),
                weight_decay=float(cand["weight_decay"]),
                epochs=args.epochs,
                patience=10,
                use_focal_loss=bool(cand["use_focal_loss"]),
                focal_gamma=float(cand["focal_gamma"]),
                threshold=cfg.training.threshold,
                grad_clip=cfg.training.grad_clip,
                seed=cfg.training.random_seed,
                device=args.device,
            )
            m = dict(res.metrics)
            s = score(m)
            row = {
                "model": model_name,
                "trial": i,
                "score": s,
                **cand,
                **m,
            }
            out_rows.append(row)
            if best is None or s > best["score"]:
                best = row

        assert best is not None
        best_overrides[model_name] = {
            "hidden_dim": int(best["hidden_dim"]),
            "dropout": float(best["dropout"]),
            "learning_rate": float(best["learning_rate"]),
            "weight_decay": float(best["weight_decay"]),
            "use_focal_loss": bool(best["use_focal_loss"]),
            "focal_gamma": float(best["focal_gamma"]),
            "heads": int(best["heads"]),
        }
        print(f"[{model_name}] best score={best['score']:.4f} pr_auc={best['pr_auc']:.4f} macro_f1={best['macro_f1']:.4f}")

    metrics_dir = Path(cfg.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "local_only_94" if use_local else "all_features"
    trials_path = metrics_dir / f"hyperparam_trials_{mode_tag}.csv"
    pd.DataFrame(out_rows).sort_values(["model", "score"], ascending=[True, False]).to_csv(trials_path, index=False)

    best_path = metrics_dir / f"best_overrides_{mode_tag}.json"
    best_path.write_text(json.dumps(best_overrides, indent=2), encoding="utf-8")

    print(f"Wrote: {trials_path}")
    print(f"Wrote: {best_path}")


if __name__ == "__main__":
    main()
