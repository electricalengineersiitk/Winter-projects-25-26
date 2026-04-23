#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gnn_eea.config import load_config
from gnn_eea.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Elliptic GNN final project pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps, auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path if cfg_path.exists() else None)

    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.device is not None:
        cfg.training.device = args.device

    summary = run_pipeline(cfg)

    output_root = Path(cfg.output_dir)
    table_path = output_root / "metrics" / "results_summary.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    with table_path.open("w", encoding="utf-8") as f:
        f.write("# Results Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print(f"Pipeline completed. Artifacts saved in: {output_root.resolve()}")
    print(f"Summary CSV: {(output_root / 'metrics' / 'results_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
