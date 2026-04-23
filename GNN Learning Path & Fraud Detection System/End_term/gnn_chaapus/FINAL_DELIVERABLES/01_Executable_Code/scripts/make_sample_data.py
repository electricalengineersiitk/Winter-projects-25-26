#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path("data/sample")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_nodes = 4000
    n_edges = 12000
    feature_dim = 166  # 94 local + 72 aggregated (as in assignment description)

    tx_ids = np.arange(100000, 100000 + n_nodes)
    time_step = rng.integers(1, 50, size=n_nodes)

    # Synthetic feature matrix: first feature is time step, rest are random.
    feats = rng.normal(loc=0.0, scale=1.0, size=(n_nodes, feature_dim)).astype(np.float32)
    feats[:, 0] = time_step

    # Class split close to Elliptic proportions: 2% illicit, 21% licit, 77% unknown.
    labels = np.full(n_nodes, "unknown", dtype=object)
    illicit_count = int(0.02 * n_nodes)
    licit_count = int(0.21 * n_nodes)

    idx = rng.permutation(n_nodes)
    illicit_idx = idx[:illicit_count]
    licit_idx = idx[illicit_count : illicit_count + licit_count]
    labels[illicit_idx] = "1"
    labels[licit_idx] = "2"

    # Inject a weak signal for illicit nodes to make training meaningful.
    feats[illicit_idx, 1:8] += 1.5
    feats[licit_idx, 1:8] -= 0.3

    # Save features in expected CSV format: txId + 166 features.
    feature_df = pd.DataFrame(np.column_stack([tx_ids, feats]))
    feature_df.to_csv(out_dir / "elliptic_txs_features.csv", header=False, index=False)

    class_df = pd.DataFrame({"txId": tx_ids, "class": labels})
    class_df.to_csv(out_dir / "elliptic_txs_classes.csv", index=False)

    # Generate directed edges with some temporal coherence.
    src = rng.choice(tx_ids, size=n_edges, replace=True)
    dst = rng.choice(tx_ids, size=n_edges, replace=True)
    keep = src != dst
    edge_df = pd.DataFrame({"txId1": src[keep], "txId2": dst[keep]})
    edge_df.to_csv(out_dir / "elliptic_txs_edgelist.csv", index=False)

    print(f"Synthetic sample dataset written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
