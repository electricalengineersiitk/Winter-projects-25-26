from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


@dataclass
class LoadedGraphData:
    data: Data
    feature_names: List[str]
    metadata: Dict[str, float]


def _label_to_binary(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "illicit", "fraud", "true"}:
        return 1
    if text in {"2", "licit", "false", "0"}:
        return 0
    return -1


def _build_masks_with_time(
    y: np.ndarray,
    time_step: np.ndarray,
    train_time_max: int,
    val_time_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labeled = y != -1
    train_mask = labeled & (time_step <= train_time_max)
    val_mask = labeled & (time_step > train_time_max) & (time_step <= val_time_max)
    test_mask = labeled & (time_step > val_time_max)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        labeled_idx = np.where(labeled)[0]
        if labeled_idx.size < 20:
            raise ValueError("Not enough labeled samples to create train/val/test splits.")

        y_labeled = y[labeled_idx]
        train_idx, tmp_idx = train_test_split(
            labeled_idx,
            test_size=0.4,
            random_state=42,
            stratify=y_labeled,
        )
        y_tmp = y[tmp_idx]
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=0.5,
            random_state=42,
            stratify=y_tmp,
        )

        train_mask = np.zeros_like(y, dtype=bool)
        val_mask = np.zeros_like(y, dtype=bool)
        test_mask = np.zeros_like(y, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def _compute_metadata(y: np.ndarray, features_all: np.ndarray, features_used: np.ndarray) -> Dict[str, float]:
    labeled = y != -1
    licit = int((y == 0).sum())
    illicit = int((y == 1).sum())
    unknown = int((y == -1).sum())
    total = int(y.shape[0])

    return {
        "num_nodes": total,
        "num_labeled": int(labeled.sum()),
        "num_licit": licit,
        "num_illicit": illicit,
        "num_unknown": unknown,
        "illicit_ratio_labeled": float(illicit / max(1, licit + illicit)),
        "all_feature_dim": int(features_all.shape[1]),
        "used_feature_dim": int(features_used.shape[1]),
    }


def _standardize_train_stats(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train_x = x[train_mask]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (x - mean) / std


def load_elliptic_dataset(
    data_dir: str | Path,
    use_local_features_only: bool = False,
    standardize: bool = True,
    make_undirected_graph: bool = True,
    train_time_max: int = 34,
    val_time_max: int = 39,
) -> LoadedGraphData:
    data_path = Path(data_dir)
    classes_path = data_path / "elliptic_txs_classes.csv"
    edges_path = data_path / "elliptic_txs_edgelist.csv"
    features_path = data_path / "elliptic_txs_features.csv"

    missing = [p for p in (classes_path, edges_path, features_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Elliptic files: "
            + ", ".join(str(m) for m in missing)
            + ". Place Kaggle files in the data directory."
        )

    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)
    features = pd.read_csv(features_path, header=None)

    if features.shape[1] < 4:
        raise ValueError("Features file appears malformed; expected txId + feature columns.")

    # Expected format: txId, time_step, <remaining features>
    numeric_dim = features.shape[1] - 1
    feature_columns = ["txId"] + [f"f_{i}" for i in range(1, numeric_dim + 1)]
    features.columns = feature_columns

    merged = features.merge(classes, on="txId", how="left")
    if "class" not in merged.columns:
        # Some variants use 'label' column.
        if "label" in merged.columns:
            merged["class"] = merged["label"]
        else:
            raise ValueError("Class column not found in classes file.")

    merged = merged.sort_values("txId").reset_index(drop=True)

    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(merged["txId"].tolist())}

    # Map labels to binary: licit=0, illicit=1, unknown=-1
    y_np = merged["class"].apply(_label_to_binary).to_numpy(dtype=np.int64)

    full_feature_cols = [c for c in merged.columns if c.startswith("f_")]
    all_features = merged[full_feature_cols].to_numpy(dtype=np.float32)

    # Based on assignment notes: 94 local + 72 aggregated = 166 total.
    if use_local_features_only:
        local_dim = min(94, all_features.shape[1])
        feature_names = full_feature_cols[:local_dim]
        used_features = all_features[:, :local_dim]
    else:
        feature_names = full_feature_cols
        used_features = all_features

    if "f_1" in merged.columns:
        time_step = merged["f_1"].to_numpy(dtype=np.int64)
    else:
        # Fallback if the first feature is not time step.
        time_step = np.zeros(merged.shape[0], dtype=np.int64)

    # Build edge index by mapping txIds to node indices and dropping out-of-scope edges.
    edges = edges.rename(columns={edges.columns[0]: "txId1", edges.columns[1]: "txId2"})
    edge_pairs = []
    for src, dst in edges[["txId1", "txId2"]].itertuples(index=False):
        if src in id_to_idx and dst in id_to_idx:
            edge_pairs.append((id_to_idx[src], id_to_idx[dst]))

    if not edge_pairs:
        raise ValueError("No valid edges after mapping txId values.")

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    if make_undirected_graph:
        edge_index = to_undirected(edge_index)

    train_np, val_np, test_np = _build_masks_with_time(
        y=y_np,
        time_step=time_step,
        train_time_max=train_time_max,
        val_time_max=val_time_max,
    )

    x = torch.tensor(used_features, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    train_mask = torch.tensor(train_np, dtype=torch.bool)
    val_mask = torch.tensor(val_np, dtype=torch.bool)
    test_mask = torch.tensor(test_np, dtype=torch.bool)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        time_step=torch.tensor(time_step, dtype=torch.long),
    )

    if standardize:
        data.x = _standardize_train_stats(data.x, data.train_mask)

    metadata = _compute_metadata(y_np, all_features, used_features)
    metadata["num_edges"] = int(edge_index.shape[1])
    metadata["train_labeled"] = int(train_mask.sum().item())
    metadata["val_labeled"] = int(val_mask.sum().item())
    metadata["test_labeled"] = int(test_mask.sum().item())

    return LoadedGraphData(data=data, feature_names=feature_names, metadata=metadata)
