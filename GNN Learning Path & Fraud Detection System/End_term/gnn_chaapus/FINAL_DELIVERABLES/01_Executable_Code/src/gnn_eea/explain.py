from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from .models import build_model


def explain_single_prediction(
    model_name: str,
    model_state: Dict[str, torch.Tensor],
    data: Data,
    feature_names: List[str],
    node_idx: int,
    hidden_dim: int,
    dropout: float,
    heads: int,
    output_path: str | Path,
    device: str = "cpu",
    epochs: int = 80,
) -> Optional[Path]:
    """Generate node-level explanation and save top edges/features to JSON.

    Uses the new torch_geometric.explain API when available, and fails gracefully.
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    except Exception:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model = build_model(
        name=model_name,
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        out_dim=2,
        dropout=dropout,
        heads=heads,
    ).to(torch_device)
    model.load_state_dict(model_state)
    model.eval()

    data = data.to(torch_device)

    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

        explanation = explainer(data.x, data.edge_index, index=int(node_idx))

        node_mask = explanation.node_mask.detach().cpu().numpy()
        edge_mask = explanation.edge_mask.detach().cpu().numpy()

        if node_mask.ndim == 2:
            # [num_nodes, num_features] for node explanations.
            feature_scores = node_mask[int(node_idx)]
        else:
            feature_scores = node_mask

        top_feature_idx = np.argsort(feature_scores)[::-1][:10]
        top_features = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"f_{i + 1}",
                "score": float(feature_scores[i]),
            }
            for i in top_feature_idx
        ]

        top_edge_idx = np.argsort(edge_mask)[::-1][:10]
        edges = data.edge_index.detach().cpu().numpy()
        top_edges = [
            {
                "source": int(edges[0, i]),
                "target": int(edges[1, i]),
                "score": float(edge_mask[i]),
            }
            for i in top_edge_idx
        ]

        payload = {
            "node_idx": int(node_idx),
            "model": model_name,
            "top_features": top_features,
            "top_edges": top_edges,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    except Exception:
        return None
