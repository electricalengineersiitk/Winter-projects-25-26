from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .baselines import run_tabular_baseline
from .config import ProjectConfig
from .data import load_elliptic_dataset
from .explain import explain_single_prediction
from .train import predict_all_probabilities, train_gnn


def _resolve_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "models": output_dir / "models",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
        "explain": output_dir / "explain",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _pick_explain_node(data: Data, probs: np.ndarray) -> int | None:
    test_mask = data.test_mask.cpu().numpy()
    y = data.y.cpu().numpy()
    candidates = np.where(test_mask & (y == 1))[0]
    if candidates.size == 0:
        return None
    best = candidates[np.argmax(probs[candidates])]
    return int(best)


def run_pipeline(config: ProjectConfig) -> pd.DataFrame:
    output_dir = Path(config.output_dir)
    dirs = _ensure_dirs(output_dir)

    feature_modes = [False, True] if config.run_feature_ablation else [config.data.use_local_features_only]
    rows: List[Dict[str, float]] = []

    for use_local in feature_modes:
        feature_tag = "local_only_94" if use_local else "all_features"

        loaded = load_elliptic_dataset(
            data_dir=config.data.data_dir,
            use_local_features_only=use_local,
            standardize=config.data.standardize,
            make_undirected_graph=config.data.make_undirected,
            train_time_max=config.data.train_time_max,
            val_time_max=config.data.val_time_max,
        )

        data = loaded.data
        baseline = run_tabular_baseline(
            data=data,
            threshold=config.training.threshold,
            seed=config.training.random_seed,
        )

        baseline_row = {
            "feature_mode": feature_tag,
            "model": baseline.model_name,
            **baseline.metrics,
            "fpr_reduction_vs_baseline": 0.0,
        }
        rows.append(baseline_row)
        baseline_fpr = baseline.metrics["false_positive_rate"]

        for model_name in config.models:
            feature_override_bucket = config.feature_model_overrides.get(feature_tag, {})
            overrides = feature_override_bucket.get(model_name, config.model_overrides.get(model_name, {}))
            hidden_dim = int(overrides.get("hidden_dim", config.training.hidden_dim))
            dropout = float(overrides.get("dropout", config.training.dropout))
            heads = int(overrides.get("heads", config.training.heads))
            learning_rate = float(overrides.get("learning_rate", config.training.learning_rate))
            weight_decay = float(overrides.get("weight_decay", config.training.weight_decay))
            use_focal_loss = bool(overrides.get("use_focal_loss", config.training.use_focal_loss))
            focal_gamma = float(overrides.get("focal_gamma", config.training.focal_gamma))
            grad_clip = float(overrides.get("grad_clip", config.training.grad_clip))

            result = train_gnn(
                model_name=model_name,
                data=data,
                hidden_dim=hidden_dim,
                dropout=dropout,
                heads=heads,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=config.training.epochs,
                patience=config.training.early_stopping_patience,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                threshold=config.training.threshold,
                grad_clip=grad_clip,
                seed=config.training.random_seed,
                device=_resolve_device(config.training.device),
            )

            model_path = dirs["models"] / f"{model_name}_{feature_tag}.pt"
            torch.save(result.model_state, model_path)

            history_df = pd.DataFrame(result.history)
            history_path = dirs["metrics"] / f"history_{model_name}_{feature_tag}.csv"
            history_df.to_csv(history_path, index=False)

            if baseline_fpr > 1e-12:
                fpr_reduction = float((baseline_fpr - result.metrics["false_positive_rate"]) / baseline_fpr)
            else:
                # If baseline FPR is zero, percentage reduction is not defined.
                # Use 0.0 to keep the summary table numeric and stable.
                fpr_reduction = 0.0

            model_row = {
                "feature_mode": feature_tag,
                "model": model_name,
                **result.metrics,
                "fpr_reduction_vs_baseline": fpr_reduction,
            }
            rows.append(model_row)

            if config.run_explainability:
                probs = predict_all_probabilities(
                    model_name=model_name,
                    model_state=result.model_state,
                    data=data,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    heads=heads,
                    device=_resolve_device(config.training.device),
                )
                node_idx = _pick_explain_node(data, probs)
                if node_idx is not None:
                    explain_single_prediction(
                        model_name=model_name,
                        model_state=result.model_state,
                        data=data,
                        feature_names=loaded.feature_names,
                        node_idx=node_idx,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        heads=heads,
                        output_path=dirs["explain"] / f"{model_name}_{feature_tag}_node_{node_idx}.json",
                        device=_resolve_device(config.training.device),
                    )

        metadata_path = dirs["metrics"] / f"dataset_metadata_{feature_tag}.json"
        metadata_path.write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")

    summary = pd.DataFrame(rows)
    summary_path = dirs["metrics"] / "results_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary_json = dirs["metrics"] / "results_summary.json"
    summary_json.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    # Add rankings for easy interpretation.
    ranked = summary.sort_values(by=["feature_mode", "pr_auc", "macro_f1"], ascending=[True, False, False])
    ranked_path = dirs["metrics"] / "results_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    return summary
