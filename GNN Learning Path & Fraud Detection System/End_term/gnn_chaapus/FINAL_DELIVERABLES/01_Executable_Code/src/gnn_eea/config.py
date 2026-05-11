from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class TrainingConfig:
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    hidden_dim: int = 128
    dropout: float = 0.3
    heads: int = 4
    focal_gamma: float = 2.0
    use_focal_loss: bool = True
    grad_clip: float = 1.0
    early_stopping_patience: int = 12
    random_seed: int = 42
    threshold: float = 0.5
    device: str = "cpu"


@dataclass
class DataConfig:
    data_dir: str = "data/elliptic"
    use_local_features_only: bool = False
    standardize: bool = True
    make_undirected: bool = True
    train_time_max: int = 34
    val_time_max: int = 39


@dataclass
class ProjectConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: List[str] = field(default_factory=lambda: ["gcn", "graphsage", "gat"])
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feature_model_overrides: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    output_dir: str = "artifacts"
    run_feature_ablation: bool = True
    run_explainability: bool = True


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> ProjectConfig:
    default = ProjectConfig()
    if path is None:
        return default

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    default_dict = {
        "training": default.training.__dict__,
        "data": default.data.__dict__,
        "models": default.models,
        "model_overrides": default.model_overrides,
        "feature_model_overrides": default.feature_model_overrides,
        "output_dir": default.output_dir,
        "run_feature_ablation": default.run_feature_ablation,
        "run_explainability": default.run_explainability,
    }
    merged = _deep_update(default_dict, user_cfg)

    return ProjectConfig(
        training=TrainingConfig(**merged.get("training", {})),
        data=DataConfig(**merged.get("data", {})),
        models=merged.get("models", ["gcn", "graphsage", "gat"]),
        model_overrides=merged.get("model_overrides", {}),
        feature_model_overrides=merged.get("feature_model_overrides", {}),
        output_dir=merged.get("output_dir", "artifacts"),
        run_feature_ablation=merged.get("run_feature_ablation", True),
        run_explainability=merged.get("run_explainability", True),
    )
