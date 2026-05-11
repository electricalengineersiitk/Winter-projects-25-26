# GNN Final Project: Illicit Financial Transaction Detection

## Team
- Akshat Mittal (240086)
- Aaditya Mukherjee (240006)
- Daksh Gupta (240317)
- Nitish Choudhary (240710)

## 1) Project Objective
This project detects illicit cryptocurrency transactions by modeling the Elliptic dataset as a graph and benchmarking graph neural networks against a strong tabular baseline.

We specifically target assignment KPIs:
- PR-AUC (primary)
- Macro F1
- Illicit-class Recall
- False Positive Rate (FPR)
- FPR reduction vs baseline

## 2) What This Repository Implements
- End-to-end graph pipeline from raw CSVs to trained models and explainability outputs.
- Models: **GCN**, **GraphSAGE**, **GATv2**, and tabular **XGBoost baseline**.
- Strong class-imbalance handling:
  - weighted loss
  - focal loss (model-specific)
  - validation-driven threshold optimization
- Feature ablation:
  - all features (166)
  - local-only features (94)
- Hyperparameter tuning utility (`scripts/tune_hyperparams.py`)
- Explainability with GNNExplainer JSON outputs.

## 3) Repository Structure
- `src/gnn_eea/` core code (data, models, training, metrics, explainability, pipeline)
- `configs/default.yaml` default experiment config + tuned overrides
- `scripts/run_project.py` one-command end-to-end runner
- `scripts/tune_hyperparams.py` hyperparameter search
- `scripts/build_report_tex.py` report generation from latest metrics
- `notebooks/final_project_submission.ipynb` final submission notebook (fully executed outputs included)
- `notebooks/final_project_submission_cached.ipynb` duplicate of executed cached-results variant
- `notebooks/final_project_submission_end_to_end.ipynb` strict end-to-end retrain notebook (`FORCE_RETRAIN=True`)
- `reports/final_report.md|tex|pdf` final report artifacts
- `artifacts/` all generated outputs

## 4) Dataset Setup
Expected files in `data/elliptic/` (Kaggle Elliptic format):
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

If real dataset is unavailable, create sample-format data:
```bash
python3 scripts/make_sample_data.py
```

## 5) Installation
```bash
python3 -m pip install -r requirements.txt
```

## 6) How To Run (Primary)
### A) Full training + evaluation
```bash
PYTHONPATH=src python3 scripts/run_project.py --config configs/default.yaml --data-dir data/elliptic --output-dir artifacts --epochs 60 --device cpu
```

If using sample data:
```bash
PYTHONPATH=src python3 scripts/run_project.py --config configs/default.yaml --data-dir data/sample --output-dir artifacts --epochs 60 --device cpu
```

### B) Hyperparameter tuning (optional but recommended)
```bash
PYTHONPATH=src python3 scripts/tune_hyperparams.py --data-dir data/sample --feature-mode all --epochs 30 --max-trials 10 --device cpu
PYTHONPATH=src python3 scripts/tune_hyperparams.py --data-dir data/sample --feature-mode local --epochs 25 --max-trials 8 --device cpu
```

### C) Build LaTeX report from latest metrics
```bash
python3 scripts/build_report_tex.py
tectonic reports/final_report.tex --outdir reports
```

If `tectonic` is not in your `PATH`, run it by absolute path (example):
```bash
'/Users/akshatmittal/Documents/New project/tectonic' reports/final_report.tex --outdir reports
```

## 7) Methodology (Logic End-to-End)
1. **Load raw data**: merge classes/features by `txId`, map labels (`illicit=1`, `licit=0`, unknown `-1`).
2. **Graph construction**: build edge index from transaction links, optional undirected conversion.
3. **Temporal split**: train/val/test via time windows (`<=34`, `35-39`, `>=40`).
4. **Feature processing**: standardize using train-mask statistics only.
5. **Model training**:
   - residual + batchnorm message-passing networks
   - focal/CE loss with class weights
   - gradient clipping + cosine scheduler
6. **Threshold calibration**: select validation threshold to optimize recall/FPR trade-off.
7. **Evaluation**: compute PR-AUC, Macro-F1, Recall, FPR on test split.
8. **Explainability**: generate node-level top-feature/top-edge explanations.

## 8) Current Results (Latest Run)
`artifacts/metrics/results_summary.csv`

| feature_mode   | model     |   pr_auc |   macro_f1 |   illicit_recall |   false_positive_rate |   selected_threshold |
|:---------------|:----------|---------:|-----------:|-----------------:|----------------------:|---------------------:|
| all_features   | xgboost   |   0.981  |     0.9825 |           0.9375 |                0      |                0.5   |
| all_features   | gcn       |   0.7999 |     0.8037 |           0.875  |                0.0734 |                0.635 |
| all_features   | graphsage |   0.8252 |     0.884  |           0.8125 |                0.0226 |                0.68  |
| all_features   | gat       |   0.6767 |     0.7691 |           0.8125 |                0.0847 |                0.71  |
| local_only_94  | xgboost   |   0.9681 |     0.9081 |           0.9375 |                0.0282 |                0.05  |
| local_only_94  | gcn       |   0.6695 |     0.7176 |           0.8125 |                0.1243 |                0.56  |
| local_only_94  | graphsage |   0.8676 |     0.7621 |           0.875  |                0.1017 |                0.51  |
| local_only_94  | gat       |   0.7653 |     0.8057 |           0.75   |                0.0508 |                0.645 |

### Best models by feature mode
| feature_mode   | model   |   pr_auc |   macro_f1 |   illicit_recall |   false_positive_rate |
|:---------------|:--------|---------:|-----------:|-----------------:|----------------------:|
| all_features   | xgboost | 0.980978 |   0.982463 |           0.9375 |             0         |
| local_only_94  | xgboost | 0.968077 |   0.908095 |           0.9375 |             0.0282486 |

Interpretation:
- XGBoost is the strongest baseline on currently available data.
- GraphSAGE/GCN/GAT are significantly improved and competitive in several settings after tuning.
- Threshold calibration materially improves practical FPR/recall balance.

## 9) Submission Artifacts
Primary folder:
- `FINAL_DELIVERABLES/01_Executable_Code/`
- `FINAL_DELIVERABLES/02_Report/`
- `FINAL_DELIVERABLES/03_Results/`

Main zip:
- `FINAL_DELIVERABLES/01_Executable_Code/gnn_final_project_submission.zip`

Notebook recommendation:
- Submit `notebooks/final_project_submission.ipynb` (fully executed and ready).
- If evaluator explicitly wants retrain-on-run behavior inside notebook, use `notebooks/final_project_submission_end_to_end.ipynb`.

## 10) Reproducibility Checklist
- [x] Config tracked in `configs/default.yaml`
- [x] Metrics in CSV/JSON/Markdown
- [x] Trained model checkpoints saved
- [x] Explainability outputs saved
- [x] Executed notebook included
- [x] Report generated from latest metrics
