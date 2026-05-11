#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def _format_results_table(df: pd.DataFrame, feature_mode: str) -> str:
    cols = [
        "model",
        "pr_auc",
        "macro_f1",
        "illicit_recall",
        "false_positive_rate",
        "selected_threshold",
    ]

    sub = df[df["feature_mode"] == feature_mode][cols].copy()
    if sub.empty:
        return "No results available for this feature setting."

    sub["model"] = sub["model"].astype(str)
    for col in cols[1:]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").map(lambda x: "-" if pd.isna(x) else f"{x:.4f}")

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & PR-AUC & Macro F1 & Illicit Recall & FPR & Threshold \\\\",
        "\\midrule",
    ]

    for _, r in sub.iterrows():
        lines.append(
            f"{r['model']} & {r['pr_auc']} & {r['macro_f1']} & {r['illicit_recall']} & {r['false_positive_rate']} & {r['selected_threshold']} \\\\"  # noqa: E501
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Results for feature mode: {feature_mode.replace('_', '\\_')}}}",
        "\\end{table}",
    ]

    return "\n".join(lines)


def build_tex(root: Path) -> Path:
    csv_path = root / "artifacts" / "metrics" / "results_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    all_table = _format_results_table(df, "all_features")
    local_table = _format_results_table(df, "local_only_94")

    tex = r'''\\documentclass[12pt,a4paper]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{lmodern}
\\usepackage{hyperref}
\\usepackage{booktabs}
\\usepackage{float}
\\usepackage{enumitem}
\\setlist{noitemsep}
\\title{Final Project Report: Leveraging Graph Neural Networks to Uncover Illicit Financial Networks}
\\author{Akshat Mittal (240086) \\and Aaditya Mukherjee (240006) \\and Daksh Gupta (240317) \\and Nitish Choudhary (240710)}
\\date{\\today}
\\begin{document}
\\maketitle

\\section{Overall Project Summary}
This project addresses illicit transaction detection in cryptocurrency payment flows using graph representation learning. Instead of treating transactions as independent tabular rows, we model the transaction ecosystem as a graph and benchmark multiple Graph Neural Network (GNN) architectures against a tabular baseline.

\\section{Dataset Details}
\\begin{itemize}
\\item Dataset: Elliptic Bitcoin transaction dataset
\\item Nodes: transactions
\\item Edges: directed payment flows
\\item Time steps: 49 temporal snapshots
\\item Feature dimensions: 166 (94 local + 72 aggregated structural features)
\\item Class imbalance: highly skewed with very few illicit labels and many unknown labels
\\end{itemize}

\\subsection{Class Mapping Used}
\\begin{itemize}
\\item Illicit: \\texttt{1}
\\item Licit: \\texttt{0}
\\item Unknown/unlabeled: \\texttt{-1} (excluded from supervised loss, retained in graph structure)
\\end{itemize}

\\section{Data Preprocessing}
\\begin{enumerate}
\\item Read raw CSV files: classes, features, and edge list.
\\item Merge node features with class labels by transaction ID.
\\item Map transaction IDs to contiguous node indices.
\\item Build \\texttt{edge\\_index} for graph neural models.
\\item Convert to undirected graph (configurable).
\\item Create temporal train/validation/test masks: train $\\leq 34$, validation $35$--$39$, test $\\geq 40$.
\\item Standardize features using train-mask statistics only.
\\end{enumerate}

\\section{Feature Extraction and Weighting Methods}
\\begin{itemize}
\\item Full-feature experiments: all available node features (166).
\\item Ablation experiments: local-only first 94 features.
\\item Class imbalance handling: inverse-frequency class weights and focal loss.
\\item Validation-based threshold tuning to optimize recall/FPR trade-off.
\\end{itemize}

\\section{Libraries Used}
Python, NumPy, Pandas, SciPy, PyTorch, PyTorch Geometric, Scikit-learn, XGBoost, Matplotlib, Seaborn, and Jupyter.

\\section{Model Architectures}
\\subsection{GCN}
Two-layer graph convolution network with dropout and final linear classifier.
\\subsection{GraphSAGE}
Two-layer neighborhood aggregation model for robust inductive graph representation.
\\subsection{GAT}
Attention-based graph model to assign dynamic importance to neighbors.
\\subsection{Baseline (XGBoost)}
Tabular model trained on node features only to compare against graph-aware learning.

\\section{Optimization and Training}
\\begin{itemize}
\\item Optimizer: AdamW
\\item Early stopping: validation PR-AUC
\\item Loss: focal loss (default) with cross-entropy fallback
\\item Hyperparameters controlled via \\texttt{configs/default.yaml}
\\end{itemize}

\\section{Evaluation Metrics (KPIs)}
\\begin{itemize}
\\item PR-AUC (primary)
\\item Macro F1-score
\\item Recall for illicit class
\\item False Positive Rate (FPR)
\\item FPR reduction vs tabular baseline
\\end{itemize}

\\section{Results}
The following tables are generated from \\texttt{artifacts/metrics/results\\_summary.csv}.

''' + all_table + r'''

''' + local_table + r'''

\\section{Explainability}
Node-level explanations are generated using GNNExplainer and saved under \\texttt{artifacts/explain/*.json}. These files include top influential features and edges for suspicious-node predictions.

\\section{Improvements Over Legacy Approaches}
\\begin{itemize}
\\item Captures multi-hop transaction behavior directly via message passing
\\item Reduces manual feature engineering effort
\\item Improves minority-class detection with better threshold calibration
\\end{itemize}

\\section{Limitations}
\\begin{itemize}
\\item Performance sensitivity to split strategy and class-label sparsity
\\item GAT and full-batch training can be computationally expensive on very large graphs
\\item Explainability quality can vary by node and model confidence
\\end{itemize}

\\section{Conclusion}
Graph-based learning is better aligned with real-world coordinated financial fraud patterns than isolated tabular classification. In this project, GNN models provide a more contextual detection framework and can outperform tabular baselines on minority-focused KPIs when carefully tuned for class imbalance.

\\section{Sources}
\\begin{itemize}
\\item Elliptic dataset (Kaggle)
\\item Kipf \\& Welling, 2017: Semi-Supervised Classification with Graph Convolutional Networks
\\item Hamilton et al., 2017: Inductive Representation Learning on Large Graphs (GraphSAGE)
\\item Velickovic et al., 2018: Graph Attention Networks
\\item Ying et al., 2019: GNNExplainer
\\end{itemize}

\\end{document}
'''
    tex = re.sub(r"\\\\(?=[A-Za-z])", r"\\", tex)
    tex = tex.replace(r"\\_", r"\_")
    tex = tex.replace(r"\\&", r"\&")

    tex_path = root / "reports" / "final_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tex_path = build_tex(root)
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
