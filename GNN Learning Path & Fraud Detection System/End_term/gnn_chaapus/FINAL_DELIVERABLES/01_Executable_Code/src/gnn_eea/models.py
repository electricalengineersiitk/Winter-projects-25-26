from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv


class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2 + residual)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.lin(h2)


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2 + residual)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.lin(h2)


class GATNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.elu(h2 + residual)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.lin(h2)


def build_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int = 2,
    dropout: float = 0.3,
    heads: int = 4,
) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "gcn":
        return GCNNet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if normalized in {"graphsage", "sage"}:
        return GraphSAGENet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if normalized == "gat":
        return GATNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads=heads,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model name: {name}")
