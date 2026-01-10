from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

class InstitutionalTrader(nn.Module):
    """
    Hybrid GCN-LSTM architecture for spatiotemporal financial forecasting.
    Fuses unstructured graph embeddings with structured time-series data.
    """

    def __init__(self, node_dim: int, kpi_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gnn = GCNConv(node_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + kpi_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, graph_list: List[Data], kpi_tensor: torch.Tensor) -> torch.Tensor:
        device = kpi_tensor.device
        graph_embeddings = []

        # Spatial feature extraction
        for graph in graph_list:
            x, edge_index = graph.x.to(device), graph.edge_index.to(device)
            
            x = torch.tanh(self.gnn(x, edge_index))
            
            # Global pooling: Node features -> Graph vector
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=device)
            graph_embeddings.append(global_mean_pool(x, batch_vec))

        # Temporal alignment and fusion
        graph_seq = torch.stack(graph_embeddings, dim=1) 
        fusion_tensor = torch.cat((graph_seq, kpi_tensor), dim=2)

        # Sequence modeling
        lstm_out, _ = self.lstm(fusion_tensor)
        
        return self.head(lstm_out[:, -1, :])