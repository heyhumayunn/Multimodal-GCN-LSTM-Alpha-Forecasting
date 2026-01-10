import torch
from torch_geometric.data import Data
import pandas as pd

class GraphBuilder:
    def __init__(self):
        self.relations = ['Ego', 'Competitor', 'Regulatory', 'Partner', 'Supplier', 'Customer', 'Neutral']
        self.rel_to_idx = {r: i for i, r in enumerate(self.relations)}

    def build_daily_graphs(self, news_df):
        """Converts processed news DataFrame into a list of PyG Data objects."""
        grouped = news_df.groupby('Date')
        dataset = []

        for date, group in grouped:
            num_news = len(group)
            num_nodes = 1 + num_news  # Node 0 is the Company (Ego), others are news items

            # Feature Matrix X: [Nodes, Features]
            # Features = One-hot encoding of Relation Type + Sentiment Score
            x = torch.zeros((num_nodes, len(self.relations) + 1), dtype=torch.float)

            # Set Ego Node
            x[0, self.rel_to_idx['Ego']] = 1.0
            x[0, -1] = 0.0 # Neutral sentiment for self initially

            # Set News Nodes
            for i, (_, row) in enumerate(group.iterrows()):
                node_idx = i + 1
                rel = row['Relation_Type']
                if rel in self.rel_to_idx:
                    x[node_idx, self.rel_to_idx[rel]] = 1.0
                x[node_idx, -1] = row['Sentiment_Score']

            # Create Star Graph Topology (All news connects to Ego)
            sources = torch.arange(1, num_nodes, dtype=torch.long)
            targets = torch.zeros(num_news, dtype=torch.long)
            edge_index = torch.stack([sources, targets], dim=0)

            data = Data(x=x, edge_index=edge_index)
            data.date = str(date)
            dataset.append(data)

        return dataset