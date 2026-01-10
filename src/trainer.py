from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dateutil.relativedelta import relativedelta
from torch_geometric.data import Data

from .models import InstitutionalTrader

class WalkForwardTrainer:
    def __init__(self, device: torch.device = None, hidden_dim: int = 64, seq_len: int = 5):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.results = []
        
    def _create_sequences(self, graphs, kpis, targets, dates):
        sequences, seq_targets, seq_dates = [], [], []
        valid_indices = range(len(graphs) - self.seq_len)
        
        for i in valid_indices:
            graph_seq = graphs[i : i + self.seq_len]
            kpi_seq = kpis[i : i + self.seq_len]
            target_idx = i + self.seq_len
            
            sequences.append((graph_seq, kpi_seq))
            seq_targets.append(targets[target_idx])
            seq_dates.append(dates[target_idx])
            
        return sequences, torch.tensor(seq_targets, dtype=torch.float).to(self.device), seq_dates

    def _train_epoch(self, model, optimizer, criterion, sequences, targets):
        model.train()
        total_loss = 0.0
        
        for i, (inputs, target_val) in enumerate(zip(sequences, targets)):
            optimizer.zero_grad()
            graph_seq, kpi_data = inputs
            
            kpi_tensor = torch.tensor(kpi_data, dtype=torch.float).unsqueeze(0).to(self.device)
            target_tensor = target_val.view(1) 
            
            prediction = model(graph_seq, kpi_tensor)
            loss = criterion(prediction.squeeze(), target_tensor.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(sequences)

    def run_validation(self, data_package: Dict[str, Any], start_date_str: str, end_date_str: str):
        print(f"[INFO] Starting Walk-Forward Validation: {start_date_str} to {end_date_str}")
        
        current_date = pd.to_datetime(start_date_str)
        final_date = pd.to_datetime(end_date_str)
        full_dates_index = pd.to_datetime(data_package['dates'])
        latest_model = None  # Track the most recent model state
        
        while current_date < final_date:
            next_month = current_date + relativedelta(months=1)
            train_cutoff = current_date.strftime('%Y-%m-%d')
            
            # Temporal Masking
            train_mask = full_dates_index <= current_date
            test_mask = (full_dates_index > current_date) & (full_dates_index <= next_month)
            
            if not any(test_mask): break

            # Data Slicing
            def get_subset(mask):
                return (
                    [g for g, m in zip(data_package['graphs'], mask) if m],
                    data_package['kpis'][mask],
                    data_package['targets'][mask],
                    [d for d, m in zip(data_package['dates'], mask) if m]
                )

            train_g, train_k, train_y, train_d = get_subset(train_mask)
            test_g, test_k, test_y, test_d_raw = get_subset(test_mask)

            X_train, y_train, _ = self._create_sequences(train_g, train_k, train_y, train_d)
            X_test, y_test, dates_test = self._create_sequences(test_g, test_k, test_y, test_d_raw)
            
            if len(X_train) == 0: 
                current_date = next_month
                continue

            # Re-initialize model to avoid look-ahead bias
            node_dim = X_train[0][0][0].x.shape[1]
            kpi_dim = X_train[0][1].shape[1]
            
            model = InstitutionalTrader(node_dim, kpi_dim, self.hidden_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training Phase
            for _ in range(10):
                self._train_epoch(model, optimizer, criterion, X_train, y_train)
            
            latest_model = model # Save state

            # Inference Phase
            model.eval()
            with torch.no_grad():
                preds = []
                for inputs in X_test:
                    graph_seq, kpi_data = inputs
                    kpi_tensor = torch.tensor(kpi_data, dtype=torch.float).unsqueeze(0).to(self.device)
                    preds.append(model(graph_seq, kpi_tensor).item())
            
            for date, prediction, actual in zip(dates_test, preds, y_test.cpu().numpy()):
                self.results.append({'Date': date, 'Pred': prediction, 'Actual': actual})
            
            current_date = next_month

        return pd.DataFrame(self.results), latest_model