import os
import pickle
import sys
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.data_loader import DataLoader
from src.trainer import WalkForwardTrainer
from src.models import InstitutionalTrader

CONFIG = {
    "TICKER": "NVDA",
    "START_DATE": "2022-09-01",
    "END_DATE": "2025-09-02",
    "WALK_FORWARD_START": "2024-01-01",
    "DATA_DIR": "data",
    "MODEL_FILE": "NVDA_Institutional_Model.pth",
    "GRAPH_FILE": "NVDA_Dynamic_Graph.pkl",
    "HIDDEN_DIM": 64,
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

PATHS = {
    "MODEL": os.path.join(CONFIG["DATA_DIR"], CONFIG["MODEL_FILE"]),
    "GRAPH": os.path.join(CONFIG["DATA_DIR"], CONFIG["GRAPH_FILE"])
}

def ensure_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_knowledge_graph(filepath: str) -> Dict[str, Data]:
    if not os.path.exists(filepath):
        # Check root for fallback
        filename = os.path.basename(filepath)
        if os.path.exists(filename):
            os.rename(filename, filepath)
        else:
            raise FileNotFoundError(f"Graph artifact not found: {filepath}")
    
    print(f"[INFO] Loading graph artifact: {filepath}")
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)

    registry = {}
    for item in raw_data:
        if isinstance(item, dict):
            x = torch.tensor(item['x'], dtype=torch.float)
            edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
            graph_obj = Data(x=x, edge_index=edge_index)
            date_key = pd.to_datetime(item['date']).strftime('%Y-%m-%d')
        else:
            graph_obj = item
            date_key = pd.to_datetime(item.date).strftime('%Y-%m-%d')
        
        registry[date_key] = graph_obj
        
    return registry

def fetch_market_data(ticker: str, start: str, end: str) -> Tuple[Dict, Dict]:
    print(f"[INFO] Fetching market data: {ticker}")
    loader = DataLoader(ticker)
    df = loader.fetch_financials(start, end)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()
    df['Target_Return'] = df['Log_Ret'].shift(-1)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    kpi_cols = ['Log_Ret', 'Volatility']
    df[kpi_cols] = scaler.fit_transform(df[kpi_cols])

    feature_map = {row['Date']: row[kpi_cols].values.astype(float) for _, row in df.iterrows()}
    target_map = {row['Date']: row['Target_Return'] for _, row in df.iterrows()}
    
    return feature_map, target_map

def verify_existing_model(model_path: str, data_sample: Dict[str, Any]) -> bool:
    if not os.path.exists(model_path):
        return False

    try:
        sample_graph = data_sample['graphs'][0]
        node_dim = sample_graph.x.shape[1]
        kpi_dim = data_sample['kpis'].shape[1]
        
        model = InstitutionalTrader(node_dim, kpi_dim, CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])
        model.load_state_dict(torch.load(model_path, map_location=CONFIG["DEVICE"]))
        model.eval()
        print(f"[INFO] Model verified. Dimensions: Node={node_dim}, KPI={kpi_dim}")
        return True
    
    except Exception as e:
        print(f"[WARN] Model mismatch or corruption: {e}")
        return False

def main():
    ensure_directory(CONFIG["DATA_DIR"])
    
    try:
        graph_map = load_knowledge_graph(PATHS["GRAPH"])
        feature_map, target_map = fetch_market_data(CONFIG["TICKER"], CONFIG["START_DATE"], CONFIG["END_DATE"])
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        sys.exit(1)
    
    valid_dates = sorted(list(set(feature_map.keys()) & set(graph_map.keys())))
    print(f"[INFO] Aligned samples: {len(valid_dates)}")
    
    data_package = {
        'graphs': [graph_map[d] for d in valid_dates],
        'kpis': np.array([feature_map[d] for d in valid_dates]),
        'targets': np.array([target_map[d] for d in valid_dates]),
        'dates': valid_dates
    }

    if verify_existing_model(PATHS["MODEL"], data_package):
        print(f"[INFO] Using existing weights: {PATHS['MODEL']}")
    else:
        print("[INFO] Starting Walk-Forward Validation...")
        trainer = WalkForwardTrainer(device=CONFIG["DEVICE"])
        
        results, final_model = trainer.run_validation(
            data_package, 
            CONFIG["WALK_FORWARD_START"], 
            CONFIG["END_DATE"]
        )
        
        if not results.empty:
            strategy_ret = np.where(results['Pred'] > 0, results['Actual'], 0)
            sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252)
            print(f"[INFO] Validation Sharpe: {sharpe:.3f}")

        if final_model:
            torch.save(final_model.state_dict(), PATHS["MODEL"])
            print(f"[INFO] Model saved to {PATHS['MODEL']}")
        else:
            print("[ERROR] Training produced no model.")
            sys.exit(1)

if __name__ == "__main__":
    main()