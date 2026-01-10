import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.models import InstitutionalTrader

# System Configuration
TICKER = "NVDA"
MODEL_FILENAME = "NVDA_Institutional_Model.pth"
GRAPH_FILENAME = "NVDA_Dynamic_Graph.pkl"

SEQUENCE_LENGTH = 5
NODE_FEATURE_DIM = 8 
GRAPH_NODES = 50 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def locate_artifact(filename: str) -> Optional[str]:
    """Resolves file paths between local root and data directory."""
    paths = [os.path.join("data", filename), filename]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = locate_artifact(MODEL_FILENAME)
GRAPH_PATH = locate_artifact(GRAPH_FILENAME)

# Data Services

@st.cache_resource
def load_graph_registry(filepath: Optional[str]) -> Dict[str, Data]:
    """Deserializes and indexes the knowledge graph."""
    if not filepath:
        return {}
    
    try:
        with open(filepath, 'rb') as f:
            raw_data = pickle.load(f)

        registry = {}
        for item in raw_data:
            if isinstance(item, dict):
                date_key = pd.to_datetime(item['date']).strftime('%Y-%m-%d')
                if 'x' in item and 'edge_index' in item:
                    x = torch.tensor(item['x'], dtype=torch.float)
                    edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
                    g_obj = Data(x=x, edge_index=edge_index)
                else:
                    g_obj = item
            else:
                date_key = pd.to_datetime(item.date).strftime('%Y-%m-%d')
                g_obj = item
                
            registry[date_key] = g_obj
        return registry
    except Exception as e:
        return {}

def get_last_trading_day(target_date: datetime.date) -> str:
    """Adjusts date to the preceding trading day."""
    dt = target_date - timedelta(days=1)
    while dt.weekday() > 4: 
        dt -= timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

def fetch_market_data(end_date: str) -> pd.DataFrame:
    """Fetches and normalizes OHLCV data."""
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    start_dt = end_dt - timedelta(days=120)
    
    df = yf.download(
        TICKER, 
        start=start_dt.strftime('%Y-%m-%d'), 
        end=end_dt.strftime('%Y-%m-%d'), 
        progress=False
    )
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = [c[0] for c in df.columns]
    
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Close'] = df['Close'].replace(0, np.nan).ffill()
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()

    return df.dropna().copy()

def load_model(device: torch.device) -> Optional[InstitutionalTrader]:
    """Initialize model architecture and load state dictionary."""
    if not MODEL_PATH:
        st.error(f"Error: Artifact '{MODEL_FILENAME}' not found.")
        return None

    model = InstitutionalTrader(node_dim=NODE_FEATURE_DIM, kpi_dim=2).to(device)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return None

# --- Application Logic ---

def main():
    st.set_page_config(page_title="Institutional Alpha Engine", layout="wide")
   
    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; }
        .stMetric { background-color: #0e1117; border: 1px solid #303030; padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

    st.title(f"{TICKER} Institutional Alpha Engine")
    st.markdown("##### GCN-LSTM Architecture | Multi-Modal Inference")

    st.sidebar.subheader("Configuration")
    target_date = st.sidebar.date_input("Forecast Date", datetime.now().date())
    target_str = target_date.strftime('%Y-%m-%d')
 
    graph_registry = load_graph_registry(GRAPH_PATH)
    
    if st.sidebar.button("Run Inference"):
        with st.spinner("Processing..."):
    
            decision_date = get_last_trading_day(target_date)
            market_df = fetch_market_data(decision_date)
        
            if market_df.empty or market_df['Date'].iloc[-1] > decision_date:
                 market_df = market_df[market_df['Date'] <= decision_date]

            if len(market_df) < SEQUENCE_LENGTH:
                st.error("Insufficient historical data for inference.")
                return

            scaler = StandardScaler()
            features = market_df[['Log_Ret', 'Volatility']].values
            scaled_features = scaler.fit_transform(features)
            
            seq_tensor = torch.tensor(
                scaled_features[-SEQUENCE_LENGTH:], 
                dtype=torch.float
            ).unsqueeze(0).to(DEVICE)

            #  Graph Context Retrieval
            if decision_date in graph_registry:
                graph_snapshot = graph_registry[decision_date]
            else:
                
                if graph_registry:
                    latest_key = max(graph_registry.keys())
                    graph_snapshot = graph_registry[latest_key]
                else:
                    st.warning("Graph artifact unavailable. Using synthetic initialization.")
                    x = torch.randn(GRAPH_NODES, NODE_FEATURE_DIM)
                    edge_index = torch.randint(0, GRAPH_NODES, (2, 100))
                    graph_snapshot = Data(x=x, edge_index=edge_index)

            graph_seq = [graph_snapshot] * SEQUENCE_LENGTH
          
            model = load_model(DEVICE)
            if not model: return

            with torch.no_grad():
                pred_log_ret = model(graph_seq, seq_tensor).item()

            current_price = market_df['Close'].iloc[-1]
            projected_price = current_price * np.exp(pred_log_ret)
            pct_change = (np.exp(pred_log_ret) - 1) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Reference Close", f"${current_price:.2f}", decision_date)
            c2.metric("Projected Price", f"${projected_price:.2f}", f"{pct_change:.2f}%")
            
            signal = "LONG" if pred_log_ret > 0 else "NEUTRAL"
            color = "#00d2be" if pred_log_ret > 0 else "#ff4b4b"
            c3.markdown(f"### Signal: <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)

            fig = go.Figure()
            plot_df = market_df.tail(60)
            
            fig.add_trace(go.Scatter(
                x=plot_df['Date'], y=plot_df['Close'],
                mode='lines', name='Historical',
                line=dict(color='#00d2be', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[target_str], y=[projected_price],
                mode='markers', name='Forecast',
                marker=dict(color=color, size=12, symbol='diamond')
            ))
            
            fig.add_trace(go.Scatter(
                x=[plot_df['Date'].iloc[-1], target_str],
                y=[plot_df['Close'].iloc[-1], projected_price],
                mode='lines', line=dict(color='gray', dash='dot', width=1),
                showlegend=False
            ))

            fig.update_layout(
                template="plotly_dark",
                height=400,
                title="Price Projection",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if MODEL_PATH:
                st.caption(f"Inference using weights: {os.path.basename(MODEL_PATH)}")

if __name__ == "__main__":
    main()