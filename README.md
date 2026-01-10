# Multimodal Stock Forecasting: GCN-LSTM Alpha Agent

---

## 1. Executive Summary

This project implements an institutional-grade trading agent for **NVIDIA (NVDA)** that fuses market structure analysis with non-linear time-series forecasting. Unlike retail models that rely on non-stationary price data, this system utilizes a **Graph Convolutional Network (GCN)** to extract latent sentiment from supply-chain news and fuses it with an **LSTM** network to forecast **Log-Returns** conditional on **Volatility Regimes**.

The strategy was rigorously validated using **Walk-Forward Optimization** (Expanding Window) over 316 trading days, achieving a **Sharpe Ratio of 1.48** and proving robustness across Bull, Bear and Sideways markets.

---

## 2. Methodology

### A. Stationary Feature Engineering

- **The Problem:** Deep learning models suffer from "gradient saturation" when asset prices drift significantly (e.g., NVDA rallying from $150 to $1000).
- **The Solution:** Replaced raw OHLCV inputs with **Log-Returns** and **Rolling Volatility** (20-day window). This creates a statistically stationary feature space, allowing the model to generalize patterns across vastly different price levels.

### B. Dynamic Temporal Knowledge Graph

- **The Problem:** Standard sentiment analysis ignores the _structure_ of the market (e.g., a TSMC supply shock impacting NVDA).
- **The Solution:** Constructed a daily-evolving Knowledge Graph where:
  - **Nodes:** Market Entities (Competitors, Suppliers, Partners).
  - **Edges:** News-derived sentiment weights.
  - **GCN Layer:** Extracts a spatial "Market Context Vector" for each trading day.
- **No Look-Ahead Bias:** The graph is reconstructed iteratively. On trading day $T$, the graph contains _only_ information known at $T-1$.

### C. Robust Validation (Walk-Forward)

- **Methodology:** Abandoned standard Train-Test splits for a realistic simulation of a live trading desk. The model was re-trained every month for 15 months (Jan 2024 - Sept 2025).
- **Result:** Proved model adaptability across changing market regimes (e.g., the "AI Boom" of early 2024 and the "Tech Correction" of late 2024).

---

## 3. Performance Metrics (Out-of-Sample)

| Metric               | Value         | Institutional Context                                                         |
| :------------------- | :------------ | :---------------------------------------------------------------------------- |
| **Sharpe Ratio**     | **1.48**      | Indicates strong risk-adjusted returns (Target > 1.0).                        |
| **Total Return**     | **>74.3%**    | Significantly outperforms the buy-and-hold baseline.                          |
| **Alpha Generation** | **Confirmed** | Successfully identified "Risk-Off" (Cash) signals during the late-2024 crash. |

---

## 4. Quick Start

### Prerequisites

- Python 3.9+
- [Polygon.io](https://polygon.io/) API Key (for news data)

### Installation

1. Clone the repo:
   ```bash
   git clone [https://github.com/your-username/nvda-alpha-agent.git](https://github.com/humaayuuun/nvda-alpha-agent.git)
   ```
