# Q-Orbit: Hybrid Quantum-Classical Portfolio Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45%2B-purple.svg)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A complete hybrid quantum-classical portfolio optimizer combining Quantum QAOA, FinBERT/VADER sentiment analysis, and Markowitz theory — all in an interactive Streamlit dashboard.**

---

## 🌟 What Q-Orbit Does

Q-Orbit selects and weights a portfolio of stocks using four distinct strategies you can compare side-by-side:

| Strategy | Technique | Description |
|---|---|---|
| **Classical Min-Var** | Markowitz CVXPY | Minimum variance portfolio |
| **Classical Max-Sharpe** | scipy + Markowitz | Maximum risk-adjusted return |
| **Quantum QAOA** | Qiskit Aer | Quantum stock selection + classical Markowitz refinement |
| **Hybrid Sentiment-Q** | VADER/FinBERT + QAOA | Sentiment-adjusted QUBO → Quantum selection |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/yourusername/q-orbit.git
cd q-orbit
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```bash
cp .env.template .env
```

Edit `.env`:
```
NEWS_API_KEY=your_newsapi_key      # Optional — VADER fallback works without it
IBM_QUANTUM_TOKEN=your_ibm_token  # Optional — uses local Aer simulator by default
```

### 3. Launch the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
q-orbit/
├── app.py                     # Streamlit web application (5 tabs)
├── benchmarks/
│   └── performance_comparison.py  # CLI 4-strategy benchmark script
├── src/
│   ├── classical/
│   │   └── baseline.py        # Markowitz optimizer (Min-Var, Max-Sharpe, Frontier)
│   ├── quantum/
│   │   ├── qaoa_optimizer.py  # QAOA + classical weight refinement
│   │   └── qubo_formulation.py # Portfolio → QUBO matrix encoding
│   ├── hybrid/
│   │   └── sentiment_quantum_optimizer.py  # Sentiment-adjusted QUBO → QAOA
│   ├── sentiment/
│   │   ├── analyzer.py        # FinBERT with persistent cache
│   │   ├── lightweight_analyzer.py  # VADER (no GPU needed)
│   │   ├── unified_analyzer.py      # Auto-selects FinBERT or VADER
│   │   ├── collector.py       # NewsAPI integration
│   │   └── news_wrapper.py    # Simplified news interface
│   └── utils/
│       ├── data_loader.py     # yfinance fetcher + caching
│       └── visualization.py   # Plotly chart helpers
├── data/                      # Cached prices, news, sentiment
├── requirements.txt
└── README.md
```

---

## 🖥️ App Tabs

| Tab | Contents |
|---|---|
| 📊 **Optimize** | Run any of the 4 strategies; view weights as table + pie chart |
| 📈 **Performance** | 6 metric cards (Sharpe, Sortino, Calmar, etc.) + True Efficient Frontier + Correlation Heatmap + Risk-Return Scatter |
| ⚖️ **Compare** | Optimized vs. Equal-Weight head-to-head with cumulative return charts |
| 🏆 **Benchmark** | Run all 4 strategies on current data; Sharpe + Sortino + Execution Time charts |
| ℹ️ **About** | Tech stack, usage guide, and configuration |

---

## 📊 Key Features

### 🧠 Quantum Portfolio Selection (QAOA)
- **QUBO formulation**: Portfolio selection encoded as a Quadratic Unconstrained Binary Optimization problem
- **Qiskit Aer simulation**: Runs on local quantum simulator (no IBM account needed)
- **Hybrid refinement**: QAOA selects the best stock subset → Classical Markowitz assigns optimal weights

### 💬 Sentiment-Aware Optimization
- **FinBERT**: Financial-domain transformer model for high-accuracy headline scoring
- **VADER fallback**: Lightweight lexicon model — works on Windows without GPU, no API key needed
- **Persistent cache**: FinBERT results saved to disk; repeat runs are instant
- **QUBO injection**: Sentiment scores directly adjust the QUBO cost matrix before quantum solving

### 📐 True Efficient Frontier
- Solves Markowitz min-variance for 50 target return levels to generate the **true parabolic frontier** (not a Monte Carlo scatter)

### 📄 PDF Report Export
- One-click download of a full portfolio analysis report including weights, all metrics, and embedded charts

---

## 📈 Performance Metrics

For every optimization run, Q-Orbit calculates:

| Metric | Formula | Meaning |
|---|---|---|
| **Sharpe Ratio** | (R - Rf) / σ | Risk-adjusted return (all volatility) |
| **Sortino Ratio** | (R - Rf) / σ_down | Risk-adjusted return (downside only) |
| **Calmar Ratio** | R / \|Max Drawdown\| | Return per unit of maximum loss |
| **Max Drawdown** | min((Vt - Vmax) / Vmax) | Worst peak-to-trough loss |
| **Expected Return** | μ × 252 | Annualised mean daily return |
| **Volatility** | σ × √252 | Annualised standard deviation |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Quantum** | Qiskit, Qiskit Aer |
| **Sentiment NLP** | FinBERT (ProsusAI/finbert), VADER, Transformers |
| **Finance** | yfinance, CVXPY, scipy |
| **Frontend** | Streamlit, Plotly, fpdf2 |
| **Data** | pandas 2.2+, numpy, NewsAPI |

---

## ⚙️ Configuration

All parameters are configurable from the Streamlit sidebar:

- **Stock Preset**: Tech Giants, Diversified, Financial, Healthcare, or Custom
- **Optimization Method**: Min Variance / Max Sharpe / Sentiment-Aware
- **Risk-Free Rate**: 0–10% (slider)
- **Data Source**: Real market data (yfinance) or synthetic simulation
- **Sentiment Lookback**: 7–30 days of news

---

## 🔬 How QAOA Works in Q-Orbit

```
1. Build QUBO matrix Q from returns covariance + expected returns
       ↕  (Hybrid mode: adjust Q diagonal using sentiment scores)
2. Encode Q into a QAOA quantum circuit (parameterised rotation gates)
3. Optimise QAOA params with COBYLA (classical loop)
4. Extract most-probable bitstring → selected stock indices
5. Run Markowitz Min-Variance on the selected subset → final weights
```

---

## 📦 Running the CLI Benchmark

```bash
# Compare all 4 strategies on synthetic data
python benchmarks/performance_comparison.py
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Qiskit / IBM Quantum** — quantum computing framework
- **ProsusAI / FinBERT** — financial sentiment model (Hugging Face)
- **Yahoo Finance** — free market data via yfinance
- **Streamlit** — interactive dashboard framework

---

**⭐ Star this repo if you find it useful!**
