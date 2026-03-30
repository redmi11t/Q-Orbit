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
| **Classical Max-Sharpe** | scipy + Markowitz | Maximum risk-adjusted return (global search with `differential_evolution`) |
| **Quantum QAOA** | Qiskit Aer / IBM QPU | Quantum stock selection + classical Markowitz weight refinement |
| **Hybrid Sentiment-Q** | VADER/FinBERT + QAOA | Sentiment-adjusted QUBO → Quantum selection |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/yourusername/q-orbit.git
cd q-orbit
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key      # Optional — VADER fallback works without it
IBM_QUANTUM_TOKEN=your_ibm_token   # Optional — uses local Aer simulator by default
```

> **Note:** Never commit your `.env` or `.streamlit/secrets.toml` files.
> Both are already listed in `.gitignore`.

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
│   │   ├── qubo_formulation.py # Portfolio → QUBO matrix encoding
│   │   └── ibm_backend.py     # IBM Quantum Runtime connector with fallback
│   ├── hybrid/
│   │   └── sentiment_quantum_optimizer.py  # Sentiment-adjusted QUBO → QAOA
│   ├── sentiment/
│   │   ├── analyzer.py        # FinBERT with persistent batched cache
│   │   ├── lightweight_analyzer.py  # VADER (no GPU needed, Windows-compatible)
│   │   ├── unified_analyzer.py      # Auto-selects FinBERT or VADER
│   │   ├── collector.py       # NewsAPI integration with caching
│   │   └── news_wrapper.py    # Simplified news interface
│   └── utils/
│       ├── data_loader.py     # yfinance fetcher + local CSV caching
│       └── visualization.py   # Plotly chart helpers
├── data/                      # Cached prices, news, sentiment (git-ignored)
├── requirements.txt
└── README.md
```

---

## 🖥️ App Tabs

| Tab | Contents |
|---|---|
| 📊 **Optimize** | Run any of the 4 strategies; view weights as table + pie chart |
| 📈 **Performance** | 6 metric cards + True Efficient Frontier + Correlation Heatmap + Risk-Return Scatter |
| ⚖️ **Compare** | Optimized vs. Equal-Weight head-to-head with cumulative return charts |
| 🏆 **Benchmark** | Run all 4 strategies on current data; Sharpe + Sortino + Execution Time charts |
| ℹ️ **About** | Tech stack, usage guide, and configuration |

---

## 📊 Key Features

### 🧠 Quantum Portfolio Selection (QAOA)
- **QUBO formulation**: Portfolio selection encoded as a Quadratic Unconstrained Binary Optimization problem with mathematically correct budget penalty (`2×` off-diagonal coefficients)
- **Qiskit Aer simulation**: Statevector simulation for fast, noise-free COBYLA optimization; shot-based decoding for final result
- **IBM QPU support**: Connect to real IBM Quantum hardware via `qiskit-ibm-runtime` with automatic channel fallback (`ibm_quantum` → `ibm_cloud`)
- **Hybrid refinement**: QAOA selects the best stock subset → Classical Markowitz assigns optimal weights
- **Hard qubit limit**: Enforces ≤10 stocks for QAOA to prevent exponential scaling issues

### 💬 Sentiment-Aware Optimization
- **FinBERT** (`ProsusAI/finbert`): Financial-domain transformer model for high-accuracy headline scoring
- **VADER fallback**: Lightweight lexicon model with financial keyword boosts — works on Windows without GPU
- **Persistent batched cache**: Sentiment results saved to disk; writes batched every 10 entries (not per-article) for performance
- **QUBO injection**: Sentiment scores directly adjust the QUBO cost matrix diagonal before quantum solving

### 📐 True Efficient Frontier
- Solves Markowitz min-variance for 50 target return levels using a **temporary optimizer instance** — never corrupts the current portfolio state

### 📄 PDF Report Export
- One-click download of a full portfolio analysis report with weights, all metrics, and embedded matplotlib charts (no kaleido dependency)

### 🛡️ Robustness
- **Solver failure detection**: CVXPY status checked before using results — no silent NaN weight propagation
- **Positive-definite covariance**: Sample data uses factor model with ≥2 factors + regularization (`1e-8 × I`)
- **Real data cache**: 1-hour TTL so intraday re-runs don't serve stale market data
- **Demo mode warning**: Clearly labelled when synthetic (non-real) data is active

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
| **Quantum** | Qiskit ≥0.45, Qiskit-Aer ≥0.13, Qiskit-IBM-Runtime ≥0.20 |
| **Sentiment NLP** | FinBERT (`ProsusAI/finbert`), VADER (`vaderSentiment`), HuggingFace Transformers |
| **Optimization** | CVXPY ≥1.4, SciPy (`differential_evolution` + SLSQP) |
| **Finance Data** | yfinance ≥0.2.32, NewsAPI (`newsapi-python`), feedparser |
| **Frontend** | Streamlit ≥1.29, Plotly ≥5.18, fpdf2 ≥2.7, matplotlib ≥3.8 |
| **Core** | pandas ≥2.1, numpy ≥1.24, python-dotenv |

---

## ⚙️ Configuration

All parameters are configurable from the Streamlit sidebar:

| Parameter | Options |
|---|---|
| **Stock Preset** | Tech Giants, Diversified, Financial, Healthcare, or Custom |
| **Optimization Method** | Min Variance / Max Sharpe / Sentiment-Aware / Quantum QAOA / Hybrid |
| **Risk-Free Rate** | 0–10% (slider) |
| **Data Source** | Real market data (yfinance) or synthetic demo |
| **IBM Backend** | Any backend name from your IBM Quantum account |
| **Quantum Budget** | Number of stocks to select (controls QUBO budget constraint) |
| **Sentiment Lookback** | 7–30 days of financial news |

---

## 🔬 How QAOA Works in Q-Orbit

```
1. Fetch historical returns (yfinance) & news (NewsAPI)
         ↓
2. Build QUBO matrix Q:
   - Risk term:    λ_risk  × xᵀ Σ x
   - Return term: -λ_ret  × μᵀ x
   - Budget term:  λ_B    × (Σᵢ xᵢ - B)²   ← 2× off-diagonal (fixed)
         ↓
3. [Hybrid mode] Adjust Q diagonal with sentiment scores
         ↓
4. Encode Q into parameterised QAOA circuit (p layers of cost + mixer)
         ↓
5. COBYLA optimises gate angles (statevector — no shot noise)
         ↓
6. Sample final circuit → decode most-probable bitstring (LSB-first corrected)
         ↓
7. Run Markowitz Min-Variance on selected stocks → final portfolio weights
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
- **NewsAPI** — financial news headlines

---

**⭐ Star this repo if you find it useful!**
