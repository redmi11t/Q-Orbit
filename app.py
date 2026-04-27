"""
Q-Orbit Portfolio Optimization Web App
Streamlit interface for hybrid quantum-classical portfolio optimization
"""

import streamlit as st
import sys
import os
# Phase 4/5 Fix #19: Use an absolute path so the app works regardless of the
# current working directory (not just when run from project root).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta
import os
import io
from dotenv import load_dotenv
from fpdf import FPDF, XPos, YPos

from classical.baseline import MarkowitzOptimizer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Q-Orbit Portfolio Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(43, 43, 60, 0.05);
        padding: 5px 10px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e293b !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Ultra-Premium Glassmorphism Metric Cards */
    .metric-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        z-index: 0;
    }

    .metric-content {
        position: relative;
        z-index: 1;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
    }
    
    .metric-label {
        font-size: 1.05rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    /* Vibrant Gradients & Glowing Shadows */
    .metric-blue { 
        background: linear-gradient(135deg, #0284c7 0%, #3b82f6 100%); 
        box-shadow: 0 10px 30px -5px rgba(59, 130, 246, 0.5);
    }
    .metric-blue:hover { box-shadow: 0 20px 40px -5px rgba(59, 130, 246, 0.7); }

    .metric-purple { 
        background: linear-gradient(135deg, #7e22ce 0%, #a855f7 100%); 
        box-shadow: 0 10px 30px -5px rgba(168, 85, 247, 0.5);
    }
    .metric-purple:hover { box-shadow: 0 20px 40px -5px rgba(168, 85, 247, 0.7); }

    .metric-green { 
        background: linear-gradient(135deg, #047857 0%, #10b981 100%); 
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.5);
    }
    .metric-green:hover { box-shadow: 0 20px 40px -5px rgba(16, 185, 129, 0.7); }

    .metric-orange { 
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); 
        box-shadow: 0 10px 30px -5px rgba(249, 115, 22, 0.5);
    }
    .metric-orange:hover { box-shadow: 0 20px 40px -5px rgba(249, 115, 22, 0.7); }
    
    /* Premium Header */
    .portfolio-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    .portfolio-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, rgba(0,0,0,0) 50%);
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================
# Header
# =====================================
st.markdown("""
    <div class="portfolio-header">
        <h1 style="font-size: 2.8rem; font-weight: 700; margin-bottom: 15px; letter-spacing: -0.02em;">🚀 Q-Orbit Optimizer</h1>
        <p style="font-size: 1.25rem; font-weight: 300; color: #94a3b8; margin: 0; letter-spacing: 0.01em;">
            Hybrid Quantum-Classical Portfolio Optimization with Sentiment Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

# =====================================
# Sidebar Configuration
# =====================================
st.sidebar.header("⚙️ Configuration")

# Stock Selection
st.sidebar.subheader("📊 Stock Selection")

# Predefined stock lists
stock_presets = {
    "Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA'],
    "Diversified (Default)": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'JPM', 'V', 'JNJ'],
    "Financial Sector": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
    "Healthcare": ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'LLY'],
    "Custom": []
}

preset_choice = st.sidebar.selectbox(
    "Select Stock Preset",
    options=list(stock_presets.keys()),
    index=1
)

if preset_choice == "Custom":
    custom_tickers = st.sidebar.text_input(
        "Enter tickers (comma-separated)",
        placeholder="AAPL, MSFT, GOOGL"
    )
    selected_tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
else:
    selected_tickers = stock_presets[preset_choice]

st.sidebar.write(f"**{len(selected_tickers)} stocks selected**")

# Optimization Parameters
st.sidebar.subheader("🎯 Optimization Settings")

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.5
) / 100

optimization_method = st.sidebar.radio(
    "Optimization Method",
    [
        "Minimum Variance",
        "Maximum Sharpe Ratio",
        "Sentiment-Aware (Phase 2)",
        "QAOA (Quantum)",
        "Hybrid Sentiment-Quantum",
    ],
    help="Choose the portfolio optimization strategy. Quantum methods are limited to ≤10 stocks."
)

# ── Fix 2: Quantum stock-count guard shown in sidebar ────────────────────
QUANTUM_METHODS = {"QAOA (Quantum)", "Hybrid Sentiment-Quantum"}
if optimization_method in QUANTUM_METHODS:
    if len(selected_tickers) > 10:
        st.sidebar.error(
            f"⚛️ Quantum methods support **≤ 10 stocks** (classical simulation is "
            f"exponential). You have **{len(selected_tickers)}** selected. "
            "Please choose a smaller preset or switch to a classical method."
        )
    else:
        st.sidebar.info(
            f"⚛️ QAOA will run on **{len(selected_tickers)} qubits** "
            "(classical Aer simulator).  This may take 1–3 minutes."
        )
    with st.sidebar.expander("⚙️ QAOA Settings", expanded=False):
        qaoa_layers = st.slider("QAOA Layers (p)", min_value=1, max_value=3, value=1)
        qaoa_max_iter = st.slider("COBYLA Iterations", min_value=10, max_value=100, value=30)
        qaoa_budget = st.slider(
            "Stocks to Select (Budget)",
            min_value=1,
            max_value=min(10, len(selected_tickers)),
            value=min(5, len(selected_tickers))
        )

        # ── IBM Quantum backend selector ──────────────────────────────────────
        st.markdown("---")
        st.markdown("**🖥️ Quantum Backend**")
        qaoa_backend_mode = st.radio(
            "Run on:",
            options=["simulator", "ibm_real"],
            format_func=lambda x: "🖥️ Local Simulator (fast, free)" if x == "simulator" else "☁️ IBM Quantum (real hardware)",
            index=0,
            key="qaoa_backend_mode",
        )

        ibm_backend_name_input = os.getenv("IBM_QUANTUM_BACKEND", "ibm_brisbane")
        if qaoa_backend_mode == "ibm_real":
            ibm_backend_name_input = st.text_input(
                "IBM Backend name",
                value=os.getenv("IBM_QUANTUM_BACKEND", "ibm_brisbane"),
                help="e.g. ibm_brisbane, ibm_kyoto, ibmq_qasm_simulator"
            )
            _ibm_token = os.getenv("IBM_QUANTUM_TOKEN", "")
            if not _ibm_token:
                st.warning(
                    "⚠️ IBM_QUANTUM_TOKEN not set in .env — "
                    "will fall back to local simulator."
                )
            else:
                st.success("✅ IBM token found — will connect to real QPU")
                st.caption(
                    "⏱️ Note: real hardware jobs may queue for 30–60 min. "
                    "Each COBYLA iteration submits one circuit job."
                )
        # ────────────────────────────────────────────────────────────────────
else:
    qaoa_layers = 1
    qaoa_max_iter = 30
    qaoa_budget = min(5, len(selected_tickers))
    qaoa_backend_mode = "simulator"
    ibm_backend_name_input = os.getenv("IBM_QUANTUM_BACKEND", "ibm_brisbane")

# Data Settings
st.sidebar.subheader("📅 Data Configuration")

use_real_data = st.sidebar.checkbox("Use Real Market Data (yfinance)", value=False)

if use_real_data:
    lookback_days = st.sidebar.slider(
        "Historical Data (days)",
        min_value=90,
        max_value=730,
        value=365,
        step=30
    )
else:
    lookback_days = 500  # Simulated data default

# Sentiment Settings — shown for both sentiment methods
if optimization_method in ("Sentiment-Aware (Phase 2)", "Hybrid Sentiment-Quantum"):
    st.sidebar.subheader("🗞️ Sentiment Settings")
    news_api_key = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
    
    if not news_api_key:
        st.sidebar.info(
            "ℹ️ No NEWS_API_KEY found — will try free Yahoo Finance RSS feeds as fallback."
        )
    
    sentiment_weight = st.sidebar.slider(
        "Sentiment Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    news_lookback = st.sidebar.slider(
        "News Lookback (days)",
        min_value=1,
        max_value=30,
        value=7
    )
else:
    news_api_key = ""
    sentiment_weight = 0.3
    news_lookback = 7

# =====================================
# Helper Functions
# =====================================

@st.cache_data
def generate_sample_returns(tickers, n_days=500):
    """Generate realistic sample returns for demo purposes"""
    # Phase 1 Fix #2: Use default_rng and ensure ≥2 factors to avoid rank-1
    # correlation matrices when n_stocks is odd or very small.
    rng = np.random.default_rng(42)

    n_stocks = len(tickers)

    # Realistic parameters
    expected_returns = rng.uniform(0.10, 0.35, n_stocks)
    volatilities = rng.uniform(0.18, 0.60, n_stocks)

    daily_returns = expected_returns / 252
    daily_vols = volatilities / np.sqrt(252)

    # Create a well-conditioned correlation matrix using at least 2 factors
    # so the covariance is always positive-definite, even for small/odd portfolios.
    n_factors = max(2, n_stocks // 2)
    L = rng.standard_normal((n_stocks, n_factors))
    cov_raw = L @ L.T
    # Normalise to a proper correlation matrix
    d = np.sqrt(np.diag(cov_raw))
    corr_matrix = cov_raw / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)

    cov_matrix = np.outer(daily_vols, daily_vols) * corr_matrix
    # Small regularization for numerical safety
    cov_matrix += 1e-8 * np.eye(n_stocks)

    returns_array = rng.multivariate_normal(daily_returns, cov_matrix, n_days)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    returns = pd.DataFrame(returns_array, index=dates, columns=tickers)

    return returns

@st.cache_data(ttl=3600)  # Phase 3 Fix #13: Cache real data for max 1 hour
def fetch_real_returns(tickers, days=365):
    """Fetch real market data using yfinance"""
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        prices = data['Close']
        # Fix #7: yf.download returns a Series (not DataFrame) for a single ticker.
        # Convert to DataFrame before calling .columns / .dropna(axis=1, ...).
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        prices = prices.dropna(axis=1, how='all')  # Remove tickers with no data
        
        if prices.empty or len(prices.columns) < 2:
            return None
        
        returns = prices.pct_change().dropna()
        return returns
    
    except Exception as e:
        st.error(f"Error fetching real data: {e}")
        return None

def create_weights_pie_chart(weights, tickers):
    """Create an interactive pie chart for portfolio weights"""
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights
    })
    weights_df = weights_df[weights_df['Weight'] > 0.01].sort_values('Weight', ascending=False)
    
    fig = px.pie(
        weights_df,
        values='Weight',
        names='Ticker',
        title='Portfolio Allocation',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_efficient_frontier_plot(returns, optimizer):
    """Create efficient frontier visualization using the true mathematical frontier"""
    # Generate the true efficient frontier by solving min-variance at each target return
    n_portfolios = 60
    risks_mc, returns_mc, labels_mc = [], [], []

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # True efficient frontier via optimizer
    try:
        ef_risks, ef_returns, _ = optimizer.generate_efficient_frontier(returns, n_points=n_portfolios)
        use_true_ef = len(ef_risks) > 2
    except Exception:
        use_true_ef = False

    # Monte Carlo cloud for context (500 random portfolios)
    # Fix #25: use a local, seeded Generator instead of mutating global np.random state.
    _mc_rng = np.random.default_rng(0)
    for _ in range(500):
        w = _mc_rng.random(len(returns.columns))
        w /= w.sum()
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        port_sharpe = (port_return - risk_free_rate) / port_vol
        risks_mc.append(port_vol)
        returns_mc.append(port_return)
        labels_mc.append(port_sharpe)

    # Optimal portfolio point
    perf = optimizer.get_performance_summary()
    opt_return = float(perf['Expected Return'].strip('%')) / 100
    opt_vol = float(perf['Volatility'].strip('%')) / 100

    fig = go.Figure()

    # Random portfolios (background cloud)
    fig.add_trace(go.Scatter(
        x=risks_mc,
        y=returns_mc,
        mode='markers',
        marker=dict(
            size=5,
            color=labels_mc,
            colorscale='Viridis',
            showscale=True,
            opacity=0.5,
            colorbar=dict(
                title="Sharpe Ratio",
                x=1.02,
                len=0.75,
                thickness=15
            )
        ),
        name='Random Portfolios',
        hovertemplate='Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
    ))

    # True efficient frontier line
    if use_true_ef:
        fig.add_trace(go.Scatter(
            x=ef_risks,
            y=ef_returns,
            mode='lines',
            line=dict(color='royalblue', width=3),
            name='Efficient Frontier',
            hovertemplate='Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
        ))

    # Optimal portfolio star
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        name='Optimal Portfolio',
        text=['Optimal'],
        textposition='bottom center',
        textfont=dict(color='red', size=12, family='Arial Black'),
        hovertemplate='Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title='Efficient Frontier (True Mathematical Curve)',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        height=520,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=13)
        ),
        margin=dict(r=100, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

# =====================================
# PDF Report Generator  (matplotlib-based — no kaleido needed)
# =====================================

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── Colour palette ────────────────────────────────────────────────────────────
_PURPLE   = '#667eea'
_PINK     = '#f5576c'
_GREEN    = '#43e97b'
_TEAL     = '#38f9d7'
_PALETTE  = ['#667eea', '#f5576c', '#43e97b', '#38f9d7', '#fa709a',
             '#fee140', '#30cfd0', '#a18cd1', '#fda085', '#96fbc4']


def _fig_to_bytes(fig):
    """Save a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _make_pie_chart(weights, tickers):
    """Donut chart of portfolio weights."""
    ws = pd.Series(weights.values, index=weights.index)
    ws = ws[ws > 0.01].sort_values(ascending=False)
    colors = _PALETTE[:len(ws)]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')
    wedges, texts, autotexts = ax.pie(
        ws.values, labels=ws.index, autopct='%1.1f%%',
        colors=colors, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        pctdistance=0.78
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color('white')
        t.set_fontweight('bold')
    ax.set_title('Portfolio Allocation', fontsize=13, fontweight='bold', pad=16)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_efficient_frontier(returns, weights, risk_free_rate):
    """Scatter cloud + optimal point."""
    mean_ret = returns.mean() * 252
    cov      = returns.cov() * 252
    np.random.seed(0)
    n = len(returns.columns)
    risks, rets, sharpes = [], [], []
    for _ in range(500):
        w = np.random.random(n); w /= w.sum()
        r = float(np.dot(w, mean_ret))
        v = float(np.sqrt(w @ cov.values @ w))
        risks.append(v); rets.append(r)
        sharpes.append((r - risk_free_rate) / v)

    opt_r = float(np.dot(weights.values, mean_ret))
    opt_v = float(np.sqrt(weights.values @ cov.values @ weights.values))

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    sc = ax.scatter(risks, rets, c=sharpes, cmap='viridis', s=18, alpha=0.55)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio', shrink=0.8)
    ax.scatter([opt_v], [opt_r], s=220, c='red', marker='*',
               zorder=5, label='Optimal Portfolio')
    ax.set_xlabel('Volatility (Risk)', fontsize=10)
    ax.set_ylabel('Expected Return', fontsize=10)
    ax.set_title('Efficient Frontier', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_cumulative_chart(opt_cum, eq_cum=None):
    """Line chart of cumulative returns."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    ax.plot(opt_cum.index, opt_cum.values, color=_PURPLE, linewidth=2,
            label='Optimized Portfolio')
    ax.fill_between(opt_cum.index, 1, opt_cum.values,
                    alpha=0.15, color=_PURPLE)
    if eq_cum is not None:
        ax.plot(eq_cum.index, eq_cum.values, color=_PINK, linewidth=2,
                linestyle='--', label='Equal-Weight')
        ax.fill_between(eq_cum.index, 1, eq_cum.values,
                        alpha=0.10, color=_PINK)
    ax.axhline(1, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Cumulative Return', fontsize=10)
    title = 'Growth Comparison' if eq_cum is not None else 'Portfolio Growth Over Time'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_bar_chart(labels, values, colors, title, ylabel, fmt=None):
    """Generic two-bar comparison chart."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='white')
    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, values):
        txt = fmt(v) if fmt else f'{v:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in values) * 0.02,
                txt, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    # Phase 3 Fix #15: Handle negative Sharpe: use a symmetric y-axis so bars
    # pointing down (negative) are still readable instead of inverting the axis.
    max_abs = max(abs(v) for v in values)
    ax.set_ylim(-max_abs * 1.3 if any(v < 0 for v in values) else 0,
                max_abs * 1.22)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def generate_pdf_report(results, selected_tickers, risk_free_rate, preset_choice, optimization_method):
    """Generate a comprehensive PDF report covering all 3 analysis sections."""
    weights   = results['weights']
    perf      = results['performance']
    returns   = results['returns']

    # ── Pre-compute comparison data ──────────────────────────────────────────
    equal_weights      = pd.Series([1 / len(weights)] * len(weights),
                                   index=weights.index)
    opt_returns_series = (returns * weights).sum(axis=1)
    eq_returns_series  = (returns * equal_weights).sum(axis=1)

    opt_annual_return  = opt_returns_series.mean() * 252
    opt_annual_vol     = opt_returns_series.std()  * np.sqrt(252)
    opt_sharpe         = (opt_annual_return - risk_free_rate) / opt_annual_vol

    eq_annual_return   = eq_returns_series.mean() * 252
    eq_annual_vol      = eq_returns_series.std()  * np.sqrt(252)
    eq_sharpe          = (eq_annual_return - risk_free_rate) / eq_annual_vol

    opt_cumulative     = (1 + opt_returns_series).cumprod()
    eq_cumulative      = (1 + eq_returns_series).cumprod()

    # ── Build chart PNGs with matplotlib ─────────────────────────────────────
    pie_bytes      = _make_pie_chart(weights, selected_tickers)
    ef_bytes       = _make_efficient_frontier(returns, weights, risk_free_rate)
    cum_bytes      = _make_cumulative_chart(opt_cumulative)
    vol_bytes      = _make_bar_chart(
        ['Optimized', 'Equal-Weight'],
        [opt_annual_vol * 100, eq_annual_vol * 100],
        [_PURPLE, _PINK],
        'Risk Comparison (Lower is Better)',
        'Volatility (%)',
        fmt=lambda v: f'{v:.1f}%'
    )
    sharpe_bytes   = _make_bar_chart(
        ['Optimized', 'Equal-Weight'],
        [max(opt_sharpe, 0.001), max(eq_sharpe, 0.001)],
        [_GREEN, _TEAL],
        'Risk-Adjusted Return (Higher is Better)',
        'Sharpe Ratio',
        fmt=lambda v: f'{v:.2f}'
    )
    comp_cum_bytes = _make_cumulative_chart(opt_cumulative, eq_cumulative)

    # ── Build PDF ────────────────────────────────────────────────────────────
    def safe_text(text):
        """Strip characters outside Latin-1 range (e.g. emoji) so Helvetica won't crash."""
        return ''.join(c if ord(c) <= 255 else '' for c in text).strip()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    PURPLE = (102, 126, 234)
    DARK   = (30, 30, 50)
    GRAY   = (100, 100, 100)
    WHITE  = (255, 255, 255)

    def section_title(text):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(*PURPLE)
        pdf.cell(0, 10, safe_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(*PURPLE)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_text_color(*DARK)

    def two_col_table(headers, rows, col_widths):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(*PURPLE)
        pdf.set_text_color(*WHITE)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, border=1, fill=True, align='C')
        pdf.ln()
        pdf.set_font('Helvetica', '', 10)
        for i, row in enumerate(rows):
            pdf.set_fill_color(245, 245, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(*DARK)
            for val, w in zip(row, col_widths):
                pdf.cell(w, 7, str(val), border=1, fill=True, align='C')
            pdf.ln()
        pdf.ln(4)

    def embed_image(img_bytes, w=180):
        if img_bytes:
            tmp = io.BytesIO(img_bytes)
            x   = (210 - w) / 2
            pdf.image(tmp, x=x, w=w)
        pdf.ln(4)

    # ────────── PAGE 1: COVER ──────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*PURPLE)
    pdf.rect(0, 0, 210, 60, 'F')
    pdf.set_font('Helvetica', 'B', 26)
    pdf.set_text_color(*WHITE)
    pdf.set_y(15)
    pdf.cell(0, 12, 'Q-Orbit Portfolio Report', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 13)
    pdf.cell(0, 8, 'Hybrid Quantum-Classical Portfolio Optimization', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(70)
    pdf.set_text_color(*DARK)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d  %H:%M")}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    # Config summary box
    pdf.set_fill_color(240, 242, 255)
    pdf.set_draw_color(*PURPLE)
    pdf.rect(20, pdf.get_y(), 170, 55, 'FD')
    pdf.set_xy(25, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'Configuration Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(25)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Portfolio Preset  :  {preset_choice}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(25)
    pdf.cell(0, 6, f'Optimization Method  :  {optimization_method}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(25)
    pdf.cell(0, 6, f'Risk-Free Rate  :  {risk_free_rate:.2%}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(25)
    pdf.cell(0, 6, f'Stocks ({len(selected_tickers)})  :  {", ".join(selected_tickers)}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(25)

    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, 'Report contains: Portfolio Optimization  |  Performance Analysis  |  Strategy Comparison', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ────────── PAGE 2: PORTFOLIO OPTIMIZATION ────────────────────────────
    pdf.add_page()
    section_title('1. Portfolio Optimization')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, f'Method: {optimization_method}   |   Portfolio: {preset_choice}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    weights_sorted = weights.sort_values(ascending=False)
    rows = [(t, f'{w:.2%}') for t, w in weights_sorted.items() if w > 0.01]
    two_col_table(['Ticker', 'Allocation Weight'], rows, [95, 95])

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, 'Portfolio Allocation', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(pie_bytes, w=160)

    # ────────── PAGE 3: PERFORMANCE ANALYSIS ─────────────────────────────
    pdf.add_page()
    section_title('2. Performance Analysis')

    metric_rows = [
        ('Expected Return', perf['Expected Return']),
        ('Volatility',      perf['Volatility']),
        ('Sharpe Ratio',    perf['Sharpe Ratio']),
        ('Max Drawdown',    perf.get('Max Drawdown', 'N/A')),
    ]
    two_col_table(['Metric', 'Value'], metric_rows, [100, 90])

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, 'Efficient Frontier', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(ef_bytes, w=175)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Cumulative Portfolio Returns', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(cum_bytes, w=175)

    # ────────── PAGE 4: STRATEGY COMPARISON ──────────────────────────────
    pdf.add_page()
    section_title('3. Strategy Comparison - Optimized vs Equal-Weight')

    improvement_return = (opt_annual_return - eq_annual_return) / abs(eq_annual_return) * 100
    improvement_vol    = (eq_annual_vol - opt_annual_vol) / eq_annual_vol * 100
    improvement_sharpe = (opt_sharpe - eq_sharpe) / abs(eq_sharpe) * 100

    comp_rows = [
        ('Expected Return',   f'{opt_annual_return:.2%}', f'{eq_annual_return:.2%}', f'{improvement_return:+.1f}%'),
        ('Volatility (Risk)', f'{opt_annual_vol:.2%}',    f'{eq_annual_vol:.2%}',    f'{improvement_vol:+.1f}%'),
        ('Sharpe Ratio',      f'{opt_sharpe:.2f}',        f'{eq_sharpe:.2f}',        f'{improvement_sharpe:+.1f}%'),
    ]
    two_col_table(['Metric', 'Optimized', 'Equal-Weight', 'Improvement'], comp_rows, [55, 45, 45, 45])

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, 'Volatility Comparison', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(vol_bytes, w=140)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Sharpe Ratio Comparison', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(sharpe_bytes, w=140)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Cumulative Returns Comparison', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    embed_image(comp_cum_bytes, w=175)

    return bytes(pdf.output())





# =====================================
# Main Application
# =====================================

if len(selected_tickers) < 2:
    st.warning("⚠️ Please select at least 2 stocks to optimize a portfolio.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Optimize", "📈 Performance", "⚖️ Compare", "🏆 Benchmark", "ℹ️ About"])

with tab1:
    st.header("Portfolio Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Selected Stocks")
        st.info(f"**Portfolio:** {preset_choice}")
        for ticker in selected_tickers:
            st.write(f"• {ticker}")
        
        optimize_button = st.button("🚀 Optimize Portfolio", type="primary", width='stretch')
    
    with col1:
        # Fix #15: Only re-run optimization when the button is pressed.
        # Previously the entire block ran on every Streamlit rerun that found
        # 'optimization_results' in session_state (tab switch, slider move, etc.),
        # causing needless repeated data fetching and solver calls.
        if optimize_button:
            with st.spinner("⏳ Fetching data and optimizing..."):
                # Phase 3 Fix #5: Show prominent DEMO MODE banner when using synthetic data.
                if not use_real_data:
                    st.warning(
                        "⚠️ **DEMO MODE** — Using *synthetic* random data. "
                        "The tickers shown are labels only; results do NOT reflect "
                        "real market performance. Enable **'Use Real Market Data'** "
                        "in the sidebar for realistic results."
                    )

                # Get returns data
                if use_real_data:
                    returns = fetch_real_returns(selected_tickers, lookback_days)
                    if returns is None:
                        st.error("Failed to fetch real data. Using sample data instead.")
                        returns = generate_sample_returns(selected_tickers, lookback_days)
                else:
                    returns = generate_sample_returns(selected_tickers, lookback_days)
                
                # Optimize
                optimizer = MarkowitzOptimizer(risk_free_rate=risk_free_rate)
                
                if optimization_method == "Minimum Variance":
                    weights = optimizer.optimize_min_variance(returns)

                elif optimization_method == "Maximum Sharpe Ratio":
                    weights = optimizer.optimize_max_sharpe(returns)

                elif optimization_method == "Sentiment-Aware (Phase 2)":
                    from sentiment.lightweight_analyzer import LightweightSentimentAnalyzer
                    from sentiment.news_wrapper import NewsCollector

                    news_api_key = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
                    sentiment_scores = {}
                    article_counts = {}

                    if news_api_key:
                        try:
                            collector = NewsCollector(api_key=news_api_key)
                            vader = LightweightSentimentAnalyzer()
                            for ticker in selected_tickers:
                                articles = collector.fetch_news(ticker=ticker, days_back=news_lookback, max_articles=10)
                                if articles:
                                    scores = [vader.analyze_text(a.get('title', ''))['sentiment_value'] for a in articles]
                                    sentiment_scores[ticker] = float(np.mean(scores))
                                    article_counts[ticker] = len(articles)
                                else:
                                    sentiment_scores[ticker] = 0.0
                                    article_counts[ticker] = 0
                            st.info(f"📡 Live news sentiment fetched for {len(sentiment_scores)} tickers.")
                        except Exception as e:
                            st.warning(f"News API failed ({e}). Trying Yahoo Finance RSS…")
                            news_api_key = ""

                    if not news_api_key:
                        # ── Fix 3: Yahoo Finance RSS fallback (replaces static headlines) ──
                        try:
                            import feedparser
                            feedparser_ok = True
                        except ImportError:
                            feedparser_ok = False
                        vader2 = LightweightSentimentAnalyzer() if 'vader' not in locals() else vader
                        rss_scores = {}
                        rss_missing = []
                        for ticker in selected_tickers:
                            score = None
                            if feedparser_ok:
                                try:
                                    rss_url = (
                                        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                                        f"?s={ticker}&region=US&lang=en-US"
                                    )
                                    feed = feedparser.parse(rss_url)
                                    headlines = [e.title for e in feed.entries[:5] if e.get("title")]
                                    if headlines:
                                        sc = [vader2.analyze_text(h)['sentiment_value'] for h in headlines]
                                        score = float(np.mean(sc))
                                        article_counts[ticker] = len(headlines)
                                except Exception:
                                    pass
                            if score is not None:
                                rss_scores[ticker] = score
                            else:
                                rss_missing.append(ticker)

                        if rss_scores:
                            sentiment_scores = rss_scores
                            miss_note = (f"  No RSS data for: {', '.join(rss_missing)}." if rss_missing else "")
                            st.info(f"📡 Yahoo Finance RSS sentiment fetched for {len(rss_scores)} ticker(s).{miss_note}")
                        else:
                            st.warning(
                                "⚠️ No live sentiment data available (no NEWS_API_KEY and Yahoo Finance "
                                "RSS returned no results). Sentiment adjustment **disabled** — "
                                "running plain Max Sharpe."
                            )
                        # ────────────────────────────────────────────────────────────

                    # ── Sentiment Analysis Results Table ─────────────────────────
                    if sentiment_scores:
                        def _sentiment_label(score):
                            if score > 0.05:
                                return "🟢 Positive"
                            elif score < -0.05:
                                return "🔴 Negative"
                            else:
                                return "🟡 Neutral"

                        sent_rows = [
                            {
                                "Ticker": t,
                                "Avg Sentiment Score": round(s, 4),
                                "Label": _sentiment_label(s),
                                "Articles Analyzed": article_counts.get(t, "N/A"),
                            }
                            for t, s in sentiment_scores.items()
                        ]
                        st.subheader("📰 Sentiment Analysis Results")
                        st.table(pd.DataFrame(sent_rows))
                    # ─────────────────────────────────────────────────────────────

                    # Apply sentiment (or skip if no data)
                    raw_weights = optimizer.optimize_max_sharpe(returns)
                    if sentiment_scores:
                        s_series = pd.Series(sentiment_scores)
                        adjusted = raw_weights * (
                            1 + sentiment_weight
                            * s_series.reindex(raw_weights.index, fill_value=0.0)
                        )
                        adjusted = adjusted.clip(lower=0)
                        weights = adjusted / adjusted.sum()
                        st.success("✅ Sentiment-aware weights computed!")
                    else:
                        weights = raw_weights
                    optimizer.weights = weights
                    optimizer._calculate_performance(returns, weights)

                # ── Fix 1: QAOA (Quantum) ─────────────────────────────────────────────
                elif optimization_method == "QAOA (Quantum)":
                    from quantum.qaoa_optimizer import QAOAOptimizer
                    if len(returns.columns) > 10:
                        st.error(
                            f"⚛️ QAOA supports ≤10 stocks (you have {len(returns.columns)}). "
                            "Please choose a smaller preset."
                        )
                        st.stop()
                    if qaoa_backend_mode == "ibm_real":
                        st.info("☁️ Connecting to IBM Quantum... (this may take a moment)")
                    qaoa = QAOAOptimizer(
                        num_layers=qaoa_layers,
                        max_iterations=qaoa_max_iter,
                        backend_mode=qaoa_backend_mode,
                        ibm_backend_name=ibm_backend_name_input,
                    )
                    if qaoa_backend_mode == "ibm_real" and not qaoa.using_ibm:
                        err_detail = getattr(qaoa, 'ibm_error', None) or "Unknown error"
                        st.warning(
                            f"⚠️ IBM connection failed — running on local Aer simulator instead.\n\n"
                            f"**Reason:** `{err_detail}`\n\n"
                            f"💡 **Tip:** Try changing the IBM Backend name to `ibmq_qasm_simulator` (IBM cloud simulator, free & no queue)."
                        )
                    try:
                        sel_q, w_arr, qaoa_info = qaoa.optimize(returns, budget=qaoa_budget)
                    except ValueError as ve:
                        st.error(str(ve))
                        st.stop()
                    if qaoa_info.get("fallback_warning"):  # Fix 6
                        st.warning(f"⚠️ QAOA Fallback: {qaoa_info['fallback_warning']}")
                    weights = pd.Series(w_arr, index=returns.columns)
                    optimizer.weights = weights
                    optimizer._calculate_performance(returns, weights)
                    st.info(
                        f"⚛️ QAOA selected **{len(sel_q)} stocks**: {', '.join(sel_q)} "
                        f"in {qaoa_info['iterations']} iterations. "
                        f"Backend: {qaoa_info.get('backend', 'N/A')}"
                    )

                # ── Fix 1: Hybrid Sentiment-Quantum ─────────────────────────────────
                elif optimization_method == "Hybrid Sentiment-Quantum":
                    from hybrid.sentiment_quantum_optimizer import SentimentQuantumOptimizer
                    if len(returns.columns) > 10:
                        st.error(
                            f"⚛️ Hybrid Sentiment-Quantum supports ≤10 stocks "
                            f"(you have {len(returns.columns)}). Please choose a smaller preset."
                        )
                        st.stop()

                    # Cache the optimizer so FinBERT is only loaded once per
                    # session — re-creating it on every Streamlit rerun was the
                    # cause of the app crash (OOM / timeout from repeated heavy
                    # model loading after the QAOA step completed).
                    @st.cache_resource(show_spinner=False)
                    def _get_hybrid_optimizer(
                        _news_api_key, _qaoa_layers, _qaoa_max_iterations,
                        _sentiment_weight, _backend_mode, _ibm_backend_name
                    ):
                        return SentimentQuantumOptimizer(
                            news_api_key=_news_api_key,
                            qaoa_layers=_qaoa_layers,
                            qaoa_max_iterations=_qaoa_max_iterations,
                            sentiment_weight=_sentiment_weight,
                            backend_mode=_backend_mode,
                            ibm_backend_name=_ibm_backend_name,
                        )

                    hybrid = _get_hybrid_optimizer(
                        st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", "")),
                        qaoa_layers,
                        qaoa_max_iter,
                        sentiment_weight,
                        qaoa_backend_mode,
                        ibm_backend_name_input,
                    )
                    try:
                        sel_h, w_arr_h, h_info = hybrid.optimize(
                            returns=returns,
                            tickers=returns.columns.tolist(),
                            budget=qaoa_budget,
                            days_back=news_lookback,
                        )
                    except ValueError as ve:
                        st.error(str(ve))
                        st.stop()
                    if h_info.get("fallback_warning"):  # Fix 6
                        st.warning(f"⚠️ Hybrid Fallback: {h_info['fallback_warning']}")
                    weights = pd.Series(w_arr_h, index=returns.columns)
                    optimizer.weights = weights
                    optimizer._calculate_performance(returns, weights)
                    st.info(
                        f"⚛️ Hybrid selected **{len(sel_h)} stocks**: {', '.join(sel_h)}.  "
                        f"News: {h_info.get('news_count', 0)} articles.  "
                        f"Backend: {h_info.get('backend', 'N/A')}."
                    )

                    # ── Sentiment Analysis Results Table (Hybrid) ─────────────
                    h_sent_summary = h_info.get('sentiment_summary')
                    if h_sent_summary is not None and not h_sent_summary.empty:
                        def _sentiment_label_h(score):
                            if score > 0.05:
                                return "🟢 Positive"
                            elif score < -0.05:
                                return "🔴 Negative"
                            else:
                                return "🟡 Neutral"

                        h_sent_rows = [
                            {
                                "Ticker": ticker,
                                "Avg Sentiment Score": round(
                                    float(h_sent_summary.loc[ticker, 'avg_sentiment']), 4
                                ),
                                "Label": _sentiment_label_h(
                                    float(h_sent_summary.loc[ticker, 'avg_sentiment'])
                                ),
                                "Articles Analyzed": int(
                                    h_sent_summary.loc[ticker, 'article_count']
                                ),
                            }
                            for ticker in h_sent_summary.index
                        ]
                        st.subheader("📰 Sentiment Analysis Results")
                        st.table(pd.DataFrame(h_sent_rows))
                    # ─────────────────────────────────────────────────────────
                
                perf = optimizer.get_performance_summary()
                
                # Store in session state
                st.session_state.optimization_results = {
                    'weights': weights,
                    'performance': perf,
                    'returns': returns,
                    'optimizer': optimizer
                }
            
            st.success("✅ Optimization Complete!")

        # Display Results — read from session_state so results persist
        # across tab switches without re-running the solver (Fix #15 cont.)
        if 'optimization_results' in st.session_state:
            _res = st.session_state.optimization_results
            st.subheader("Optimal Portfolio Weights")
            
            weights_sorted = _res['weights'].sort_values(ascending=False)
            weights_data = []
            
            for ticker, weight in weights_sorted.items():
                if weight > 0.01:
                    weights_data.append({
                        'Ticker': ticker,
                        'Weight': f"{weight:.2%}",
                        'Bar': '█' * int(weight * 100)
                    })
            
            st.table(pd.DataFrame(weights_data))
            
            # Pie Chart
            st.plotly_chart(
                create_weights_pie_chart(_res['weights'], list(_res['weights'].index)),
                width='stretch'
            )

with tab2:
    st.header("Performance Analysis")
    
    if 'optimization_results' not in st.session_state:
        st.info("👈 Please run optimization in the 'Optimize' tab first.")
    else:
        results = st.session_state.optimization_results
        perf = results['performance']
        returns = results['returns']
        optimizer = results['optimizer']
        
        # Performance Metrics — Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card metric-blue">
                    <div class="metric-content">
                        <p class="metric-label">Expected Return</p>
                        <h2 class="metric-value">{perf["Expected Return"]}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card metric-purple">
                    <div class="metric-content">
                        <p class="metric-label">Volatility</p>
                        <h2 class="metric-value">{perf["Volatility"]}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card metric-green">
                    <div class="metric-content">
                        <p class="metric-label">Sharpe Ratio</p>
                        <h2 class="metric-value">{perf["Sharpe Ratio"]}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card metric-orange">
                    <div class="metric-content">
                        <p class="metric-label">Max Drawdown</p>
                        <h2 class="metric-value">{perf["Max Drawdown"]}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance Metrics — Row 2 (new ratios)
        col5, col6, col7 = st.columns(3)
        with col5:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%); box-shadow: 0 10px 30px -5px rgba(6,182,212,0.5); border-radius:16px; padding:24px; text-align:center; border:1px solid rgba(255,255,255,0.3);">
                    <div class="metric-content">
                        <p class="metric-label">Sortino Ratio</p>
                        <h2 class="metric-value">{perf.get("Sortino Ratio", "N/A")}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #b45309 0%, #d97706 100%); box-shadow: 0 10px 30px -5px rgba(217,119,6,0.5); border-radius:16px; padding:24px; text-align:center; border:1px solid rgba(255,255,255,0.3);">
                    <div class="metric-content">
                        <p class="metric-label">Calmar Ratio</p>
                        <h2 class="metric-value">{perf.get("Calmar Ratio", "N/A")}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col7:
            sharpe_val = float(perf['Sharpe Ratio'])
            quality = "Excellent" if sharpe_val > 2 else "Good" if sharpe_val > 1 else "Fair" if sharpe_val > 0.5 else "Poor"
            color = "#10b981" if sharpe_val > 1 else "#f97316" if sharpe_val > 0.5 else "#ef4444"
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); box-shadow: 0 10px 30px -5px rgba(124,58,237,0.5); border-radius:16px; padding:24px; text-align:center; border:1px solid rgba(255,255,255,0.3);">
                    <div class="metric-content">
                        <p class="metric-label">Risk Quality</p>
                        <h2 class="metric-value" style="font-size:1.8rem;">{quality}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Efficient Frontier
        st.subheader("Efficient Frontier")
        with st.spinner("Computing true efficient frontier..."):
            st.plotly_chart(
                create_efficient_frontier_plot(returns, optimizer),
                width='stretch'
            )
        
        # Cumulative Returns
        st.subheader("Cumulative Portfolio Returns")
        
        portfolio_returns = (returns * results['weights']).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.update_layout(
            title='Portfolio Growth Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=400,
            template='plotly_white',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, width='stretch')

        # ── Correlation Heatmap ──────────────────────────────────────────────
        st.subheader("Asset Correlation Heatmap")
        corr = returns.corr()
        corr_fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale='RdBu',
            zmid=0,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
        ))
        corr_fig.update_layout(
            title='Asset Correlation Matrix',
            height=500,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(corr_fig, width='stretch')

        # ── Risk-Return Scatter ──────────────────────────────────────────────
        st.subheader("Individual Asset Risk vs Return")
        annual_ret = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        asset_sharpe = (annual_ret - risk_free_rate) / annual_vol

        rr_fig = go.Figure()
        rr_fig.add_trace(go.Scatter(
            x=annual_vol.values,
            y=annual_ret.values,
            mode='markers+text',
            marker=dict(
                size=16,
                color=asset_sharpe.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe'),
                line=dict(width=1, color='white')
            ),
            text=annual_ret.index.tolist(),
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
        ))
        rr_fig.update_layout(
            title='Individual Asset Risk vs Return',
            xaxis_title='Annual Volatility (Risk)',
            yaxis_title='Annual Return',
            height=450,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(rr_fig, width='stretch')

with tab3:
    st.header("Strategy Comparison")
    
    if 'optimization_results' not in st.session_state:
        st.info("👈 Please run optimization in the 'Optimize' tab first.")
    else:
        results = st.session_state.optimization_results
        returns = results['returns']
        opt_weights = results['weights']
        
        # Phase 3 Fix #6: Use returns.columns (not selected_tickers) for the
        # equal-weight baseline so real-data runs that drop tickers stay aligned.
        eq_index = returns.columns
        equal_weights = pd.Series(
            [1 / len(eq_index)] * len(eq_index), index=eq_index
        )
        
        # Calculate metrics for both strategies
        opt_returns_series = (returns * opt_weights).sum(axis=1)
        eq_returns_series = (returns * equal_weights).sum(axis=1)
        
        opt_annual_return = opt_returns_series.mean() * 252
        opt_annual_vol = opt_returns_series.std() * np.sqrt(252)
        opt_sharpe = (opt_annual_return - risk_free_rate) / opt_annual_vol
        
        eq_annual_return = eq_returns_series.mean() * 252
        eq_annual_vol = eq_returns_series.std() * np.sqrt(252)
        eq_sharpe = (eq_annual_return - risk_free_rate) / eq_annual_vol
        
        # Comparison Table
        st.subheader("Optimized vs Equal-Weight Portfolio")
        
        comparison_data = {
            'Metric': ['Expected Return', 'Volatility (Risk)', 'Sharpe Ratio'],
            'Optimized': [
                f"{opt_annual_return:.2%}",
                f"{opt_annual_vol:.2%}",
                f"{opt_sharpe:.2f}"
            ],
            'Equal-Weight': [
                f"{eq_annual_return:.2%}",
                f"{eq_annual_vol:.2%}",
                f"{eq_sharpe:.2f}"
            ],
            # Fix #18: Guard all divisions against zero to prevent ZeroDivisionError
            # when synthetic data produces near-zero equal-weight returns.
            'Improvement': [
                f"{((opt_annual_return - eq_annual_return) / (abs(eq_annual_return) or 1e-10) * 100):.1f}%",
                f"{((eq_annual_vol - opt_annual_vol) / (eq_annual_vol or 1e-10) * 100):.1f}%",
                f"{((opt_sharpe - eq_sharpe) / (abs(eq_sharpe) or 1e-10) * 100):.1f}%"
            ]
        }
        
        st.table(pd.DataFrame(comparison_data))
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Volatility Comparison")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Optimized', 'Equal-Weight'],
                    y=[opt_annual_vol * 100, eq_annual_vol * 100],
                    marker_color=['#667eea', '#f5576c'],
                    text=[f"{opt_annual_vol:.2%}", f"{eq_annual_vol:.2%}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title='Risk Comparison (Lower is Better)',
                yaxis_title='Volatility (%)',
                height=350,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Sharpe Ratio Comparison")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Optimized', 'Equal-Weight'],
                    y=[opt_sharpe, eq_sharpe],
                    marker_color=['#43e97b', '#38f9d7'],
                    text=[f"{opt_sharpe:.2f}", f"{eq_sharpe:.2f}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title='Risk-Adjusted Return (Higher is Better)',
                yaxis_title='Sharpe Ratio',
                height=350,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Cumulative comparison
        st.subheader("Cumulative Returns Comparison")
        
        opt_cumulative = (1 + opt_returns_series).cumprod()
        eq_cumulative = (1 + eq_returns_series).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=opt_cumulative.index,
            y=opt_cumulative.values,
            mode='lines',
            name='Optimized Portfolio',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=eq_cumulative.index,
            y=eq_cumulative.values,
            mode='lines',
            name='Equal-Weight Portfolio',
            line=dict(color='#f5576c', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Growth Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=400,
            template='plotly_white',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, width='stretch')

with tab4:
    st.header("Strategy Benchmarking")
    st.markdown("Run all available strategies on the current dataset and compare performance metrics side-by-side.")
    
    if 'optimization_results' not in st.session_state:
        st.info("👈 Please run optimization in the 'Optimize' tab first to load the dataset.")
    else:
        # Get data from optimization results
        returns = st.session_state.optimization_results['returns']
        
        if st.button("🚀 Run Multi-Strategy Benchmark"):
            import time
            from classical.baseline import MarkowitzOptimizer as _MO
            from quantum.qaoa_optimizer import QAOAOptimizer as _QAOA
            from hybrid.sentiment_quantum_optimizer import SentimentQuantumOptimizer as _HQ
            
            # Helper: extract raw numeric performance dict from MarkowitzOptimizer
            def _bench_entry(strategy_name, opt_instance, weights_series, duration):
                opt_instance._calculate_performance(returns, weights_series)
                p = opt_instance.performance  # raw float dict
                return {
                    'Strategy': strategy_name,
                    'Sharpe':   round(p['sharpe_ratio'], 3),
                    'Sortino':  round(p['sortino_ratio'], 3),
                    'Return':   f"{p['annual_return']:.2%}",
                    'Risk':     f"{p['annual_volatility']:.2%}",
                    'Max DD':   f"{p['max_drawdown']:.2%}",
                    'Time (s)': round(duration, 3)
                }
            
            benchmark_results = []
            
            try:
                # ── Classical Min-Variance ───────────────────────────────────
                with st.spinner("Benchmarking Classical Min Variance..."):
                    opt = _MO(risk_free_rate=risk_free_rate)
                    t0 = time.time()
                    w = opt.optimize_min_variance(returns)
                    benchmark_results.append(_bench_entry("Classical Min-Var", opt, w, time.time() - t0))
                
                # ── Classical Max-Sharpe ─────────────────────────────────────
                with st.spinner("Benchmarking Classical Max Sharpe..."):
                    opt = _MO(risk_free_rate=risk_free_rate)
                    t0 = time.time()
                    w = opt.optimize_max_sharpe(returns)
                    benchmark_results.append(_bench_entry("Classical Max-Sharpe", opt, w, time.time() - t0))
                
                # ── Quantum QAOA ─────────────────────────────────────────────
                # Fix #10: Guard against >10 qubits before running quantum methods
                if len(returns.columns) > 10:
                    st.warning(
                        f"⚛️ Skipping Quantum QAOA & Hybrid — portfolio has "
                        f"**{len(returns.columns)} stocks** (>10 qubit limit). "
                        "Reduce the portfolio to ≤10 stocks to include quantum benchmarks."
                    )
                else:
                    with st.spinner("Benchmarking Quantum QAOA (may take ~1 min)..."):
                        qaoa = _QAOA(num_layers=1)
                        t0 = time.time()
                        target_budget = min(5, len(returns.columns))
                        _, w_arr, _ = qaoa.optimize(returns, budget=target_budget)
                        w_series = pd.Series(w_arr, index=returns.columns)
                        perf_calc = _MO(risk_free_rate=risk_free_rate)
                        benchmark_results.append(_bench_entry("Quantum QAOA", perf_calc, w_series, time.time() - t0))
                    
                    # ── Hybrid Sentiment-Quantum ─────────────────────────────────
                    with st.spinner("Benchmarking Hybrid Sentiment-Quantum (may take ~2 min)..."):
                        hybrid = _HQ(news_api_key=st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", "")))
                        t0 = time.time()
                        _, w_arr_h, _ = hybrid.optimize(
                            returns=returns,
                            tickers=returns.columns.tolist(),
                            budget=min(5, len(returns.columns))
                        )
                        w_series_h = pd.Series(w_arr_h, index=returns.columns)
                        perf_calc_h = _MO(risk_free_rate=risk_free_rate)
                        benchmark_results.append(_bench_entry("Hybrid Sentiment-Q", perf_calc_h, w_series_h, time.time() - t0))
                
                bench_df = pd.DataFrame(benchmark_results)
                st.session_state.benchmark_df = bench_df
                st.success("✅ Benchmark completed!")
            
            except Exception as e:
                st.error(f"Benchmark failed: {str(e)}")
                st.exception(e)
        
        if 'benchmark_df' in st.session_state:
            df = st.session_state.benchmark_df
            
            # Sharpe bar chart
            st.subheader("Sharpe Ratio Comparison")
            fig = px.bar(df, x='Strategy', y='Sharpe', color='Strategy',
                        text_auto='.3f', title="Risk-Adjusted Performance (Sharpe Ratio)",
                        color_discrete_sequence=px.colors.qualitative.G10)
            fig.update_layout(showlegend=False, template='plotly_white', height=380)
            st.plotly_chart(fig, width='stretch')
            
            # Sortino bar chart
            col_a, col_b = st.columns(2)
            with col_a:
                fig_s = px.bar(df, x='Strategy', y='Sortino', color='Strategy',
                               text_auto='.3f', title="Sortino Ratio (Downside Risk)",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_s.update_layout(showlegend=False, template='plotly_white', height=300)
                st.plotly_chart(fig_s, width='stretch')
            with col_b:
                fig_t = px.bar(df, x='Strategy', y='Time (s)', color='Strategy',
                               text_auto='.2f', title="Execution Time (seconds)",
                               color_discrete_sequence=px.colors.qualitative.Safe)
                fig_t.update_layout(showlegend=False, template='plotly_white', height=300)
                st.plotly_chart(fig_t, width='stretch')
            
            # Full table
            st.subheader("Detailed Comparison Matrix")
            st.table(df)

with tab5:
    st.header("About Q-Orbit")
    
    st.markdown("""
    ## 🚀 Hybrid LLM-Quantum Portfolio Optimization
    
    **Q-Orbit** is a cutting-edge portfolio optimization system that combines:
    
    ### 🔬 Technologies
    
    1. **Classical Optimization** (Phase 1)
       - Markowitz Mean-Variance Optimization
       - Minimum Variance & Maximum Sharpe Ratio strategies
       - Efficient Frontier analysis
    
    2. **Sentiment Analysis** (Phase 2)
       - News sentiment integration using FinBERT
       - Real-time news data from NewsAPI
       - Sentiment-aware portfolio constraints
    
    3. **Quantum Computing** (Phase 3)
       - QAOA (Quantum Approximate Optimization Algorithm)
       - Quantum portfolio optimization on Qiskit
       - Hybrid classical-quantum approaches
    
    4. **LLM Integration** (Phase 4)
       - Natural language portfolio analysis
       - Market commentary generation
       - Risk narrative synthesis
    
    ### 📊 Features
    
    - **Interactive Portfolio Builder**: Select stocks from presets or create custom portfolios
    - **Real-time Data**: Fetch live market data via yfinance
    - **Multiple Strategies**: Compare different optimization methods
    - **Rich Visualizations**: Efficient frontier, pie charts, cumulative returns
    - **Performance Metrics**: Sharpe ratio, max drawdown, volatility analysis
    
    ### 🛠️ Tech Stack
    
    - **Frontend**: Streamlit with Plotly visualizations
    - **Backend**: Python with CVXPY & SciPy optimization
    - **Quantum**: Qiskit, Qiskit-Aer (local simulator), Qiskit-IBM-Runtime (QPU)
    - **ML/NLP**: Transformers (FinBERT), VADER Sentiment
    - **Data**: yfinance (market data), NewsAPI (financial news)
    
    ### 📚 Usage
    
    1. Select stocks from the sidebar
    2. Configure optimization parameters
    3. Click "Optimize Portfolio" to run the analysis
    4. Explore results across different tabs
    
    ### ⚙️ Configuration
    
    Create a `.env` file with your API keys:
    ```
    NEWS_API_KEY=your_newsapi_key
    ALPHA_VANTAGE_API_KEY=your_alphavantage_key
    IBM_QUANTUM_TOKEN=your_ibm_token  # Optional
    ```
    
    ### 📄 License & Credits
    
    Built with ❤️ using Streamlit, Qiskit, and Transformers
    """)
    
    st.info("💡 **Tip**: Start with the default Diversified portfolio and Minimum Variance optimization to see the system in action!")

# =====================================
# PDF Download Button
# =====================================
st.markdown("---")

if 'optimization_results' in st.session_state:
    results_for_pdf = st.session_state.optimization_results

    st.markdown(
        "<h3 style='text-align:center; color:#667eea;'>📄 Full Analysis Report</h3>",
        unsafe_allow_html=True
    )
    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        if st.button("📄 Generate & Download PDF Report", type="primary", width='stretch'):
            # Clear any previous kaleido error
            st.session_state.pop("kaleido_error", None)
            with st.spinner("Generating PDF report…"):
                try:
                    pdf_bytes = generate_pdf_report(
                        results_for_pdf,
                        selected_tickers,
                        risk_free_rate,
                        preset_choice,
                        optimization_method
                    )
                    filename = f"qorbit_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                    # Warn if kaleido failed (charts will be absent from PDF)
                    if st.session_state.get("kaleido_error"):
                        st.warning(
                            f"⚠️ Chart images could not be rendered (kaleido error): "
                            f"{st.session_state['kaleido_error']}\n\n"
                            "The PDF will be downloaded without chart images. "
                            "This is a known issue on Streamlit Cloud — the tables and metrics are still included."
                        )

                    st.download_button(
                        label="📥 Click here to download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        width='stretch',
                    )
                    st.caption("Includes: Portfolio Weights · Performance Metrics · Strategy Comparison")
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {e}")
