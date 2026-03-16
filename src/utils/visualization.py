"""
Visualization Utilities
Create charts and plots for portfolio analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

# Optional plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Set style (only if seaborn is available)
if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PortfolioVisualizer:
    """Visualization tools for portfolio analysis"""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with style"""
        self.style = style
        
    def plot_efficient_frontier(
        self,
        risks: np.ndarray,
        returns: np.ndarray,
        weights_list: List[np.ndarray],
        highlight_portfolio: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot the efficient frontier
        
        Args:
            risks: Array of portfolio volatilities
            returns: Array of portfolio returns
            weights_list: List of weight arrays
            highlight_portfolio: Dict with 'risk' and 'return' to highlight
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot efficient frontier
        ax.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
        
        # Highlight specific portfolio
        if highlight_portfolio:
            ax.scatter(
                highlight_portfolio['risk'],
                highlight_portfolio['return'],
                marker='*',
                s=500,
                c='red',
                label='Optimal Portfolio',
                zorder=3
            )
        
        ax.set_xlabel('Annual Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Annual Return', fontsize=12)
        ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_weights(
        self,
        weights: pd.Series,
        title: str = "Portfolio Weights",
        save_path: Optional[str] = None
    ):
        """
        Plot portfolio weights as a bar chart
        
        Args:
            weights: Series of weights indexed by ticker
            title: Chart title
            save_path: Optional path to save figure
        """
        # Filter out zero weights
        weights_filtered = weights[weights > 0.001].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(weights_filtered))
        weights_filtered.plot(kind='bar', ax=ax, color=colors)
        
        ax.set_xlabel('Asset', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cumulative_returns(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot cumulative returns over time
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            benchmark_returns: Optional benchmark returns
            save_path: Optional path to save figure
        """
        # Calculate portfolio returns
        portfolio_returns = (returns @ weights)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot portfolio
        ax.plot(cumulative_portfolio.index, cumulative_portfolio.values,
                label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            ax.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                    label='Benchmark (S&P 500)', linewidth=2, 
                    color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(
        self,
        returns: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix heatmap
        
        Args:
            returns: DataFrame of asset returns
            save_path: Optional path to save figure
        """
        corr = returns.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_risk_return_scatter(
        self,
        returns: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Scatter plot of individual asset risk vs return
        
        Args:
            returns: DataFrame of asset returns
            save_path: Optional path to save figure
        """
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(annual_volatility, annual_returns, s=100, alpha=0.6)
        
        # Add labels for each point
        for ticker, vol, ret in zip(returns.columns, annual_volatility, annual_returns):
            ax.annotate(ticker, (vol, ret), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Annual Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Annual Return', fontsize=12)
        ax.set_title('Individual Asset Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_performance_dashboard(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        performance: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a comprehensive dashboard with multiple charts
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            performance: Performance metrics dict
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Weights
        ax1 = fig.add_subplot(gs[0, 0])
        weights_filtered = weights[weights > 0.001].sort_values(ascending=False)
        colors = sns.color_palette("husl", len(weights_filtered))
        weights_filtered.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Portfolio Weights', fontweight='bold')
        ax1.set_ylabel('Weight')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 2. Performance metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        metrics_text = '\n'.join([f"{k}: {v}" for k, v in performance.items()])
        ax2.text(0.1, 0.5, metrics_text, fontsize=12,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax2.set_title('Performance Metrics', fontweight='bold')
        
        # 3. Cumulative returns
        ax3 = fig.add_subplot(gs[1, :])
        portfolio_returns = (returns @ weights)
        cumulative = (1 + portfolio_returns).cumprod()
        ax3.plot(cumulative.index, cumulative.values, linewidth=2)
        ax3.set_title('Cumulative Returns', fontweight='bold')
        ax3.set_ylabel('Growth of $1')
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation heatmap
        ax4 = fig.add_subplot(gs[2, 0])
        # Show only assets in portfolio
        active_assets = weights[weights > 0.001].index
        corr = returns[active_assets].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax4, cbar_kws={"shrink": 0.8})
        ax4.set_title('Correlation Matrix (Active Assets)', fontweight='bold')
        
        # 5. Drawdown
        ax5 = fig.add_subplot(gs[2, 1])
        cumulative_max = cumulative.expanding().max()
        drawdown = (cumulative - cumulative_max) / cumulative_max
        ax5.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
        ax5.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax5.set_title('Drawdown', fontweight='bold')
        ax5.set_ylabel('Drawdown')
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax5.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from utils.data_loader import DataLoader, get_sample_portfolio
    from classical.baseline import MarkowitzOptimizer
    
    # Load data
    loader = DataLoader()
    portfolio = get_sample_portfolio()
    prices = loader.fetch_price_data(
        portfolio['tickers'],
        portfolio['start_date'],
        portfolio['end_date']
    )
    returns = loader.calculate_returns(prices)
    
    # Optimize
    optimizer = MarkowitzOptimizer()
    optimizer.optimize_max_sharpe(returns)
    
    # Visualize
    viz = PortfolioVisualizer()
    
    print("Generating visualizations...")
    viz.plot_weights(optimizer.weights)
    viz.plot_cumulative_returns(returns, optimizer.weights)
    viz.plot_correlation_matrix(returns)
    viz.create_performance_dashboard(
        returns,
        optimizer.weights,
        optimizer.get_performance_summary()
    )
