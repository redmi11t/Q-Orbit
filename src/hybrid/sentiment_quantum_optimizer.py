"""
Hybrid Sentiment-Quantum Portfolio Optimizer
Combines sentiment analysis from financial news with QAOA quantum optimization
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.unified_analyzer import SentimentAnalyzer
from sentiment.news_wrapper import NewsCollector
from quantum.qaoa_optimizer import QAOAOptimizer
from quantum.qubo_formulation import PortfolioQUBO


class SentimentQuantumOptimizer:
    """
    Hybrid optimizer combining sentiment analysis with quantum QAOA
    
    Workflow:
    1. Fetch financial news for portfolio stocks
    2. Analyze sentiment using FinBERT/VADER
    3. Map sentiment to portfolio constraints/penalties
    4. Formulate sentiment-aware QUBO problem
    5. Optimize with QAOA quantum algorithm
    """
    
    def __init__(
        self,
        news_api_key: Optional[str] = None,
        qaoa_layers: int = 2,
        qaoa_max_iterations: int = 50,
        sentiment_weight: float = 0.3,
        prefer_finbert: bool = True
    ):
        """
        Initialize hybrid optimizer
        
        Args:
            news_api_key: NewsAPI key (optional, will try from env)
            qaoa_layers: Number of QAOA layers
            qaoa_max_iterations: Max classical optimization iterations
            sentiment_weight: Weight for sentiment influence (0-1)
            prefer_finbert: Try to use FinBERT (falls back to VADER)
        """
        print("Initializing Hybrid Sentiment-Quantum Optimizer")
        print("=" * 70)
        
        # Sentiment analysis components
        print("\n[1/3] Loading sentiment analyzer...")
        self.sentiment_analyzer = SentimentAnalyzer(prefer_finbert=prefer_finbert)
        print(f"      Active backend: {self.sentiment_analyzer.get_backend_info()}")
        
        print("\n[2/3] Initializing news collector...")
        self.news_collector = NewsCollector(api_key=news_api_key)
        
        print("\n[3/3] Setting up quantum optimizer...")
        self.qaoa = QAOAOptimizer(
            num_layers=qaoa_layers,
            max_iterations=qaoa_max_iterations
        )
        
        self.sentiment_weight = sentiment_weight
        
        # Store results for analysis
        self.sentiment_scores = None
        self.news_data = None
        
        print("\n✓ Hybrid optimizer ready!")
        print("=" * 70)
    
    def optimize(
        self,
        returns: pd.DataFrame,
        tickers: List[str],
        budget: int = 5,
        days_back: int = 7,
        max_articles_per_stock: int = 10,
        risk_factor: float = 1.0,
        return_factor: float = 0.5,
        budget_penalty: float = 10.0
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Run hybrid sentiment-quantum optimization
        
        Args:
            returns: Historical returns DataFrame
            tickers: List of ticker symbols
            budget: Number of stocks to select
            days_back: Days of news to fetch
            max_articles_per_stock: Max articles per ticker
            risk_factor: QUBO risk weight
            return_factor: QUBO return weight
            budget_penalty: QUBO budget constraint penalty
            
        Returns:
            Tuple of (selected_tickers, weights, info_dict)
        """
        print("\n" + "=" * 70)
        print("HYBRID SENTIMENT-QUANTUM OPTIMIZATION")
        print("=" * 70)
        print(f"Portfolio: {len(tickers)} stocks")
        print(f"Budget: Select {budget} stocks")
        print(f"Sentiment weight: {self.sentiment_weight}")
        print(f"QAOA layers: {self.qaoa.num_layers}")
        
        # Step 1: Collect news
        print(f"\n[1/4] Fetching news (last {days_back} days)...")
        all_news = []
        for ticker in tickers:
            news = self.news_collector.fetch_news(
                ticker=ticker,
                days_back=days_back,
                max_articles=max_articles_per_stock
            )
            all_news.extend(news)
            if news:
                print(f"      {ticker}: {len(news)} articles")
        
        if not all_news:
            print("      ⚠️  No news found, using neutral sentiment")
            # Create neutral sentiment scores
            sentiment_summary = pd.DataFrame({
                'avg_sentiment': [0.0] * len(tickers),
                'sentiment_std': [0.0] * len(tickers),
                'article_count': [0] * len(tickers)
            }, index=tickers)
        else:
            print(f"      Total: {len(all_news)} articles collected")
            
            # Step 2: Analyze sentiment
            print("\n[2/4] Analyzing sentiment...")
            news_df = pd.DataFrame(all_news)
            news_df = self.sentiment_analyzer.analyze_news_dataframe(news_df)
            sentiment_summary = self.sentiment_analyzer.get_stock_sentiment_summary(news_df)
            
            self.news_data = news_df
            
            # Display sentiment summary
            print("\n      Sentiment Summary:")
            for ticker in tickers:
                if ticker in sentiment_summary.index:
                    avg_sent = sentiment_summary.loc[ticker, 'avg_sentiment']
                    count = sentiment_summary.loc[ticker, 'article_count']
                    emoji = "📈" if avg_sent > 0 else "📉" if avg_sent < 0 else "➡️"
                    print(f"      {ticker}: {avg_sent:+.3f} {emoji} ({count} articles)")
                else:
                    print(f"      {ticker}: No data")
        
        # Step 3: Create sentiment-adjusted QUBO
        print("\n[3/4] Formulating sentiment-aware QUBO...")
        
        # Standard QUBO formulation
        qubo_formulator = PortfolioQUBO(risk_factor, return_factor, budget_penalty)
        Q = qubo_formulator.formulate(returns, budget)
        
        # Apply sentiment adjustments
        Q_adjusted = self._apply_sentiment_to_qubo(
            Q, tickers, sentiment_summary, returns
        )
        
        # Step 4: Run QAOA with the sentiment-adjusted QUBO injected directly
        print("\n[4/4] Running QAOA quantum optimization (with sentiment-adjusted QUBO)...")

        # Pass Q_adjusted via precomputed_Q so QAOA does NOT rebuild its own matrix.
        # This is what makes the hybrid mode actually different from plain QAOA.
        selected_tickers, weights, info = self.qaoa.optimize(
            returns=returns,
            budget=budget,
            risk_factor=risk_factor,
            return_factor=return_factor,
            budget_penalty=budget_penalty,
            precomputed_Q=Q_adjusted      # <-- sentiment adjustments are now live
        )
        
        # Add sentiment information to results
        info['sentiment_summary'] = sentiment_summary
        info['sentiment_weight'] = self.sentiment_weight
        info['backend'] = self.sentiment_analyzer.get_backend_info()
        info['news_count'] = len(all_news)
        
        self.sentiment_scores = sentiment_summary
        
        return selected_tickers, weights, info
    
    def _apply_sentiment_to_qubo(
        self,
        Q: np.ndarray,
        tickers: List[str],
        sentiment_summary: pd.DataFrame,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Apply sentiment scores as penalties/bonuses to QUBO matrix
        
        Positive sentiment → Bonus (reduce cost for selecting)
        Negative sentiment → Penalty (increase cost for selecting)
        """
        Q_adjusted = Q.copy()
        
        for i, ticker in enumerate(tickers):
            if ticker in sentiment_summary.index:
                sentiment = sentiment_summary.loc[ticker, 'avg_sentiment']
            else:
                sentiment = 0.0
            
            # Sentiment bonus/penalty on diagonal
            # Negative sentiment = higher cost = less likely to select
            # Positive sentiment = lower cost = more likely to select
            sentiment_adjustment = -sentiment * self.sentiment_weight * 10
            Q_adjusted[i, i] += sentiment_adjustment
        
        print(f"      Applied sentiment adjustments to QUBO matrix")
        print(f"      Range: [{Q_adjusted.min():.2f}, {Q_adjusted.max():.2f}]")
        
        return Q_adjusted
    
    def get_sentiment_report(self) -> str:
        """Generate detailed sentiment report"""
        if self.sentiment_scores is None:
            return "No sentiment data available"
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("SENTIMENT ANALYSIS REPORT")
        report.append("=" * 70)
        
        for ticker in self.sentiment_scores.index:
            avg = self.sentiment_scores.loc[ticker, 'avg_sentiment']
            count = self.sentiment_scores.loc[ticker, 'article_count']
            
            if avg > 0.2:
                sentiment_label = "BULLISH 📈"
            elif avg < -0.2:
                sentiment_label = "BEARISH 📉"
            else:
                sentiment_label = "NEUTRAL ➡️"
            
            report.append(f"\n{ticker}:")
            report.append(f"  Sentiment: {sentiment_label}")
            report.append(f"  Score: {avg:+.3f}")
            report.append(f"  Articles: {count}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the hybrid optimizer
    print("Testing Hybrid Sentiment-Quantum Optimizer")
    
    # Generate test data
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    n_days = 252
    
    mean_returns = np.array([0.25, 0.22, 0.20, 0.35, 0.15]) / 252
    volatilities = np.array([0.30, 0.25, 0.28, 0.45, 0.60]) / np.sqrt(252)
    corr = np.eye(5) + 0.3 * (np.ones((5, 5)) - np.eye(5))
    
    cov = np.outer(volatilities, volatilities) * corr
    returns_array = np.random.multivariate_normal(mean_returns, cov, n_days)
    returns = pd.DataFrame(returns_array, columns=tickers)
    
    # Initialize hybrid optimizer
    optimizer = SentimentQuantumOptimizer(
        qaoa_layers=2,
        qaoa_max_iterations=30,
        sentiment_weight=0.3,
        prefer_finbert=True
    )
    
    # Run optimization
    selected, weights, info = optimizer.optimize(
        returns=returns,
        tickers=tickers,
        budget=3,
        days_back=7
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSelected: {', '.join(selected)}")
    print(f"Backend: {info['backend']}")
    print(f"News articles: {info['news_count']}")
    
    print(optimizer.get_sentiment_report())
