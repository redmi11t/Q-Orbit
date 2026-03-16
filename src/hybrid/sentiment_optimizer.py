"""
Sentiment-Aware Portfolio Optimizer
Integrates classical optimization with sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import cvxpy as cp

from ..classical.baseline import MarkowitzOptimizer
from ..sentiment.collector import NewsCollector
from ..sentiment.constraints import SentimentConstraintMapper

# FinBERT is heavy (PyTorch) – import only if available
try:
    from ..sentiment.analyzer import FinancialSentimentAnalyzer
    _FINBERT_AVAILABLE = True
except (ImportError, OSError):
    _FINBERT_AVAILABLE = False
    FinancialSentimentAnalyzer = None


class SentimentAwareOptimizer(MarkowitzOptimizer):
    """
    Portfolio optimizer that incorporates sentiment analysis
    Extends classical Markowitz optimization with news sentiment
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        news_api_key: Optional[str] = None,
        sentiment_weight: float = 0.3
    ):
        """
        Initialize sentiment-aware optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate
            news_api_key: NewsAPI key for fetching news
            sentiment_weight: How much to weight sentiment (0-1)
        """
        super().__init__(risk_free_rate)
        
        self.news_api_key = news_api_key
        self.sentiment_weight = sentiment_weight
        
        # Initialize components
        if news_api_key:
            self.news_collector = NewsCollector(news_api_key)
        else:
            self.news_collector = None
            
        self.sentiment_analyzer = None  # Lazy load (heavy model)
        self.constraint_mapper = SentimentConstraintMapper(sentiment_weight=sentiment_weight)
        
        # Storage
        self.news_data = None
        self.sentiment_data = None
        self.sentiment_constraints = None
        
    def _ensure_sentiment_analyzer(self):
        """Lazy load sentiment analyzer only when needed"""
        if self.sentiment_analyzer is None:
            print("Loading FinBERT sentiment model (this may take a minute)...")
            self.sentiment_analyzer = FinancialSentimentAnalyzer()
    
    def fetch_and_analyze_sentiment(
        self,
        tickers: List[str],
        ticker_to_company: Dict[str, str],
        days_back: int = 7,
        max_articles_per_stock: int = 20
    ) -> pd.DataFrame:
        """
        Fetch news and analyze sentiment for portfolio stocks
        
        Args:
            tickers: List of stock tickers
            ticker_to_company: Mapping of tickers to company names
            days_back: Days of news history
            max_articles_per_stock: Max articles per stock
            
        Returns:
            DataFrame with news and sentiment
        """
        if self.news_collector is None:
            raise ValueError("NewsAPI key required for fetching news")
        
        # Fetch news
        print("\n" + "=" * 60)
        print("PHASE 1: NEWS COLLECTION")
        print("=" * 60)
        
        self.news_data = self.news_collector.fetch_portfolio_news(
            ticker_to_company,
            days_back=days_back,
            max_articles_per_stock=max_articles_per_stock
        )
        
        if self.news_data.empty:
            print("⚠ No news data collected")
            return pd.DataFrame()
        
        # Analyze sentiment
        print("\n" + "=" * 60)
        print("PHASE 2: SENTIMENT ANALYSIS")
        print("=" * 60)
        
        self._ensure_sentiment_analyzer()
        self.news_data = self.sentiment_analyzer.analyze_news_dataframe(self.news_data)
        
        # Get summary
        self.sentiment_data = self.sentiment_analyzer.get_stock_sentiment_summary(self.news_data)
        
        print("\n" + "=" * 60)
        print("SENTIMENT SUMMARY")
        print("=" * 60)
        print(self.sentiment_data)
        
        return self.news_data
    
    def optimize_with_sentiment(
        self,
        returns: pd.DataFrame,
        tickers: List[str],
        ticker_to_company: Dict[str, str],
        days_back: int = 7,
        max_articles_per_stock: int = 20,
        use_cached_sentiment: bool = False
    ) -> np.ndarray:
        """
        Optimize portfolio with sentiment-aware constraints
        
        Args:
            returns: Historical returns DataFrame
            tickers: List of tickers (must match returns columns)
            ticker_to_company: Ticker to company name mapping
            days_back: Days of news to fetch
            max_articles_per_stock: Max articles per stock
            use_cached_sentiment: Use previously fetched sentiment
            
        Returns:
            Optimal weights array
        """
        # Fetch and analyze sentiment (if not using cache)
        if not use_cached_sentiment or self.sentiment_data is None:
            self.fetch_and_analyze_sentiment(
                tickers,
                ticker_to_company,
                days_back,
                max_articles_per_stock
            )
        
        if self.sentiment_data is None or self.sentiment_data.empty:
            print("\n⚠ No sentiment data available, using classical optimization")
            return self.optimize_max_sharpe(returns)
        
        # Map sentiment to constraints
        print("\n" + "=" * 60)
        print("PHASE 3: CONSTRAINT MAPPING")
        print("=" * 60)
        
        self.sentiment_constraints = self.constraint_mapper.map_sentiment_to_constraints(
            self.sentiment_data
        )
        
        for ticker, constraint in self.sentiment_constraints.items():
            print(f"\n{ticker}:")
            print(f"  Sentiment: {constraint.sentiment_score:+.3f}")
            print(f"  Weight Multiplier: {constraint.weight_multiplier:.2f}x")
            print(f"  Risk Penalty: {constraint.risk_penalty:.2f}x")
        
        # Adjust returns and covariance based on sentiment
        print("\n" + "=" * 60)
        print("PHASE 4: OPTIMIZATION")
        print("=" * 60)
        
        adj_returns, adj_cov = self.constraint_mapper.apply_constraints_to_returns(
            returns,
            self.sentiment_constraints
        )
        
        # Get weight bounds
        bounds = self.constraint_mapper.get_weight_bounds(
            list(returns.columns),
            self.sentiment_constraints,
            base_max_weight=0.30
        )
        
        # Optimize with sentiment-adjusted parameters
        n_assets = len(returns.columns)
        w = cp.Variable(n_assets)
        
        # Objective: maximize Sharpe ratio with sentiment-adjusted returns
        portfolio_return = adj_returns.values @ w
        
        # Ensure symmetry for cvxpy
        S = (adj_cov.values + adj_cov.values.T) / 2
        portfolio_risk = cp.quad_form(w, S)
        
        objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))
        
        # Constraints with sentiment-based bounds
        constraints_list = [cp.sum(w) == 1]
        
        for i, ticker in enumerate(returns.columns):
            min_w, max_w = bounds.get(ticker, (0.0, 0.3))
            constraints_list.append(w[i] >= min_w)
            constraints_list.append(w[i] <= max_w)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if w.value is None:
            raise ValueError("Optimization failed to converge")
        
        self.weights = pd.Series(w.value, index=returns.columns)
        self._calculate_performance(returns, self.weights)
        
        print("\n✓ Sentiment-aware optimization complete!")
        
        return self.weights.values


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from ..utils.data_loader import DataLoader
    
    # Load environment
    load_dotenv()
    
    # Sample portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    ticker_to_company = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'TSLA': 'Tesla',
        'NVDA': 'NVIDIA'
    }
    
    # Load price data
    loader = DataLoader()
    prices = loader.fetch_price_data(tickers, '2023-01-01', '2024-01-01')
    returns = loader.calculate_returns(prices)
    
    # Initialize sentiment-aware optimizer
    optimizer = SentimentAwareOptimizer(
        news_api_key=os.getenv('NEWS_API_KEY'),
        sentiment_weight=0.3
    )
    
    # Optimize with sentiment
    weights = optimizer.optimize_with_sentiment(
        returns,
        tickers,
        ticker_to_company,
        days_back=7,
        max_articles_per_stock=15
    )
    
    print("\n" + "=" * 60)
    print("FINAL PORTFOLIO")
    print("=" * 60)
    
    for ticker, weight in optimizer.weights.items():
        if weight > 0.01:
            sentiment = optimizer.sentiment_constraints.get(ticker)
            sentiment_str = f"({sentiment.sentiment_score:+.2f})" if sentiment else ""
            print(f"{ticker:6s}: {weight:6.2%} {sentiment_str}")
    
    print("\nPerformance:")
    for metric, value in optimizer.get_performance_summary().items():
        print(f"  {metric:20s}: {value}")
