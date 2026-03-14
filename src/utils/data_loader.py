"""
Data Loading Utilities
Fetch and manage financial data for portfolio optimization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

class DataLoader:
    """Fetch and cache financial market data"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path("data/prices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_price_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for multiple tickers
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with adjusted close prices
        """
        cache_file = self.cache_dir / f"prices_{'_'.join(tickers)}_{start_date}_{end_date}.csv"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Fetch from Yahoo Finance
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)
        
        # Extract adjusted close prices
        # Note: With auto_adjust=True (default since yfinance>=0.2.38),
        # 'Close' is already the adjusted close price; 'Adj Close' no longer exists.
        if len(tickers) == 1:
            prices = data[['Close']].copy()
            prices.columns = tickers
        else:
            prices = data['Close'].copy()
        
        # Handle missing data
        prices = prices.dropna(how='all')  # Remove dates with no data
        prices = prices.ffill().bfill()    # Forward/backward fill (pandas 2.2+ compatible)
        
        # Cache the data
        prices.to_csv(cache_file)
        print(f"Cached data to {cache_file}")
        
        return prices
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: DataFrame of prices
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame of returns
        """
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_sp500_tickers(self, count: int = 20) -> List[str]:
        """
        Get a subset of S&P 500 tickers
        
        Args:
            count: Number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        # Popular large-cap stocks for testing
        popular_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
            'WMT', 'JPM', 'MA', 'PG', 'UNH',
            'HD', 'DIS', 'BAC', 'XOM', 'COST',
            'PFE', 'ABBV', 'KO', 'PEP', 'CSCO',
            'ADBE', 'NFLX', 'CMCSA', 'VZ', 'INTC'
        ]
        
        return popular_tickers[:count]
    
    def get_benchmark_data(
        self,
        start_date: str,
        end_date: str,
        benchmark: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Fetch benchmark index data (e.g., S&P 500)
        
        Args:
            start_date: Start date
            end_date: End date
            benchmark: Benchmark ticker (default: SPY)
            
        Returns:
            DataFrame with benchmark prices
        """
        return self.fetch_price_data([benchmark], start_date, end_date)


def get_sample_portfolio() -> Dict[str, any]:
    """
    Get a sample portfolio configuration for testing
    
    Returns:
        Dictionary with tickers and date range
    """
    return {
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                   'JPM', 'JNJ', 'V', 'PG', 'NVDA'],
        'start_date': '2020-01-01',
        'end_date': '2024-01-01',
        'benchmark': 'SPY'
    }


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Get sample portfolio
    portfolio = get_sample_portfolio()
    
    # Fetch data
    prices = loader.fetch_price_data(
        portfolio['tickers'],
        portfolio['start_date'],
        portfolio['end_date']
    )
    
    print("\nPrice Data Shape:", prices.shape)
    print("\nFirst few rows:")
    print(prices.head())
    
    # Calculate returns
    returns = loader.calculate_returns(prices)
    print("\nReturns Data Shape:", returns.shape)
    print("\nReturns Statistics:")
    print(returns.describe())
