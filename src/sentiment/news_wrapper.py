"""
Simple News Collector Wrapper
Provides a simplified interface to news collection with optional parameters
"""

from pathlib import Path
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

try:
    from sentiment.collector import NewsCollector as BaseNewsCollector, get_stock_company_mapping
except ImportError:
    from collector import NewsCollector as BaseNewsCollector, get_stock_company_mapping


class NewsCollector:
    """Simplified news collector with optional API key"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news collector
        
        Args:
            api_key: NewsAPI key (if None, loads from environment)
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('NEWS_API_KEY')
        
        if api_key:
            self.collector = BaseNewsCollector(api_key=api_key)
            self.has_api = True
        else:
            self.collector = None
            self.has_api = False
    
    def fetch_news(
        self,
        ticker: str,
        days_back: int = 7,
        max_articles: int = 10
    ) -> List[Dict]:
        """
        Fetch news for a single ticker
        
        Args:
            ticker: Stock ticker
            days_back: Days to look back
            max_articles: Max articles to fetch
            
        Returns:
            List of news articles (empty if no API key)
        """
        if not self.has_api:
            return []
        
        company_mapping = get_stock_company_mapping()
        company_name = company_mapping.get(ticker,ticker)
        
        return self.collector.fetch_company_news(
            ticker=ticker,
            company_name=company_name,
            days_back=days_back,
            max_articles=max_articles
        )
