"""
News Collection Module
Fetch financial news articles for portfolio assets
"""

from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import json
import time


class NewsCollector:
    """Fetch and cache financial news articles"""
    
    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        """
        Initialize news collector
        
        Args:
            api_key: NewsAPI key
            cache_dir: Directory to cache news data
        """
        self.newsapi = NewsApiClient(api_key=api_key)
        self.cache_dir = cache_dir or Path("data/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_company_news(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 7,
        max_articles: int = 50,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Fetch news articles for a specific company
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            company_name: Full company name (e.g., 'Apple')
            days_back: Number of days to look back
            max_articles: Maximum articles to fetch
            use_cache: Whether to use cached data
            
        Returns:
            List of news article dictionaries
        """
        # Check cache
        cache_file = self.cache_dir / f"{ticker}_{days_back}days.json"
        
        if use_cache and cache_file.exists():
            # Check if cache is recent (less than 1 day old)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                print(f"Loading cached news for {ticker}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Fetching news for {ticker} ({company_name})...")
        
        try:
            # Search for news using both ticker and company name
            query = f'"{company_name}" OR {ticker}'
            
            response = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                page_size=min(max_articles, 100)  # API limit
            )
            
            articles = response['articles']
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'ticker': ticker,
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': article['publishedAt'],
                    'author': article.get('author', 'Unknown')
                })
            
            # Cache the results
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Found {len(processed_articles)} articles for {ticker}")
            return processed_articles
            
        except Exception as e:
            print(f"  ✗ Error fetching news for {ticker}: {str(e)}")
            return []
    
    def fetch_portfolio_news(
        self,
        tickers_and_names: Dict[str, str],
        days_back: int = 7,
        max_articles_per_stock: int = 20
    ) -> pd.DataFrame:
        """
        Fetch news for multiple stocks in a portfolio
        
        Args:
            tickers_and_names: Dict mapping tickers to company names
                              e.g., {'AAPL': 'Apple', 'MSFT': 'Microsoft'}
            days_back: Number of days to look back
            max_articles_per_stock: Max articles per stock
            
        Returns:
            DataFrame with all news articles
        """
        all_articles = []
        
        print(f"\nFetching news for {len(tickers_and_names)} stocks...")
        print("=" * 60)
        
        for ticker, company_name in tickers_and_names.items():
            articles = self.fetch_company_news(
                ticker,
                company_name,
                days_back=days_back,
                max_articles=max_articles_per_stock
            )
            all_articles.extend(articles)
            
            # Rate limiting: NewsAPI free tier = 100 requests/day
            # Be conservative with requests
            time.sleep(0.5)  # Small delay between requests
        
        print("=" * 60)
        print(f"Total articles collected: {len(all_articles)}")
        
        if not all_articles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        return df
    
    def get_business_headlines(
        self,
        country: str = 'us',
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get top business headlines
        
        Args:
            country: Country code (e.g., 'us')
            page_size: Number of headlines to fetch
            
        Returns:
            List of headline dictionaries
        """
        try:
            response = self.newsapi.get_top_headlines(
                category='business',
                language='en',
                country=country,
                page_size=page_size
            )
            
            return response['articles']
            
        except Exception as e:
            print(f"Error fetching headlines: {str(e)}")
            return []


def get_stock_company_mapping() -> Dict[str, str]:
    """
    Get mapping of common stock tickers to company names
    
    Returns:
        Dictionary mapping tickers to company names
    """
    return {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'GOOG': 'Alphabet',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta',
        'NVDA': 'NVIDIA',
        'JPM': 'JPMorgan',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa',
        'PG': 'Procter & Gamble',
        'WMT': 'Walmart',
        'BAC': 'Bank of America',
        'XOM': 'Exxon',
        'COST': 'Costco',
        'PFE': 'Pfizer',
        'KO': 'Coca-Cola',
        'PEP': 'PepsiCo',
        'CSCO': 'Cisco'
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')
    
    # Initialize collector
    collector = NewsCollector(api_key)
    
    # Test: Fetch news for a few stocks
    test_stocks = {
        'AAPL': 'Apple',
        'TSLA': 'Tesla',
        'MSFT': 'Microsoft'
    }
    
    print("Testing News Collection Module")
    print("=" * 60)
    
    # Fetch portfolio news
    news_df = collector.fetch_portfolio_news(
        test_stocks,
        days_back=7,
        max_articles_per_stock=10
    )
    
    print("\nNews Summary:")
    print(f"Total articles: {len(news_df)}")
    print(f"\nArticles per stock:")
    print(news_df['ticker'].value_counts())
    
    print(f"\nSample articles:")
    for idx, row in news_df.head(3).iterrows():
        print(f"\n{row['ticker']} - {row['title']}")
        print(f"  Source: {row['source']}")
        print(f"  Published: {row['published_at']}")
        if row['description']:
            print(f"  {row['description'][:100]}...")
