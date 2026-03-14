"""
Unified Sentiment Analyzer Interface
Automatically selects the best available sentiment analyzer:
1. Try FinBERT (most accurate for financial news)
2. Fallback to VADER (lightweight, always works on Windows)
"""

import warnings
from typing import Dict, Optional
from pathlib import Path
import pandas as pd


class SentimentAnalyzer:
    """
    Smart sentiment analyzer that auto-selects the best available backend
    """
    
    def __init__(self, prefer_finbert: bool = True, cache_dir: Optional[Path] = None):
        """
        Initialize sentiment analyzer with automatic fallback
        
        Args:
            prefer_finbert: Try to load FinBERT first (most accurate)
            cache_dir: Directory for caching models
        """
        self.backend = None
        self.backend_name = None
        
        if prefer_finbert:
            # Try FinBERT first
            try:
                print("Attempting to load FinBERT (best for financial news)...")
                try:
                    from sentiment.analyzer import FinancialSentimentAnalyzer
                except ImportError:
                    from analyzer import FinancialSentimentAnalyzer
                self.backend = FinancialSentimentAnalyzer(cache_dir=cache_dir)
                self.backend_name = "FinBERT"
                print("✓ Using FinBERT (high accuracy)")
                return
            except Exception as e:
                print(f"⚠️  FinBERT unavailable: {str(e)[:100]}")
                print("   Falling back to VADER...")
        
        # Fallback to VADER
        try:
            try:
                from sentiment.lightweight_analyzer import LightweightSentimentAnalyzer
            except ImportError:
                from lightweight_analyzer import LightweightSentimentAnalyzer
            self.backend = LightweightSentimentAnalyzer()
            self.backend_name = "VADER"
            print("✓ Using VADER (lightweight, Windows-compatible)")
        except Exception as e:
            raise RuntimeError(f"Failed to load any sentiment analyzer: {e}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        return self.backend.analyze_text(text)
    
    def analyze_article(self, article: Dict) -> Dict:
        """Analyze sentiment of news article"""
        return self.backend.analyze_article(article)
    
    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all articles in DataFrame"""
        return self.backend.analyze_news_dataframe(news_df)
    
    def get_stock_sentiment_summary(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Get aggregated sentiment summary per stock"""
        return self.backend.get_stock_sentiment_summary(news_df)
    
    def get_backend_info(self) -> str:
        """Get information about which backend is being used"""
        return self.backend_name


if __name__ == "__main__":
    # Test the unified interface
    print("=" * 70)
    print("UNIFIED SENTIMENT ANALYZER TEST")
    print("=" * 70)
    
    # This will automatically select the best available backend
    analyzer = SentimentAnalyzer(prefer_finbert=True)
    
    print(f"\n✓ Active backend: {analyzer.get_backend_info()}")
    
    # Test with financial news
    test_cases = [
        "Apple reports record quarterly earnings, beating expectations.",
        "Tesla stock plummets on production concerns.",
        "Microsoft cloud revenue surges in latest quarter."
    ]
    
    print("\nSample Analysis:")
    print("-" * 70)
    
    for text in test_cases:
        result = analyzer.analyze_text(text)
        print(f"\n{text}")
        print(f"  → {result['label'].upper()} ({result['score']:.2f})")
        print(f"  → Sentiment: {result['sentiment_value']:+.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Unified sentiment analyzer working!")
    print("=" * 70)
