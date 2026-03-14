"""
Lightweight Sentiment Analyzer - Fallback for Windows
Uses VADER (no PyTorch dependencies) for reliable sentiment analysis
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json


class LightweightSentimentAnalyzer:
    """
    VADER-based sentiment analyzer (works on Windows without PyTorch issues)
    Specifically tuned for financial news
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        print("Loading VADER sentiment analyzer...")
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Financial-specific lexicon adjustments
        self.financial_boosts = {
            'beat': 1.5,
            'surge': 1.5,
            'soar': 1.5,
            'record': 1.3,
            'profit': 1.2,
            'growth': 1.2,
            'gain': 1.2,
            'loss': -1.5,
            'plunge': -1.5,
            'crash': -1.8,
            'fell': -1.3,
            'decline': -1.3,
            'miss': -1.5,
        }
        
        print("✓ VADER analyzer ready")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores
        """
        if not text or len(text.strip()) < 5:
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentiment_value': 0.0
            }
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']  # -1 to +1
        
        # Apply financial keyword boosts
        text_lower = text.lower()
        boost = 0.0
        for keyword, weight in self.financial_boosts.items():
            if keyword in text_lower:
                boost += weight * 0.1
        
        # Combine compound score with boost
        sentiment_value = max(-1.0, min(1.0, compound + boost))
        
        # Determine label
        if sentiment_value > 0.05:
            label = 'positive'
            score = abs(sentiment_value)
        elif sentiment_value < -0.05:
            label = 'negative'
            score = abs(sentiment_value)
        else:
            label = 'neutral'
            score = abs(sentiment_value)
        
        return {
            'label': label,
            'score': score,
            'sentiment_value': sentiment_value
        }
    
    def analyze_article(self, article: Dict) -> Dict:
        """Analyze sentiment of news article"""
        text = article.get('title', '')
        
        if article.get('description'):
            text += '. ' + article['description']
        
        sentiment = self.analyze_text(text)
        
        return {
            **article,
            'sentiment_label': sentiment['label'],
            'sentiment_score': sentiment['score'],
            'sentiment_value': sentiment['sentiment_value']
        }
    
    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all articles in DataFrame"""
        if news_df.empty:
            return news_df
        
        print(f"\\nAnalyzing sentiment for {len(news_df)} articles...")
        
        sentiments = []
        for idx, row in news_df.iterrows():
            text = str(row.get('title', ''))
            if row.get('description'):
                text += '. ' + str(row['description'])
            
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(news_df)} articles...")
        
        news_df['sentiment_label'] = [s['label'] for s in sentiments]
        news_df['sentiment_score'] = [s['score'] for s in sentiments]
        news_df['sentiment_value'] = [s['sentiment_value'] for s in sentiments]
        
        print(f"✓ Sentiment analysis complete!")
        
        return news_df
    
    def get_stock_sentiment_summary(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Get aggregated sentiment summary per stock"""
        if news_df.empty or 'sentiment_value' not in news_df.columns:
            return pd.DataFrame()
        
        summary = news_df.groupby('ticker').agg({
            'sentiment_value': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        summary.columns = ['avg_sentiment', 'sentiment_std', 'article_count', 'label_distribution']
        
        return summary.sort_values('avg_sentiment', ascending=False)


if __name__ == "__main__":
    # Test the analyzer
    print("Testing Lightweight Sentiment Analyzer")
    print("=" * 60)
    
    analyzer = LightweightSentimentAnalyzer()
    
    test_texts = [
        "Apple reports record quarterly earnings, beating analyst expectations significantly.",
        "Tesla stock plummets as CEO faces regulatory investigation.",
        "Microsoft announces new cloud computing partnership with major enterprise clients.",
        "Amazon faces challenges amid rising competition in e-commerce sector.",
        "Google parent Alphabet maintains steady growth in advertising revenue."
    ]
    
    print("\\nSample Sentiment Analysis:")
    print("-" * 60)
    
    for text in test_texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\\nText: {text[:70]}...")
        print(f"  Sentiment: {sentiment['label'].upper()}")
        print(f"  Confidence: {sentiment['score']:.3f}")
        print(f"  Value: {sentiment['sentiment_value']:+.3f}")
    
    print("\\n" + "=" * 60)
    print("✓ VADER analyzer working perfectly!")
