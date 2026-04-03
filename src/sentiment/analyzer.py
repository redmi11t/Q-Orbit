"""
Sentiment Analysis Module
Analyze sentiment of financial news using FinBERT
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import hashlib


class FinancialSentimentAnalyzer:
    """Sentiment analysis using FinBERT (financial domain-specific model)"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: Optional[Path] = None, use_cache: bool = True):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model (legacy param, now handles via transformers/SentimentCache)
            use_cache: Whether to use local JSON cache for results
        """
        print(f"Loading sentiment model: {model_name}...")
        
        # Load model and tokenizer
        # Note: The transformers library handles its own model caching by default
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create sentiment analysis pipeline
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Load cache if requested
        self.use_cache = use_cache
        if use_cache:
            self.cache = SentimentCache()
            print("  Sentiment cache enabled.")
        else:
            self.cache = None
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
            {
                'label': 'positive'/'negative'/'neutral',
                'score': confidence score (0-1),
                'sentiment_value': normalized score (-1 to +1)
            }
        """
        if not text or len(text.strip()) < 10:
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentiment_value': 0.0
            }
        
        # Truncate if too long (BERT max length = 512 tokens)
        # The pipeline handles truncation, but we can pre-truncate for cache key consistency
        processed_text = text[:512] # Max length for FinBERT
        
        try:
            # Check cache first
            if self.use_cache and self.cache:
                text_hash = hashlib.md5(processed_text.encode('utf-8')).hexdigest()
                cached_result = self.cache.get(text_hash)
                if cached_result:
                    return cached_result
            
            # Run inference
            results = self.pipeline(processed_text)
            result = results[0]
            
            # Convert to sentiment value (-1 to +1)
            label = result['label'].lower()
            score = result['score']
            
            sentiment_value = 0.0
            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            # else: neutral, sentiment_value remains 0.0
            
            final_result = {
                'label': label,
                'score': score,
                'sentiment_value': sentiment_value
            }
            
            # Save to cache
            if self.use_cache and self.cache:
                self.cache.set(text_hash, final_result)
                
            return final_result
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentiment_value': 0.0
            }
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a news article
        
        Args:
            article: Article dictionary with 'title', 'description', 'content'
            
        Returns:
            Sentiment analysis results
        """
        # Combine title and description for analysis
        # Title often has the most important sentiment signal
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
        """
        Analyze sentiment for all articles in DataFrame

        Args:
            news_df: DataFrame with news articles

        Returns:
            DataFrame with added sentiment columns
        """
        if news_df.empty:
            return news_df

        # Phase 4 Fix #14: Work on a copy so the caller's DataFrame is not
        # mutated in-place (avoids SettingWithCopyWarning and unexpected
        # side-effects when the same news_df is reused downstream).
        news_df = news_df.copy()

        print(f"\nAnalyzing sentiment for {len(news_df)} articles...")

        sentiments = []
        # Fix #6: use enumerate() for a proper loop counter instead of
        # the DataFrame index, which can be non-sequential after filtering.
        for loop_i, (idx, row) in enumerate(news_df.iterrows()):
            # Combine title and description
            text = str(row.get('title', ''))
            if row.get('description'):
                text += '. ' + str(row['description'])

            sentiment = self.analyze_text(text)
            sentiments.append(sentiment)

            # Progress indicator
            if (loop_i + 1) % 10 == 0:
                print(f"  Processed {loop_i + 1}/{len(news_df)} articles...")

        # Add sentiment columns
        news_df['sentiment_label'] = [s['label'] for s in sentiments]
        news_df['sentiment_score'] = [s['score'] for s in sentiments]
        news_df['sentiment_value'] = [s['sentiment_value'] for s in sentiments]

        print(f"✓ Sentiment analysis complete!")

        return news_df
    
    def get_stock_sentiment_summary(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get aggregated sentiment summary per stock
        
        Args:
            news_df: DataFrame with news and sentiment
            
        Returns:
            DataFrame with sentiment summary per ticker
        """
        if news_df.empty or 'sentiment_value' not in news_df.columns:
            return pd.DataFrame()
        
        summary = news_df.groupby('ticker').agg({
            'sentiment_value': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        summary.columns = ['avg_sentiment', 'sentiment_std', 'article_count', 'label_distribution']
        
        return summary.sort_values('avg_sentiment', ascending=False)


class SentimentCache:
    """Cache sentiment analysis results to avoid re-processing"""

    def __init__(self, cache_file: Path = None,
                 save_interval: int = 10):
        # Fix #14: Default to an absolute path anchored to this module's directory
        # so the cache always lands in data/sentiment/ under the project root,
        # regardless of the CWD when the process is started.
        if cache_file is None:
            _module_dir = Path(__file__).resolve().parent  # src/sentiment/
            cache_file = _module_dir.parent.parent / "data" / "sentiment" / "cache.json"
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        # Phase 4 Fix #10: Track dirty count so we write in batches.
        self._dirty = 0
        self.save_interval = save_interval

        # Fix #2: Register atexit flush so the cache is always persisted on
        # process exit — __del__ is unreliable on Windows and with hot-reloads.
        import atexit
        atexit.register(self._flush_if_dirty)

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        self._dirty = 0

    def _flush_if_dirty(self):
        """Flush only if there are unsaved entries (used by atexit)."""
        try:
            if self._dirty > 0:
                self._save_cache()
        except Exception:
            pass

    def get(self, text_hash: str) -> Optional[Dict]:
        return self.cache.get(text_hash)

    def set(self, text_hash: str, sentiment: Dict):
        self.cache[text_hash] = sentiment
        self._dirty += 1
        # Flush only every save_interval new entries
        if self._dirty >= self.save_interval:
            self._save_cache()

    def __del__(self):
        """Flush any remaining dirty entries when the cache object is GC'd."""
        try:
            if self._dirty > 0:
                self._save_cache()
        except Exception:
            pass  # Never raise in __del__


if __name__ == "__main__":
    # Test sentiment analyzer
    print("Testing Financial Sentiment Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Test with sample financial texts
    test_texts = [
        "Apple reports record quarterly earnings, beating analyst expectations significantly.",
        "Tesla stock plummets as CEO faces regulatory investigation.",
        "Microsoft announces new cloud computing partnership with major enterprise clients.",
        "Amazon faces challenges amid rising competition in e-commerce sector.",
        "Google parent Alphabet maintains steady growth in advertising revenue."
    ]
    
    print("\nSample Sentiment Analysis:")
    print("-" * 60)
    
    for text in test_texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\nText: {text[:70]}...")
        print(f"  Sentiment: {sentiment['label'].upper()}")
        print(f"  Confidence: {sentiment['score']:.3f}")
        print(f"  Value: {sentiment['sentiment_value']:+.3f}")
