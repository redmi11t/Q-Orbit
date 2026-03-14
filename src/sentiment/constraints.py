"""
Sentiment to Portfolio Constraints Mapping
Convert sentiment scores into quantitative portfolio constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SentimentConstraints:
    """Portfolio constraints derived from sentiment"""
    ticker: str
    sentiment_score: float  # -1 to +1
    weight_multiplier: float  # 0.5 to 1.5 (how much to adjust weight)
    risk_penalty: float  # 0 to 1 (additional risk to consider)
    confidence: float  # 0 to 1 (confidence in sentiment)


class SentimentConstraintMapper:
    """Map sentiment scores to portfolio optimization constraints"""
    
    def __init__(
        self,
        sentiment_weight: float = 0.3,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5
    ):
        """
        Initialize constraint mapper
        
        Args:
            sentiment_weight: How much to weight sentiment (0-1)
            min_multiplier: Minimum weight multiplier for very negative sentiment
            max_multiplier: Maximum weight multiplier for very positive sentiment
        """
        self.sentiment_weight = sentiment_weight
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        
    def sentiment_to_multiplier(self, sentiment_value: float) -> float:
        """
        Convert sentiment value to weight multiplier
        
        Sentiment: -1 (very negative) to +1 (very positive)
        Multiplier: 0.5 (reduce weight) to 1.5 (increase weight)
        
        Args:
            sentiment_value: Normalized sentiment (-1 to +1)
            
        Returns:
            Weight multiplier
        """
        # Linear mapping: -1 -> min_multiplier, +1 -> max_multiplier
        multiplier = (
            1.0 +  # Base multiplier
            sentiment_value * self.sentiment_weight *
            (self.max_multiplier - self.min_multiplier)
        )
        
        return np.clip(multiplier, self.min_multiplier, self.max_multiplier)
    
    def sentiment_to_risk_penalty(self, sentiment_value: float) -> float:
        """
        Convert negative sentiment to additional risk penalty
        
        Negative sentiment increases perceived risk
        
        Args:
            sentiment_value: Normalized sentiment (-1 to +1)
            
        Returns:
            Risk penalty multiplier (1.0 = normal, >1.0 = higher risk)
        """
        if sentiment_value >= 0:
            return 1.0  # No additional risk for positive/neutral sentiment
        
        # Negative sentiment increases risk perception
        # -1 -> 1.5x risk, 0 -> 1.0x risk
        risk_multiplier = 1.0 + abs(sentiment_value) * 0.5
        
        return risk_multiplier
    
    def calculate_confidence(self, article_count: int, sentiment_std: float) -> float:
        """
        Calculate confidence in sentiment signal
        
        More articles + lower std deviation = higher confidence
        
        Args:
            article_count: Number of articles analyzed
            sentiment_std: Standard deviation of sentiment scores
            
        Returns:
            Confidence score (0 to 1)
        """
        # More articles = higher confidence (saturates at 20 articles)
        count_confidence = min(article_count / 20.0, 1.0)
        
        # Lower std = higher confidence
        # std of 0 = 1.0 confidence, std of 0.5+ = 0 confidence
        std_confidence = max(0, 1.0 - sentiment_std * 2.0)
        
        # Combine (weighted average)
        confidence = 0.6 * count_confidence + 0.4 * std_confidence
        
        return np.clip(confidence, 0.0, 1.0)
    
    def map_sentiment_to_constraints(
        self,
        sentiment_summary: pd.DataFrame
    ) -> Dict[str, SentimentConstraints]:
        """
        Map sentiment summary to portfolio constraints
        
        Args:
            sentiment_summary: DataFrame with columns:
                - ticker (index)
                - avg_sentiment
                - sentiment_std
                - article_count
                
        Returns:
            Dictionary mapping tickers to SentimentConstraints
        """
        constraints = {}
        
        for ticker in sentiment_summary.index:
            row = sentiment_summary.loc[ticker]
            
            avg_sentiment = row['avg_sentiment']
            sentiment_std = row.get('sentiment_std', 0.3)
            article_count = int(row['article_count'])
            
            # Calculate constraint values
            weight_multiplier = self.sentiment_to_multiplier(avg_sentiment)
            risk_penalty = self.sentiment_to_risk_penalty(avg_sentiment)
            confidence = self.calculate_confidence(article_count, sentiment_std)
            
            constraints[ticker] = SentimentConstraints(
                ticker=ticker,
                sentiment_score=avg_sentiment,
                weight_multiplier=weight_multiplier,
                risk_penalty=risk_penalty,
                confidence=confidence
            )
        
        return constraints
    
    def apply_constraints_to_returns(
        self,
        returns: pd.DataFrame,
        constraints: Dict[str, SentimentConstraints]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Adjust expected returns and covariance based on sentiment
        
        Args:
            returns: DataFrame of historical returns
            constraints: Sentiment constraints dictionary
            
        Returns:
            Tuple of (adjusted_expected_returns, adjusted_covariance)
        """
        # Calculate base statistics
        expected_returns = returns.mean() * 252  # Annualized
        covariance = returns.cov() * 252
        
        # Apply sentiment adjustments
        adjusted_returns = expected_returns.copy()
        adjusted_cov = covariance.copy()
        
        for ticker in returns.columns:
            if ticker in constraints:
                constraint = constraints[ticker]
                
                # Adjust expected return based on sentiment
                # Positive sentiment -> increase expected return
                # Negative sentiment -> decrease expected return
                sentiment_adjustment = constraint.sentiment_score * 0.05  # Max ±5%
                adjusted_returns[ticker] += sentiment_adjustment * constraint.confidence
                
                # Adjust risk (variance) based on risk penalty
                # Negative sentiment -> increase perceived risk
                adjusted_cov.loc[ticker, ticker] *= constraint.risk_penalty
        
        return adjusted_returns, adjusted_cov
    
    def get_weight_bounds(
        self,
        tickers: list,
        constraints: Dict[str, SentimentConstraints],
        base_max_weight: float = 0.3
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get weight bounds for each asset based on sentiment
        
        Args:
            tickers: List of stock tickers
            constraints: Sentiment constraints
            base_max_weight: Base maximum weight per asset
            
        Returns:
            Dictionary mapping tickers to (min_weight, max_weight) tuples
        """
        bounds = {}
        
        for ticker in tickers:
            if ticker in constraints:
                constraint = constraints[ticker]
                
                # Positive sentiment -> allow higher weights
                # Negative sentiment -> restrict weights
                max_weight = base_max_weight * constraint.weight_multiplier
                max_weight = np.clip(max_weight, 0.0, 0.5)  # Cap at 50%
                
            else:
                # No sentiment data -> use base bounds
                max_weight = base_max_weight
            
            bounds[ticker] = (0.0, max_weight)
        
        return bounds


if __name__ == "__main__":
    # Test constraint mapping
    print("Testing Sentiment Constraint Mapper")
    print("=" * 60)
    
    # Create sample sentiment summary
    sentiment_data = {
        'ticker': ['AAPL', 'TSLA', 'MSFT', 'GOOGL'],
        'avg_sentiment': [0.65, -0.45, 0.30, 0.10],
        'sentiment_std': [0.15, 0.35, 0.20, 0.25],
        'article_count': [25, 30, 15, 12]
    }
    sentiment_summary = pd.DataFrame(sentiment_data).set_index('ticker')
    
    print("\nSentiment Summary:")
    print(sentiment_summary)
    
    # Initialize mapper
    mapper = SentimentConstraintMapper()
    
    # Map to constraints
    constraints = mapper.map_sentiment_to_constraints(sentiment_summary)
    
    print("\n" + "=" * 60)
    print("SENTIMENT CONSTRAINTS")
    print("=" * 60)
    
    for ticker, constraint in constraints.items():
        print(f"\n{ticker}:")
        print(f"  Sentiment Score: {constraint.sentiment_score:+.3f}")
        print(f"  Weight Multiplier: {constraint.weight_multiplier:.3f}x")
        print(f"  Risk Penalty: {constraint.risk_penalty:.3f}x")
        print(f"  Confidence: {constraint.confidence:.3f}")
        
        if constraint.sentiment_score > 0.3:
            print(f"  → BULLISH: Increase allocation")
        elif constraint.sentiment_score < -0.3:
            print(f"  → BEARISH: Reduce allocation")
        else:
            print(f"  → NEUTRAL")
    
    # Demo weight bounds
    print("\n" + "=" * 60)
    print("WEIGHT BOUNDS")
    print("=" * 60)
    
    bounds = mapper.get_weight_bounds(
        list(constraints.keys()),
        constraints,
        base_max_weight=0.25
    )
    
    for ticker, (min_w, max_w) in bounds.items():
        print(f"{ticker}: {min_w:.1%} - {max_w:.1%}")
