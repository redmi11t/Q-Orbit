"""Q-Orbit sentiment analysis package"""

# Only import modules that don't require PyTorch
from .collector import NewsCollector, get_stock_company_mapping
from .constraints import SentimentConstraintMapper, SentimentConstraints

# Conditional import for FinBERT (only if PyTorch works)
try:
    from .analyzer import FinancialSentimentAnalyzer
    FINBERT_AVAILABLE = True
except (ImportError, OSError):
    FINBERT_AVAILABLE = False

# Always available lightweight modules
from .lightweight_analyzer import LightweightSentimentAnalyzer
from .unified_analyzer import SentimentAnalyzer
from .news_wrapper import NewsCollector as SimpleNewsCollector

__all__ = [
    'NewsCollector',
    'get_stock_company_mapping',
    'SentimentConstraintMapper',
    'SentimentConstraints',
    'LightweightSentimentAnalyzer',
    'SentimentAnalyzer',
    'SimpleNewsCollector',
]

# Only export FinBERT if available
if FINBERT_AVAILABLE:
    __all__.append('FinancialSentimentAnalyzer')
