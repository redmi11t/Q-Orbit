"""Q-Orbit sentiment analysis package"""

# Conditional import for NewsCollector (requires newsapi-python)
try:
    from .collector import NewsCollector, get_stock_company_mapping
    NEWSAPI_COLLECTOR_AVAILABLE = True
except (ImportError, Exception):
    NEWSAPI_COLLECTOR_AVAILABLE = False
    NewsCollector = None
    get_stock_company_mapping = None

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
    'SentimentConstraintMapper',
    'SentimentConstraints',
    'LightweightSentimentAnalyzer',
    'SentimentAnalyzer',
    'SimpleNewsCollector',
]

# Only export full NewsCollector if newsapi-python is installed
if NEWSAPI_COLLECTOR_AVAILABLE:
    __all__.extend(['NewsCollector', 'get_stock_company_mapping'])

# Only export FinBERT if available
if FINBERT_AVAILABLE:
    __all__.append('FinancialSentimentAnalyzer')
