"""
Q-Orbit Configuration
Loads settings from .env file and provides centralized configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for Q-Orbit project"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
    
    # Portfolio Settings
    DEFAULT_ASSET_COUNT = int(os.getenv("DEFAULT_ASSET_COUNT", "10"))
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))
    REBALANCE_FREQUENCY = os.getenv("REBALANCE_FREQUENCY", "monthly")
    
    # Sentiment Analysis
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
    SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.5"))
    NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
    
    # Quantum Settings
    QAOA_LAYERS = int(os.getenv("QAOA_LAYERS", "3"))
    QUANTUM_BACKEND = os.getenv("QUANTUM_BACKEND", "qasm_simulator")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "100"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def validate_api_keys(cls):
        """Check if required API keys are set"""
        warnings = []
        
        if not cls.NEWS_API_KEY:
            warnings.append("NEWS_API_KEY not set - news fetching will be limited")
        
        if not any([cls.OPENAI_API_KEY, cls.GOOGLE_API_KEY, cls.ANTHROPIC_API_KEY]):
            warnings.append("No LLM API key set - using local models only")
        
        return warnings

# Create singleton instance
config = Config()
config.ensure_directories()
