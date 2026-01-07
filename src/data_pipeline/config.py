"""Configuration for Jan 2025 data download pipeline.

Based on Jim Simons methodology:
- Liquid assets only (BTC, ETH, SOL, BNB front 10)
- Stable timeframes (15m, 1h, 4h, 1d)
- Regime-aware dates (2020/1/1 onwards - institutional era)
"""

from typing import List, Dict
from datetime import datetime

# ============================================================================
# ASSET CONFIGURATION
# ============================================================================

# Top 10 cryptocurrencies by market cap (as of Jan 2025)
# Focus on liquid, established assets only
TOP_10_SYMBOLS = [
    "BTC",   # Bitcoin
    "ETH",   # Ethereum
    "SOL",   # Solana
    "BNB",   # Binance Coin
    "XRP",   # Ripple
    "ADA",   # Cardano
    "DOGE",  # Dogecoin
    "AVAX",  # Avalanche
    "SUI",   # Sui
    "OP",    # Optimism
]

# Additional tracked symbols (optional, for breadth)
ADDITIONAL_SYMBOLS = []

# All symbols to download
ALL_SYMBOLS = TOP_10_SYMBOLS + ADDITIONAL_SYMBOLS

# Filter: exclude low-liquidity/scam tokens
EXCLUDED_PATTERNS = ["SCAM", "TEST", "TEMP"]

def filter_valid_symbols(symbols: List[str]) -> List[str]:
    """Filter out blacklisted symbols."""
    return [s for s in symbols if not any(p in s.upper() for p in EXCLUDED_PATTERNS)]

VALID_SYMBOLS = filter_valid_symbols(ALL_SYMBOLS)

# ============================================================================
# TIMEFRAME CONFIGURATION
# ============================================================================

# Jim Simons uses 15m and 1h for crypto (signal/noise ratio optimal)
# 4h and 1d for confirmation and regime detection
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Alternative grouping by use case
TIMEFRAME_GROUPS = {
    "fast": ["15m", "1h"],      # Intra-day trading
    "medium": ["4h", "1d"],     # Swing trading
    "all": ["15m", "1h", "4h", "1d"],  # Full analysis
}

# ============================================================================
# DATE RANGE CONFIGURATION
# ============================================================================

# 2020/1/1: Start of institutional crypto era
# - Before 2020: Market dominated by retail, different dynamics
# - After 2020: Futures, institutional adoption, better data quality
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

# For quick testing, use smaller date range
TEST_START_DATE = datetime(2024, 1, 1)
TEST_END_DATE = datetime.now()

print(f"📅 Download range: {START_DATE.date()} to {END_DATE.date()}")
print(f"⚠️  Test range (optional): {TEST_START_DATE.date()} to {TEST_END_DATE.date()}")

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

DATA_SOURCES = {
    "ohlcv": {
        "primary": "binance",
        "exchange_class": "ccxt.binance",
        "rate_limit": 1200,  # requests/min
        "data_type": "OHLCV",
        "required": True,
    },
    "funding_rate": {
        "primary": "binance_futures",
        "api_endpoint": "https://fapi.binance.com/fapi/v1/fundingRate",
        "rate_limit": 1200,
        "data_type": "Funding Rate",
        "required": True,
    },
    "liquidations": {
        "primary": "coinglass",
        "api_endpoint": "https://api.coinglass.com/api/liquidation/today",
        "requires_key": False,  # Free tier available
        "data_type": "Liquidations",
        "required": False,  # Nice to have
    },
    "whale_flows": {
        "primary": "glassnode",
        "api_endpoint": "https://api.glassnode.com/v1/metrics",
        "requires_key": True,
        "data_type": "On-chain metrics",
        "required": False,  # Optional
    },
}

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

# Use Parquet for 10x compression vs CSV
STORAGE_FORMAT = "parquet"  # or "csv" for legacy

STORAGE_PATHS = {
    "base": "src/data/jan2025_download",
    "ohlcv": "src/data/jan2025_download/ohlcv",
    "funding": "src/data/jan2025_download/funding",
    "liquidations": "src/data/jan2025_download/liquidations",
    "whale": "src/data/jan2025_download/whale",
    "metadata": "src/data/jan2025_download/metadata.json",
    "backup": "src/data/jan2025_download/backup",
}

# ============================================================================
# DOWNLOAD STRATEGY
# ============================================================================

# Rate limiting to avoid API bans
CCXT_RATE_LIMIT = {
    "calls_per_minute": 1000,
    "calls_per_hour": 50000,
    "retry_count": 5,
    "retry_delay": 60,  # seconds
}

# Batch download settings
BATCH_SIZE = 100  # Download 100 candles per call
PARALLEL_DOWNLOADS = 3  # Max 3 concurrent downloads

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Data quality thresholds
VALIDATION_RULES = {
    "ohlcv": {
        "min_candles_per_day": 96,  # 15m: 24*4 = 96 candles/day
        "max_gap_minutes": 30,  # Alert if gap > 30m
        "outlier_std": 5,  # Flag if price move > 5σ
    },
    "funding_rate": {
        "max_rate_abs": 0.5,  # Alert if funding > 50%
        "min_samples_per_day": 3,  # Usually 3x daily
    },
}

# ============================================================================
# BACKTEST SETTINGS (Connected to AI screening)
# ============================================================================

BACKTEST_CONFIG = {
    "initial_cash": 10000,  # $10k starting capital
    "commission": 0.001,  # 0.1% per trade (realistic for perps)
    "slippage_bps": 10,  # 10 basis points slippage
    "leverage": 1.0,  # Start unlevered
}

# For Moon Dev RBI agent integration
RBI_SCREENING_CONFIG = {
    "quick_symbols": ["BTC", "ETH"],  # Fast screening on top 2
    "quick_timeframes": ["1h", "4h"],
    "quick_date_range": "6M",  # Last 6 months
    "full_symbols": VALID_SYMBOLS,
    "full_timeframes": TIMEFRAMES,
    "full_date_range": "all",  # All available data
}

print(f"✅ Configuration loaded: {len(VALID_SYMBOLS)} symbols, {len(TIMEFRAMES)} timeframes")
