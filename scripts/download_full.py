#!/usr/bin/env python3
"""Full Dataset Download: All symbols, all timeframes, 2020-present.

This downloads the COMPLETE dataset as specified in your requirements:
- Symbols: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, SUI, OP (top 10)
- Timeframes: 15m, 1h, 4h, 1d (all 4)
- Date Range: 2020-01-01 to present (institutional era)

Estimated:
- Time: 45-60 minutes
- Size: ~500MB (compressed Parquet)
- Cost: Free (CCXT public API)

Usage:
  python scripts/download_full.py
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline.config import (
    VALID_SYMBOLS,
    TIMEFRAMES,
    START_DATE,
    END_DATE,
)
from data_pipeline.downloaders.ccxt_ohlcv import download_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download complete dataset."""
    
    logger.info("\n" + "="*70)
    logger.info("📥 FULL DATASET DOWNLOAD")
    logger.info("="*70)
    
    logger.info(f"\n📊 Configuration:")
    logger.info(f"  Symbols: {', '.join(VALID_SYMBOLS)}")
    logger.info(f"  Timeframes: {', '.join(TIMEFRAMES)}")
    logger.info(f"  Date Range: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"  Total datasets: {len(VALID_SYMBOLS)} × {len(TIMEFRAMES)} = {len(VALID_SYMBOLS) * len(TIMEFRAMES)}")
    
    # Estimate
    total_datasets = len(VALID_SYMBOLS) * len(TIMEFRAMES)
    estimated_minutes = total_datasets * 1.2  # ~1.2 min per dataset
    estimated_size_mb = total_datasets * 12   # ~12MB per dataset
    
    logger.info(f"\n⏱️  Estimated Time: {estimated_minutes:.0f} minutes")
    logger.info(f"💾 Estimated Size: {estimated_size_mb:.0f} MB")
    
    # Confirm
    logger.info(f"\n⚠️  This will download {total_datasets} datasets.")
    logger.info(f"⚠️  Press Ctrl+C within 10 seconds to cancel...")
    
    import time
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("\n❌ Download cancelled by user")
        return
    
    logger.info(f"\n🚀 Starting download...")
    
    # ====================================================================
    # Download All Data
    # ====================================================================
    
    start_time = datetime.now()
    
    try:
        files = download_main(
            symbols=VALID_SYMBOLS,
            timeframes=TIMEFRAMES,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir="src/data/jan2025_download/ohlcv",
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        logger.info(f"\n" + "="*70)
        logger.info(f"✅ DOWNLOAD COMPLETE!")
        logger.info(f"="*70)
        logger.info(f"  Files Downloaded: {len(files)}")
        logger.info(f"  Duration: {duration:.1f} minutes")
        logger.info(f"  Location: src/data/jan2025_download/ohlcv/")
        
        # Calculate total size
        total_size = 0
        for filepath in files:
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
        
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"  Total Size: {total_size_mb:.1f} MB")
        
        # Summary by symbol
        logger.info(f"\n📊 Downloaded Datasets:")
        symbols_count = {}
        for filepath in files:
            filename = os.path.basename(filepath)
            symbol = filename.split('_')[0].split('-')[0]
            symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
        
        for symbol, count in sorted(symbols_count.items()):
            logger.info(f"  {symbol}: {count} timeframes")
        
        logger.info(f"\n👉 Next Steps:")
        logger.info(f"  1. Run data validation: python scripts/validate_data.py")
        logger.info(f"  2. Run full backtests: python scripts/backtest_all.py")
        logger.info(f"  3. Export for RBI Agent: included in backtest_all.py")
        logger.info(f"  4. Run RBI screening: python src/agents/rbi_agent_pp_multi.py")
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️  Download interrupted by user")
        logger.info(f"  Partial data saved to: src/data/jan2025_download/ohlcv/")
        logger.info(f"  You can resume by running this script again.")
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        logger.info(f"\n🔍 Troubleshooting:")
        logger.info(f"  1. Check internet connection")
        logger.info(f"  2. Check if Binance API is accessible")
        logger.info(f"  3. Check rate limits (1200 calls/min)")
        logger.info(f"  4. Try again later (API might be temporarily down)")
        raise


if __name__ == "__main__":
    main()
