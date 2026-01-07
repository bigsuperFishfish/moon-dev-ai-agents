#!/usr/bin/env python3
"""Quick Start: Download data + Run factor backtest.

Jim Simons principle: Start small, validate rigorously, scale.

This script:
1. Downloads BTC + ETH for last 6 months (2x timeframes)
2. Validates data quality
3. Runs 3 simple factor backtests
4. Generates summary report
5. Prepares for RBI Agent screening

Runtime: ~5-10 minutes
Cost: Free (using CCXT + public APIs)

Usage:
  python scripts/quick_start.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline.config import VALID_SYMBOLS, TIMEFRAMES, TEST_START_DATE, TEST_END_DATE
from data_pipeline.downloaders.ccxt_ohlcv import CCXTDownloader, download_main
from data_pipeline.factors.momentum_factor import (
    MomentumFactor,
    MeanReversionFactor,
    TrendFollowingFactor,
)
from data_pipeline.backtest.vectorized_engine import VectorizedBacktester
from data_pipeline.ai_agents.moondev_integration import MoonDevBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution."""
    
    logger.info("\n" + "="*60)
    logger.info("🚀 JAN 2025 DATA DOWNLOAD + FACTOR BACKTEST")
    logger.info("="*60)
    
    # ====================================================================
    # STEP 1: Download OHLCV Data
    # ====================================================================
    logger.info(f"\n📄 STEP 1: Download OHLCV Data")
    logger.info(f"Symbols: BTC, ETH")
    logger.info(f"Timeframes: 1h, 4h")
    logger.info(f"Date Range: {TEST_START_DATE.date()} to {TEST_END_DATE.date()}")
    
    try:
        files = download_main(
            symbols=["BTC", "ETH"],
            timeframes=["1h", "4h"],
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            output_dir="src/data/jan2025_download/ohlcv",
        )
        logger.info(f"✅ Downloaded {len(files)} files")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("\n🔍 Troubleshooting:")
        logger.info("1. Check internet connection")
        logger.info("2. Check if Binance API is accessible")
        logger.info("3. Check rate limits (1200 calls/min)")
        return
    
    # Load downloaded data
    logger.info(f"\n📄 Loading downloaded data...")
    downloader = CCXTDownloader("binance")
    pairs = downloader.get_symbol_pairs(["BTC", "ETH"])
    
    # Load parquet files
    data_dict = {}
    for pair in pairs:
        for timeframe in ["1h", "4h"]:
            try:
                filename = f"{pair.replace('/', '-')}_{timeframe}.parquet"
                filepath = f"src/data/jan2025_download/ohlcv/{filename}"
                df = pd.read_parquet(filepath)
                data_dict[(pair, timeframe)] = df
                logger.info(f"✅ Loaded {pair} {timeframe}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"⚠️  Could not load {pair} {timeframe}: {e}")
    
    if not data_dict:
        logger.error("No data loaded!")
        return
    
    # ====================================================================
    # STEP 2: Run Factor Backtests
    # ====================================================================
    logger.info(f"\n📄 STEP 2: Run Factor Backtests")
    
    backtester = VectorizedBacktester(
        initial_cash=10000,
        commission=0.001,  # 0.1%
        slippage_bps=10,   # 10 bps
    )
    
    results_summary = {}
    
    for (pair, timeframe), df in list(data_dict.items())[:2]:  # Test first 2
        logger.info(f"\n🏗️  Testing {pair} {timeframe}")
        
        # Factor 1: Momentum (RSI)
        logger.info(f"  🤕 Factor 1: Momentum (RSI)")
        momentum = MomentumFactor(period=14)
        df_mom, stats_mom = momentum.generate_signals(df)
        result_mom = backtester.run_backtest(df_mom, df_mom)
        
        backtester.print_stats(result_mom['stats'])
        results_summary[f"{pair}_{timeframe}_momentum"] = result_mom['stats']
        
        # Factor 2: Mean Reversion
        logger.info(f"  🤕 Factor 2: Mean Reversion (BB)")
        mean_rev = MeanReversionFactor(period=20, num_std=2.0)
        df_mr, stats_mr = mean_rev.generate_signals(df)
        result_mr = backtester.run_backtest(df_mr, df_mr)
        
        backtester.print_stats(result_mr['stats'])
        results_summary[f"{pair}_{timeframe}_meanrev"] = result_mr['stats']
        
        # Factor 3: Trend Following
        logger.info(f"  🤕 Factor 3: Trend Following (SMA)")
        trend = TrendFollowingFactor(fast_period=5, slow_period=20)
        df_trend, stats_trend = trend.generate_signals(df)
        result_trend = backtester.run_backtest(df_trend, df_trend)
        
        backtester.print_stats(result_trend['stats'])
        results_summary[f"{pair}_{timeframe}_trend"] = result_trend['stats']
    
    # ====================================================================
    # STEP 3: Moon Dev AI Agent Integration
    # ====================================================================
    logger.info(f"\n📄 STEP 3: Moon Dev AI Agent Integration")
    
    bridge = MoonDevBridge(moondev_root="./")
    
    # Export data for RBI screening
    logger.info(f"  💾 Exporting data for RBI agent...")
    exported = bridge.export_data_for_rbi(data_dict, output_format="csv")
    logger.info(f"  ✅ Exported {len(exported)} datasets")
    
    # Trigger RBI screening
    logger.info(f"\n  🤖 Preparing RBI Agent prompt...")
    rbi_result = bridge.run_rbi_screening(
        factor_description="""Test multiple technical factors:
        - RSI oversold/overbought (periods: 7, 14, 21)
        - Bollinger Band mean reversion (periods: 15, 20, 25)
        - SMA trend following (fast: 3-5, slow: 15-25)
        Filter: Sharpe > 1.0, Max DD < 20%, Win Rate > 45%
        """,
        symbols=["BTC", "ETH"],
        quick_test=True,
    )
    
    # ====================================================================
    # STEP 4: Summary Report
    # ====================================================================
    logger.info(f"\n" + "="*60)
    logger.info("📄 SUMMARY REPORT")
    logger.info("="*60)
    
    # Best performing factor
    best_sharpe = 0
    best_factor = None
    
    for factor_name, stats in results_summary.items():
        if stats['sharpe_ratio'] > best_sharpe:
            best_sharpe = stats['sharpe_ratio']
            best_factor = factor_name
    
    logger.info(f"\n🌟 Best Performing Factor: {best_factor}")
    logger.info(f"  Sharpe Ratio: {best_sharpe:.2f}")
    logger.info(f"  Total Return: {results_summary[best_factor]['total_return']:.2%}")
    logger.info(f"  Max Drawdown: {results_summary[best_factor]['max_drawdown']:.2%}")
    
    # Next steps
    logger.info(f"\n👉 Next Steps:")
    logger.info(f"  1. Run RBI Agent screening (copy prompt above)")
    logger.info(f"  2. Validate top factors with cross-validation")
    logger.info(f"  3. Deploy via Moon Dev strategy_agent.py")
    logger.info(f"  4. Monitor live trading with risk_agent.py")
    
    # Data location
    logger.info(f"\n💾 Data Location:")
    logger.info(f"  OHLCV: src/data/jan2025_download/ohlcv/")
    logger.info(f"  Export: src/data/jan2025_data_export/")
    logger.info(f"  Backtest Results: (above log output)")
    
    logger.info(f"\n" + "="*60)
    logger.info("✅ Quick start complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
