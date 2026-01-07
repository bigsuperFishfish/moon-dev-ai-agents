"""CCXT-based OHLCV downloader for Binance.

Jim Simons principle: Get clean, high-quality data first.
No data = No alpha.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CCXTDownloader:
    """Download OHLCV data using CCXT.
    
    Features:
    - Rate limiting aware (1200 calls/min)
    - Retry logic with exponential backoff
    - Progress tracking
    - Metadata recording
    """
    
    def __init__(self, exchange_name: str = "binance"):
        """Initialize CCXT exchange.
        
        Args:
            exchange_name: CCXT exchange name (default: binance)
        """
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'rateLimit': 50,  # ms between requests
        })
        
        # Verify connection
        try:
            markets = self.exchange.load_markets()
            logger.info(f"✅ {exchange_name} connected. {len(markets)} markets available.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to {exchange_name}: {e}")
            raise
    
    def get_symbol_pairs(self, symbols: List[str], quote: str = "USDT") -> List[str]:
        """Convert symbol list to trading pairs.
        
        Args:
            symbols: List of symbols (e.g., ["BTC", "ETH"])
            quote: Quote currency (default: USDT)
            
        Returns:
            List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
        """
        pairs = []
        for symbol in symbols:
            pair = f"{symbol}/{quote}"
            if pair in self.exchange.symbols:
                pairs.append(pair)
            else:
                logger.warning(f"⚠️  {pair} not found on {self.exchange_name}")
        
        logger.info(f"✅ Loaded {len(pairs)} pairs: {pairs[:5]}...")
        return pairs
    
    def download_ohlcv(
        self,
        pair: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 5,
    ) -> pd.DataFrame:
        """Download OHLCV data with retry logic.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe ("15m", "1h", "4h", "1d")
            start_date: Start date
            end_date: End date
            max_retries: Max retry attempts
            
        Returns:
            DataFrame with OHLCV data
        """
        since = int(start_date.timestamp() * 1000)  # milliseconds
        end_ms = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        retry_count = 0
        
        while current_since < end_ms:
            try:
                # Download batch of 1000 candles (Binance limit)
                logger.info(f"📄 Downloading {pair} {timeframe} from {datetime.fromtimestamp(current_since/1000)}")
                
                ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, since=current_since, limit=1000)
                
                if not ohlcv:
                    logger.info(f"✅ {pair} {timeframe}: Completed")
                    break
                
                all_candles.extend(ohlcv)
                
                # Next batch starts from last candle
                current_since = int(ohlcv[-1][0]) + self._get_timeframe_ms(timeframe)
                
                # Reset retry count on success
                retry_count = 0
                
                # Rate limit: 1200 calls/min = 50ms per call
                time.sleep(0.05)
                
            except ccxt.RateLimitExceeded as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # exponential backoff
                
                if retry_count >= max_retries:
                    logger.error(f"❌ Max retries exceeded for {pair} {timeframe}")
                    raise
                
                logger.warning(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"❌ Error downloading {pair} {timeframe}: {e}")
                raise
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        df['symbol'] = pair
        df['timeframe'] = timeframe
        
        logger.info(f"✅ {pair} {timeframe}: {len(df)} candles (from {df.index[0]} to {df.index[-1]})")
        
        return df
    
    @staticmethod
    def _get_timeframe_ms(timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        mapping = {
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return mapping.get(timeframe, 60 * 60 * 1000)
    
    def download_batch(
        self,
        pairs: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Download multiple pairs and timeframes.
        
        Args:
            pairs: List of trading pairs
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict keyed by (pair, timeframe)
        """
        results = {}
        total = len(pairs) * len(timeframes)
        current = 0
        
        for pair in pairs:
            for timeframe in timeframes:
                current += 1
                logger.info(f"📄 [{current}/{total}] Downloading {pair} {timeframe}")
                
                try:
                    df = self.download_ohlcv(pair, timeframe, start_date, end_date)
                    results[(pair, timeframe)] = df
                except Exception as e:
                    logger.error(f"❌ Failed: {pair} {timeframe} - {e}")
                    continue
        
        logger.info(f"✅ Download complete: {len(results)}/{total} successful")
        return results
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        pair: str,
        timeframe: str,
        output_dir: str,
    ) -> str:
        """Save dataframe to Parquet format.
        
        Parquet is 10x smaller than CSV and preserves dtypes.
        
        Args:
            df: DataFrame to save
            pair: Trading pair (for filename)
            timeframe: Timeframe (for filename)
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filename: BTC-USDT_15m.parquet
        filename = f"{pair.replace('/', '-')}_{timeframe}.parquet"
        filepath = os.path.join(output_dir, filename)
        
        df.to_parquet(filepath, compression='snappy')
        
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        logger.info(f"💾 Saved {pair} {timeframe} ({len(df)} rows, {size_mb:.2f} MB)")
        
        return filepath


def download_main(
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "src/data/jan2025_download/ohlcv",
) -> Dict[str, str]:
    """Main download function.
    
    Args:
        symbols: List of symbols (e.g., ["BTC", "ETH", "SOL"])
        timeframes: List of timeframes (e.g., ["15m", "1h", "4h", "1d"])
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        
    Returns:
        Dict of saved files
    """
    downloader = CCXTDownloader("binance")
    pairs = downloader.get_symbol_pairs(symbols)
    
    # Download all
    data = downloader.download_batch(pairs, timeframes, start_date, end_date)
    
    # Save to parquet
    saved_files = {}
    for (pair, timeframe), df in data.items():
        filepath = downloader.save_to_parquet(df, pair, timeframe, output_dir)
        saved_files[f"{pair}_{timeframe}"] = filepath
    
    # Save metadata
    metadata = {
        "download_date": datetime.now().isoformat(),
        "exchange": "binance",
        "symbols": symbols,
        "timeframes": timeframes,
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "files_count": len(saved_files),
        "total_candles": sum(len(df) for df in data.values()),
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Metadata saved to {metadata_path}")
    
    return saved_files


if __name__ == "__main__":
    from config import VALID_SYMBOLS, TIMEFRAMES, TEST_START_DATE, TEST_END_DATE
    
    logger.info("🚀 Starting OHLCV download...")
    logger.info(f"Symbols: {VALID_SYMBOLS[:3]}... (first 3)")
    logger.info(f"Timeframes: {TIMEFRAMES}")
    logger.info(f"Date range: {TEST_START_DATE} to {TEST_END_DATE}")
    
    # Quick test: download BTC and ETH for last month
    files = download_main(
        symbols=["BTC", "ETH"],
        timeframes=["1h", "4h"],
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
    )
    
    logger.info(f"✅ Download complete! {len(files)} files saved.")
