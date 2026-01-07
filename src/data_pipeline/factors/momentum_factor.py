"""Simple Momentum Factor for RBI Agent Integration.

Jim Simons principle: Start simple, validate rigorously.

This factor identifies oversold/overbought conditions using RSI.
Used by RBI agent for rapid screening before mathematical formulation.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumFactor:
    """RSI-based momentum factor.
    
    Signals:
    - BUY: RSI < 30 (oversold)
    - SELL: RSI > 70 (overbought)
    """
    
    def __init__(self, period: int = 14):
        """Initialize factor.
        
        Args:
            period: RSI period (default 14)
        """
        self.period = period
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI.
        
        Args:
            prices: Close prices
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Generate trading signals.
        
        Args:
            df: DataFrame with OHLCV
            
        Returns:
            (DataFrame with signals, stats dict)
        """
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.period)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['rsi'] < 30, 'signal'] = 1   # BUY
        df.loc[df['rsi'] > 70, 'signal'] = -1  # SELL
        
        # Calculate factor strength (0 to 1)
        df['factor_strength'] = (100 - df['rsi']) / 100
        df.loc[df['rsi'] < 50, 'factor_strength'] = df.loc[df['rsi'] < 50, 'rsi'] / 100
        
        # Statistics
        stats = {
            "factor": "Momentum (RSI)",
            "period": self.period,
            "buy_signals": (df['signal'] == 1).sum(),
            "sell_signals": (df['signal'] == -1).sum(),
            "signal_frequency": ((df['signal'] != 0).sum() / len(df) * 100),
            "mean_rsi": df['rsi'].mean(),
            "std_rsi": df['rsi'].std(),
        }
        
        return df, stats


class MeanReversionFactor:
    """Bollinger Bands mean reversion factor.
    
    Signals:
    - BUY: Price below lower band
    - SELL: Price above upper band
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """Initialize factor.
        
        Args:
            period: Moving average period
            num_std: Number of standard deviations for bands
        """
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Generate mean reversion signals.
        
        Args:
            df: DataFrame with OHLCV
            
        Returns:
            (DataFrame with signals, stats dict)
        """
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(self.period).mean()
        df['std'] = df['close'].rolling(self.period).std()
        df['upper_band'] = df['sma'] + (self.num_std * df['std'])
        df['lower_band'] = df['sma'] - (self.num_std * df['std'])
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1   # BUY
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # SELL
        
        # Position in band (0 = at lower, 1 = at upper)
        df['band_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        df['band_position'] = df['band_position'].clip(0, 1)
        
        # Statistics
        stats = {
            "factor": "Mean Reversion (Bollinger Bands)",
            "period": self.period,
            "std_dev": self.num_std,
            "buy_signals": (df['signal'] == 1).sum(),
            "sell_signals": (df['signal'] == -1).sum(),
            "signal_frequency": ((df['signal'] != 0).sum() / len(df) * 100),
        }
        
        return df, stats


class TrendFollowingFactor:
    """Simple trend following factor (SMA crossover).
    
    Signals:
    - BUY: Fast SMA > Slow SMA
    - SELL: Fast SMA < Slow SMA
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        """Initialize factor.
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Generate trend following signals.
        
        Args:
            df: DataFrame with OHLCV
            
        Returns:
            (DataFrame with signals, stats dict)
        """
        df = df.copy()
        
        # Calculate moving averages
        df['sma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['sma_slow'] = df['close'].rolling(self.slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1   # UPTREND
        df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1  # DOWNTREND
        
        # Trend strength
        df['trend_strength'] = (df['sma_fast'] - df['sma_slow']).abs() / df['sma_slow']
        
        # Statistics
        stats = {
            "factor": "Trend Following (SMA Crossover)",
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "uptrend_periods": (df['signal'] == 1).sum(),
            "downtrend_periods": (df['signal'] == -1).sum(),
            "signal_changes": (df['signal'].diff() != 0).sum(),
        }
        
        return df, stats


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range("2024-01-01", periods=200, freq="1H")
    prices = 40000 + np.cumsum(np.random.normal(0, 50, 200))
    df = pd.DataFrame({
        "close": prices,
        "open": prices + np.random.normal(0, 10, 200),
        "high": prices + np.random.uniform(0, 100, 200),
        "low": prices - np.random.uniform(0, 100, 200),
        "volume": np.random.uniform(1000, 5000, 200),
    }, index=dates)
    
    # Test momentum factor
    momentum = MomentumFactor(period=14)
    df_mom, stats_mom = momentum.generate_signals(df)
    print(f"\nMomentum Factor: {stats_mom}")
    
    # Test mean reversion factor
    mean_rev = MeanReversionFactor(period=20, num_std=2.0)
    df_mr, stats_mr = mean_rev.generate_signals(df)
    print(f"\nMean Reversion Factor: {stats_mr}")
    
    # Test trend following factor
    trend = TrendFollowingFactor(fast_period=5, slow_period=20)
    df_trend, stats_trend = trend.generate_signals(df)
    print(f"\nTrend Following Factor: {stats_trend}")
