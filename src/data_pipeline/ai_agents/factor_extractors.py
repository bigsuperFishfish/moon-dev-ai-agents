"""Factor Extractors: Common entry factors for RBI screening.

Each factor returns a score in [-1, 1]:
  - 1.0 = maximum long signal
  - 0.0 = no signal
  - -1.0 = maximum short signal

All factors are vectorized for speed.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FactorExtractors:
    """Static factory for common trading factors."""
    
    @staticmethod
    def rsi_mean_reversion(
        close: np.ndarray,
        period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
    ) -> np.ndarray:
        """
        RSI-based mean reversion.
        
        Returns:
            1.0 when RSI < oversold_threshold (buy oversold)
            -1.0 when RSI > overbought_threshold (sell overbought)
            0.0 otherwise
            
        Args:
            close: Close prices
            period: RSI lookback period
            oversold_threshold: RSI threshold for long signal
            overbought_threshold: RSI threshold for short signal
        """
        # Calculate RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period+1])
        avg_loss[period] = np.mean(loss[:period+1])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # Generate score
        score = np.zeros_like(rsi)
        score[rsi < oversold_threshold] = 1.0
        score[rsi > overbought_threshold] = -1.0
        
        return score
    
    @staticmethod
    def moving_average_crossover(
        close: np.ndarray,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> np.ndarray:
        """
        MA crossover momentum.
        
        Returns:
            1.0 when fast MA > slow MA (uptrend)
            -1.0 when fast MA < slow MA (downtrend)
            0.0 during transitions
        """
        # Calculate MAs (vectorized)
        fast_ma = pd.Series(close).rolling(window=fast_period).mean().values
        slow_ma = pd.Series(close).rolling(window=slow_period).mean().values
        
        # Generate score
        score = np.zeros_like(close, dtype=float)
        score[fast_ma > slow_ma] = 1.0
        score[fast_ma < slow_ma] = -1.0
        
        # Avoid transitions
        score[np.isnan(fast_ma) | np.isnan(slow_ma)] = 0
        
        return score
    
    @staticmethod
    def momentum(
        close: np.ndarray,
        period: int = 20,
        threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Price momentum (rate of change).
        
        Returns:
            1.0 when momentum > threshold (strong up)
            -1.0 when momentum < -threshold (strong down)
            0.0 otherwise
        """
        # Calculate momentum (ROC)
        momentum_vals = (close - np.roll(close, period)) / np.roll(close, period)
        momentum_vals[np.isnan(momentum_vals)] = 0
        
        # Generate score
        score = np.zeros_like(momentum_vals)
        score[momentum_vals > threshold] = 1.0
        score[momentum_vals < -threshold] = -1.0
        
        return score
    
    @staticmethod
    def bollinger_breakout(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> np.ndarray:
        """
        Bollinger Band breakout.
        
        Returns:
            1.0 when price breaks above upper BB (buy breakout)
            -1.0 when price breaks below lower BB (sell breakdown)
            0.0 otherwise
        """
        # Calculate BB
        sma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        upper_bb = sma + num_std * std
        lower_bb = sma - num_std * std
        
        # Generate score
        score = np.zeros_like(close, dtype=float)
        score[high > upper_bb] = 1.0
        score[low < lower_bb] = -1.0
        
        return score
    
    @staticmethod
    def atr_expansion(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr_period: int = 14,
        expansion_threshold: float = 1.2,
    ) -> np.ndarray:
        """
        ATR expansion (volatility breakout).
        
        Returns:
            1.0 when current TR > expansion_threshold * average TR
            -1.0 when current TR < expansion_threshold * average TR
            0.0 otherwise
        """
        # Calculate TR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR
        atr = pd.Series(tr).rolling(window=atr_period).mean().values
        
        # Generate score
        score = np.zeros_like(tr, dtype=float)
        score[tr > expansion_threshold * atr] = 1.0
        score[tr < (1 / expansion_threshold) * atr] = -1.0
        
        return score
    
    @staticmethod
    def zscore_mean_reversion(
        close: np.ndarray,
        period: int = 20,
        zscore_threshold: float = 2.0,
    ) -> np.ndarray:
        """
        Z-score mean reversion.
        
        Returns:
            1.0 when price is > zscore_threshold std below mean (buy dip)
            -1.0 when price is > zscore_threshold std above mean (sell rip)
            0.0 otherwise
        """
        sma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        zscore = (close - sma) / (std + 1e-9)
        
        # Generate score
        score = np.zeros_like(zscore)
        score[zscore < -zscore_threshold] = 1.0
        score[zscore > zscore_threshold] = -1.0
        
        return score
    
    @staticmethod
    def high_low_breakout(
        high: np.ndarray,
        low: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        """
        High/Low breakout over lookback period.
        
        Returns:
            1.0 when price makes new high
            -1.0 when price makes new low
            0.0 otherwise
        """
        # Calculate rolling high/low
        rolling_high = pd.Series(high).rolling(window=period).max().values
        rolling_low = pd.Series(low).rolling(window=period).min().values
        
        # Generate score
        score = np.zeros_like(high, dtype=float)
        score[high >= rolling_high] = 1.0
        score[low <= rolling_low] = -1.0
        
        return score
    
    @staticmethod
    def volume_weighted_momentum(
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 10,
        vol_threshold_percentile: float = 75,
    ) -> np.ndarray:
        """
        Volume-weighted momentum: Only trade when volume is high.
        
        Returns:
            Score if volume > percentile threshold, else 0
        """
        # Calculate momentum
        mom = (close - np.roll(close, period)) / np.roll(close, period)
        
        # Calculate volume threshold
        vol_threshold = np.percentile(volume, vol_threshold_percentile)
        
        # Generate score
        score = np.zeros_like(mom)
        high_vol = volume > vol_threshold
        score[high_vol & (mom > 0)] = 1.0
        score[high_vol & (mom < 0)] = -1.0
        
        return score
    
    @staticmethod
    def close_to_range(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 5,
    ) -> np.ndarray:
        """
        Close-to-range indicator (0-1 where 0=low, 1=high).
        
        Useful for intrabar:
        - > 0.7 = price near top (short signal)
        - < 0.3 = price near bottom (long signal)
        """
        rolling_high = pd.Series(high).rolling(window=period).max().values
        rolling_low = pd.Series(low).rolling(window=period).min().values
        
        # Close-to-range (0-1)
        ctr = (close - rolling_low) / (rolling_high - rolling_low + 1e-9)
        ctr[np.isnan(ctr)] = 0.5
        
        # Convert to score (-1 to 1)
        score = 2 * (ctr - 0.5)  # Center at 0, range [-1, 1]
        # Squash to make signal clearer
        score = np.sign(score) * np.clip(np.abs(score), 0, 1)
        
        return score


def create_factor_from_description(
    factor_type: str,
    df: pd.DataFrame,
    **kwargs,
) -> np.ndarray:
    """
    Factory function to create factor score from description.
    
    Args:
        factor_type: Type of factor (e.g., "rsi_mean_reversion", "ma_crossover")
        df: DataFrame with OHLCV data
        **kwargs: Factor-specific parameters
        
    Returns:
        Score array [-1 to 1]
    """
    extractors = {
        "rsi_mean_reversion": FactorExtractors.rsi_mean_reversion,
        "rsi": FactorExtractors.rsi_mean_reversion,
        "ma_crossover": FactorExtractors.moving_average_crossover,
        "momentum": FactorExtractors.momentum,
        "bollinger_breakout": FactorExtractors.bollinger_breakout,
        "atr_expansion": FactorExtractors.atr_expansion,
        "zscore_mean_reversion": FactorExtractors.zscore_mean_reversion,
        "high_low_breakout": FactorExtractors.high_low_breakout,
        "volume_weighted_momentum": FactorExtractors.volume_weighted_momentum,
        "close_to_range": FactorExtractors.close_to_range,
    }
    
    if factor_type not in extractors:
        raise ValueError(
            f"Unknown factor type: {factor_type}. "
            f"Available: {list(extractors.keys())}"
        )
    
    extractor = extractors[factor_type]
    
    # Call with appropriate columns
    if "close" in df.columns:
        kwargs.setdefault("close", df["close"].values)
    if "high" in df.columns:
        kwargs.setdefault("high", df["high"].values)
    if "low" in df.columns:
        kwargs.setdefault("low", df["low"].values)
    if "volume" in df.columns:
        kwargs.setdefault("volume", df["volume"].values)
    
    return extractor(**kwargs)
