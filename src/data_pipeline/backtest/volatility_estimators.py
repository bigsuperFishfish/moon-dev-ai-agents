"""Volatility Estimators for v3: Garman-Klass (default) + Parkinson (fallback).

Jim Simons principle: Use range-based estimators that utilize OHLC efficiently.
Garman-Klass captures drift + range; Parkinson uses HL only.

Crypto markets 24/7, so annualization uses 365*24 = 8760 hourly bars/year.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityEstimator:
    """Base volatility estimator."""
    
    @staticmethod
    def gk_variance(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Garman-Klass variance estimator (bar-wise).
        
        Formula: var_t = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
        
        Args:
            open_: Open prices (array)
            high: High prices (array)
            low: Low prices (array)
            close: Close prices (array)
            
        Returns:
            variance array (clipped to >= 0)
        """
        hl = np.log(high / low)
        co = np.log(close / open_)
        
        # GK formula
        var = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        
        # Ensure non-negative (GK can go negative)
        var = np.maximum(var, 0.0)
        
        return var
    
    @staticmethod
    def parkinson_variance(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Parkinson variance estimator (bar-wise, uses HL only).
        
        Formula: var_t = (ln(H/L))^2 / (4*ln(2))
        
        Args:
            high: High prices (array)
            low: Low prices (array)
            
        Returns:
            variance array
        """
        hl = np.log(high / low)
        var = (hl ** 2) / (4 * np.log(2))
        return var
    
    @staticmethod
    def realized_volatility(
        var_series: np.ndarray,
        window: int = 20,
        bars_per_year: int = 8760,  # 365*24 for hourly crypto
        vol_floor: float = 0.001,   # 0.1% annualized minimum
    ) -> np.ndarray:
        """
        Convert bar variances to rolling realized volatility (annualized).
        
        Args:
            var_series: Bar-wise variances
            window: Rolling window (bars)
            bars_per_year: Annualization factor (8760 for crypto hourly)
            vol_floor: Minimum volatility threshold (annualized)
            
        Returns:
            realized_vol array (annualized)
        """
        # Use pandas rolling mean for efficiency
        rolling_mean_var = pd.Series(var_series).rolling(window=window, min_periods=1).mean().values
        
        # Annualize: sqrt(mean_var * bars_per_year)
        realized_vol = np.sqrt(np.maximum(rolling_mean_var, 0.0) * bars_per_year)
        
        # Apply floor
        realized_vol = np.maximum(realized_vol, vol_floor)
        
        return realized_vol


class VolumeAdjustedEstimator:
    """Optional: volume-weighted volatility (for advanced use)."""
    
    @staticmethod
    def gk_volume_adjusted(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        volume_window: int = 20,
    ) -> np.ndarray:
        """
        Garman-Klass adjusted by volume volatility.
        Reduces vol estimate if volume is low (less reliable pricing).
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            volume: Trading volume
            volume_window: Rolling window for volume baseline
            
        Returns:
            adjusted variance array
        """
        # Base GK variance
        var = VolatilityEstimator.gk_variance(open_, high, low, close)
        
        # Volume adjustment factor: current_vol / avg_vol
        avg_volume = pd.Series(volume).rolling(window=volume_window, min_periods=1).mean().values
        vol_adjustment = volume / (avg_volume + 1e-9)  # Avoid division by zero
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Bound adjustment
        
        # Apply adjustment
        adjusted_var = var * vol_adjustment
        
        return np.maximum(adjusted_var, 0.0)


def estimate_vol(
    df: pd.DataFrame,
    method: str = "gk",
    window: int = 20,
    bars_per_year: int = 8760,
    vol_floor: float = 0.001,
    use_volume_adjustment: bool = False,
) -> np.ndarray:
    """
    Estimate realized volatility for a dataframe.
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        method: 'gk' (Garman-Klass) or 'parkinson'
        window: Rolling window (bars)
        bars_per_year: Annualization factor
        vol_floor: Minimum volatility
        use_volume_adjustment: Apply volume weighting (GK only)
        
    Returns:
        realized_vol array
    """
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Estimate bar-wise variance
    if method == "gk":
        if use_volume_adjustment:
            volume = df['volume'].values
            var_series = VolumeAdjustedEstimator.gk_volume_adjusted(
                open_, high, low, close, volume
            )
        else:
            var_series = VolatilityEstimator.gk_variance(open_, high, low, close)
    elif method == "parkinson":
        var_series = VolatilityEstimator.parkinson_variance(high, low)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to annualized realized vol
    realized_vol = VolatilityEstimator.realized_volatility(
        var_series, window=window, bars_per_year=bars_per_year, vol_floor=vol_floor
    )
    
    return realized_vol


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="1H")
    close_prices = 40000 + np.cumsum(np.random.normal(0, 50, 252))
    
    df = pd.DataFrame({
        "open": close_prices + np.random.normal(0, 10, 252),
        "high": close_prices + np.random.uniform(0, 100, 252),
        "low": close_prices - np.random.uniform(0, 100, 252),
        "close": close_prices,
        "volume": np.random.uniform(1000, 5000, 252),
    }, index=dates)
    
    # Estimate volatility
    vol_gk = estimate_vol(df, method="gk", window=20, bars_per_year=8760)
    vol_parkinson = estimate_vol(df, method="parkinson", window=20, bars_per_year=8760)
    
    print(f"GK Vol (last 10):")
    print(vol_gk[-10:])
    print(f"\nParkinson Vol (last 10):")
    print(vol_parkinson[-10:])
