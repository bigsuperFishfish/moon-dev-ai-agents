"""Position Sizer v3: Volatility-targeted position sizing with rate limiting.

Formula: pos_raw = (target_vol / realized_vol) * sign(score)
Then: pos = clip(pos_raw, -1, 1) with delta smoothing to prevent excessive rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class PositionSizerV3:
    """Volume-targeted position sizing engine."""
    
    @staticmethod
    def calculate_raw_position(
        score: np.ndarray,
        realized_vol: np.ndarray,
        target_vol: float = 0.10,
    ) -> np.ndarray:
        """
        Calculate raw position from score and realized volatility.
        
        Formula: pos_raw = (target_vol / realized_vol) * sign(score)
        
        This scales position by the inverse of realized vol:
        - Low vol -> larger position to maintain consistent volatility
        - High vol -> smaller position to prevent excessive risk
        
        Args:
            score: Factor score array (continuous, -1 to 1)
            realized_vol: Realized volatility array (annualized)
            target_vol: Target volatility (annualized)
            
        Returns:
            raw position array (unbounded)
        """
        # Avoid division by zero
        realized_vol = np.maximum(realized_vol, 1e-9)
        
        # Vol scaling factor
        vol_scale = target_vol / realized_vol
        
        # Apply sign to scores to get direction
        side = np.sign(score)
        
        # Raw position = vol_scale * |score| * sign
        # Clamp score to [-1, 1] to avoid extreme positions
        score_clipped = np.clip(score, -1.0, 1.0)
        pos_raw = vol_scale * score_clipped
        
        return pos_raw
    
    @staticmethod
    def clip_and_smooth(
        pos_raw: np.ndarray,
        delta_pos_max: float = 0.1,
        apply_smoothing: bool = True,
    ) -> np.ndarray:
        """
        Clip position to [-1, 1] and apply delta smoothing to prevent excessive rebalancing.
        
        Args:
            pos_raw: Raw unbounded position array
            delta_pos_max: Max position change per bar (e.g., 0.1 = max ±0.1 change)
            apply_smoothing: Whether to apply delta smoothing
            
        Returns:
            smoothed position array
        """
        # Clip to [-1, 1]
        pos_clipped = np.clip(pos_raw, -1.0, 1.0)
        
        if not apply_smoothing:
            return pos_clipped
        
        # Apply delta smoothing to prevent excessive rebalancing
        pos_smooth = np.zeros_like(pos_clipped)
        pos_smooth[0] = pos_clipped[0]
        
        for i in range(1, len(pos_clipped)):
            # Max move from previous position
            max_pos = pos_smooth[i-1] + delta_pos_max
            min_pos = pos_smooth[i-1] - delta_pos_max
            
            # Clip to max move
            pos_smooth[i] = np.clip(pos_clipped[i], min_pos, max_pos)
        
        return pos_smooth
    
    @staticmethod
    def calculate_position(
        score: np.ndarray,
        realized_vol: np.ndarray,
        target_vol: float = 0.10,
        delta_pos_max: float = 0.1,
        apply_smoothing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete position sizing: vol-targeted -> clipped -> smoothed.
        
        Args:
            score: Factor score array
            realized_vol: Realized volatility array
            target_vol: Target volatility
            delta_pos_max: Max position change per bar
            apply_smoothing: Apply delta smoothing
            
        Returns:
            (position_array, position_raw_array) for debugging
        """
        # Step 1: Vol-targeted sizing
        pos_raw = PositionSizerV3.calculate_raw_position(
            score, realized_vol, target_vol
        )
        
        # Step 2: Clip and smooth
        pos_final = PositionSizerV3.clip_and_smooth(
            pos_raw, delta_pos_max, apply_smoothing
        )
        
        return pos_final, pos_raw
    
    @staticmethod
    def calculate_notional_leverage(
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate notional leverage from position array.
        
        For long/short positions, leverage = |position|.
        This is useful for monitoring actual risk exposure.
        
        Args:
            position: Position array [-1, 1]
            
        Returns:
            leverage array (always >= 0)
        """
        return np.abs(position)
    
    @staticmethod
    def calculate_position_changes(
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate position changes between bars.
        Useful for calculating transaction costs.
        
        Args:
            position: Position array
            
        Returns:
            delta position array
        """
        delta = np.diff(position, prepend=position[0])
        return delta


class PositionSizerV3WithEMA:
    """Alternative: Use EMA smoothing instead of delta limiting."""
    
    @staticmethod
    def smooth_with_ema(
        pos_clipped: np.ndarray,
        ema_alpha: float = 0.2,  # Smoothing factor (0-1, higher = more responsive)
    ) -> np.ndarray:
        """
        Apply EMA smoothing to position changes.
        More responsive than delta limiting.
        
        Args:
            pos_clipped: Clipped position array
            ema_alpha: EMA smoothing factor
            
        Returns:
            EMA-smoothed position
        """
        pos_ema = np.zeros_like(pos_clipped)
        pos_ema[0] = pos_clipped[0]
        
        for i in range(1, len(pos_clipped)):
            pos_ema[i] = ema_alpha * pos_clipped[i] + (1 - ema_alpha) * pos_ema[i-1]
        
        return pos_ema
    
    @staticmethod
    def calculate_position_with_ema(
        score: np.ndarray,
        realized_vol: np.ndarray,
        target_vol: float = 0.10,
        ema_alpha: float = 0.2,
    ) -> np.ndarray:
        """
        Position sizing with EMA smoothing.
        
        Args:
            score: Factor score array
            realized_vol: Realized volatility array
            target_vol: Target volatility
            ema_alpha: EMA smoothing factor
            
        Returns:
            final position array
        """
        # Vol-targeted sizing
        pos_raw = PositionSizerV3.calculate_raw_position(
            score, realized_vol, target_vol
        )
        
        # Clip
        pos_clipped = np.clip(pos_raw, -1.0, 1.0)
        
        # EMA smooth
        pos_ema = PositionSizerV3WithEMA.smooth_with_ema(
            pos_clipped, ema_alpha
        )
        
        return pos_ema


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 100
    score = np.sin(np.linspace(0, 4*np.pi, n))  # Oscillating score
    realized_vol = np.ones(n) * 0.10 + np.random.normal(0, 0.02, n)  # ~10% vol with noise
    
    # Calculate position
    pos, pos_raw = PositionSizerV3.calculate_position(
        score, realized_vol, target_vol=0.10, delta_pos_max=0.1
    )
    
    print(f"Score range: [{score.min():.2f}, {score.max():.2f}]")
    print(f"Vol range: [{realized_vol.min():.4f}, {realized_vol.max():.4f}]")
    print(f"Position range: [{pos.min():.2f}, {pos.max():.2f}]")
    print(f"Max delta pos: {np.abs(np.diff(pos)).max():.4f}")
