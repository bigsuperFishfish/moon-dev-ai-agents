"""Exit Engine v3: ATR-based stop-loss + take-profit + time stop.

Key principles:
1. SL/TP calculated at entry using ATR at entry time (fixed during trade)
2. Time stop as additional exit condition
3. Intrabar fill logic: HIGH/LOW determine if hit before exit signal
4. Long/short asymmetric handling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Information about an exit event."""
    exit_bar_idx: int
    exit_type: str  # 'tp', 'sl', 'time_stop', 'signal_exit'
    exit_price: float
    bars_held: int
    pnl_pct: float
    reason: str


class ExitEngineV3:
    """Manage exits for v3 strategies using ATR-based levels."""
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR array
        """
        tr = np.maximum(
            np.maximum(high - low, np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
        tr[0] = high[0] - low[0]  # First value
        
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        return atr
    
    @staticmethod
    def get_exit_levels(
        entry_idx: int,
        entry_price: float,
        direction: int,  # 1 for long, -1 for short
        atr_at_entry: float,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
    ) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit levels based on entry ATR.
        
        Args:
            entry_idx: Entry bar index (for reference)
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr_at_entry: ATR value at entry
            sl_atr_mult: SL = entry ± (sl_atr_mult * ATR)
            tp_atr_mult: TP = entry ± (tp_atr_mult * ATR)
            
        Returns:
            (stop_loss_price, take_profit_price)
        """
        if direction == 1:  # Long
            sl = entry_price - sl_atr_mult * atr_at_entry
            tp = entry_price + tp_atr_mult * atr_at_entry
        elif direction == -1:  # Short
            sl = entry_price + sl_atr_mult * atr_at_entry
            tp = entry_price - tp_atr_mult * atr_at_entry
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        return sl, tp
    
    @staticmethod
    def check_exit(
        current_idx: int,
        entry_idx: int,
        entry_price: float,
        entry_atr: float,
        direction: int,
        sl_atr_mult: float,
        tp_atr_mult: float,
        time_stop_bars: int,
        high: float,
        low: float,
        close: float,
    ) -> Optional[ExitSignal]:
        """
        Check if position should exit at current bar.
        
        Intrabar logic: HIGH/LOW determine if hit BEFORE close.
        - Long: Hit TP if high >= tp_level, SL if low <= sl_level
        - Short: Hit TP if low <= tp_level, SL if high >= sl_level
        - If both hit same bar: TP takes priority (profit > loss)
        
        Args:
            current_idx: Current bar index
            entry_idx: Entry bar index
            entry_price: Entry price
            entry_atr: ATR at entry
            direction: 1 for long, -1 for short
            sl_atr_mult: SL multiplier
            tp_atr_mult: TP multiplier
            time_stop_bars: Max holding bars
            high: Current bar high
            low: Current bar low
            close: Current bar close
            
        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        bars_held = current_idx - entry_idx
        sl_price, tp_price = ExitEngineV3.get_exit_levels(
            entry_idx, entry_price, direction, entry_atr, sl_atr_mult, tp_atr_mult
        )
        
        # Check time stop first (lowest priority)
        if bars_held >= time_stop_bars:
            pnl_pct = ((close - entry_price) / entry_price * direction) if entry_price > 0 else 0
            return ExitSignal(
                exit_bar_idx=current_idx,
                exit_type="time_stop",
                exit_price=close,
                bars_held=bars_held,
                pnl_pct=pnl_pct,
                reason=f"Time stop reached ({bars_held} bars)",
            )
        
        # Check TP and SL with intrabar logic
        if direction == 1:  # Long
            # Check TP first (prefer profit)
            if high >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                return ExitSignal(
                    exit_bar_idx=current_idx,
                    exit_type="tp",
                    exit_price=tp_price,
                    bars_held=bars_held,
                    pnl_pct=pnl_pct,
                    reason=f"Take profit hit (high {high:.2f} >= tp {tp_price:.2f})",
                )
            # Check SL
            if low <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                return ExitSignal(
                    exit_bar_idx=current_idx,
                    exit_type="sl",
                    exit_price=sl_price,
                    bars_held=bars_held,
                    pnl_pct=pnl_pct,
                    reason=f"Stop loss hit (low {low:.2f} <= sl {sl_price:.2f})",
                )
        
        elif direction == -1:  # Short
            # Check TP first
            if low <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price
                return ExitSignal(
                    exit_bar_idx=current_idx,
                    exit_type="tp",
                    exit_price=tp_price,
                    bars_held=bars_held,
                    pnl_pct=pnl_pct,
                    reason=f"Take profit hit (low {low:.2f} <= tp {tp_price:.2f})",
                )
            # Check SL
            if high >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                return ExitSignal(
                    exit_bar_idx=current_idx,
                    exit_type="sl",
                    exit_price=sl_price,
                    bars_held=bars_held,
                    pnl_pct=pnl_pct,
                    reason=f"Stop loss hit (high {high:.2f} >= sl {sl_price:.2f})",
                )
        
        return None
    
    @staticmethod
    def vectorized_exit_simulation(
        df: pd.DataFrame,
        entry_signals: pd.Series,  # 1 for long, -1 for short, 0 for neutral
        atr: np.ndarray,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
        time_stop_bars: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized exit simulation (for backtest).
        
        Returns:
            (position_array, exit_prices, exit_types)
        """
        n = len(df)
        position = np.zeros(n)
        exit_prices = np.zeros(n)
        exit_types = np.full(n, "", dtype=object)  # 'tp', 'sl', 'time_stop', etc.
        
        entry_idx = None
        entry_price = None
        entry_atr = None
        entry_direction = None
        
        for i in range(n):
            # Check for new entry
            if entry_signals.iloc[i] != 0 and entry_idx is None:
                entry_idx = i
                entry_price = df['close'].iloc[i]
                entry_atr = atr[i]
                entry_direction = int(entry_signals.iloc[i])
                position[i] = entry_direction
            
            # Maintain position
            elif entry_idx is not None:
                position[i] = entry_direction
                
                # Check exit
                exit_signal = ExitEngineV3.check_exit(
                    i, entry_idx, entry_price, entry_atr, entry_direction,
                    sl_atr_mult, tp_atr_mult, time_stop_bars,
                    df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
                )
                
                if exit_signal:
                    position[i] = 0
                    exit_prices[i] = exit_signal.exit_price
                    exit_types[i] = exit_signal.exit_type
                    entry_idx = None
                    entry_price = None
                    entry_atr = None
                    entry_direction = None
        
        return position, exit_prices, exit_types
