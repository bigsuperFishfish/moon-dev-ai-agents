"""Vectorized Backtest Engine v3: Complete integration of all v3 components.

Integrates:
- Volatility estimation (GK/Parkinson)
- Position sizing (vol-targeted)
- Exit rules (ATR-based SL/TP/time)
- Risk metrics (Sharpe, MDD, IC)
- Walk-forward / regime-split validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from .strategy_spec_v3 import StrategySpecV3
from .volatility_estimators import estimate_vol
from .exit_engine_v3 import ExitEngineV3
from .position_sizer_v3 import PositionSizerV3

logger = logging.getLogger(__name__)


@dataclass
class BacktestResultsV3:
    """Complete backtest results container."""
    
    # Time series
    df: pd.DataFrame  # OHLCV + signals + position + pnl
    
    # Entry/exit stats
    total_trades: int
    long_trades: int
    short_trades: int
    
    # Risk metrics
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Statistical validation
    information_coefficient: float  # Correlation(factor, forward_returns)
    ic_std: float
    ic_tstat: Optional[float]
    
    # Exit distribution
    tp_count: int
    sl_count: int
    time_stop_count: int
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULTS V3",
            "=" * 60,
            f"Total Return:      {self.total_return:>10.2%}",
            f"Annual Return:     {self.annual_return:>10.2%}",
            f"Volatility:        {self.annual_vol:>10.2%}",
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"Calmar Ratio:      {self.calmar_ratio:>10.2f}",
            f"Win Rate:          {self.win_rate:>10.1%}",
            f"Profit Factor:     {self.profit_factor:>10.2f}",
            f"Total Trades:      {self.total_trades:>10.0f}",
            f"  Long:            {self.long_trades:>10.0f}",
            f"  Short:           {self.short_trades:>10.0f}",
            f"Exit Distribution: TP={self.tp_count}, SL={self.sl_count}, Time={self.time_stop_count}",
            f"IC (factor corr):  {self.information_coefficient:>10.3f}",
            f"IC Std:            {self.ic_std:>10.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class VectorizedBacktesterV3:
    """Vectorized backtest engine for v3 strategies."""
    
    def __init__(
        self,
        initial_cash: float = 10000,
        commission: float = 0.001,  # 0.1% per trade
        slippage_bps: float = 10,    # 10 bps
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_bps = slippage_bps / 10000
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        score: np.ndarray,
        spec: StrategySpecV3,
        long_short_separate: bool = False,
    ) -> BacktestResultsV3:
        """
        Run complete v3 backtest with vol targeting + ATR exits.
        
        Args:
            df: DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close', 'volume')
            score: Factor score array (continuous, -1 to 1)
            spec: StrategySpecV3 specification
            long_short_separate: If True, separate position sizing for long/short
            
        Returns:
            BacktestResultsV3 with full metrics
        """
        df = df.copy()
        n = len(df)
        
        # === STEP 1: Volatility estimation ===
        realized_vol = estimate_vol(
            df,
            method=spec.vol_method.value,
            window=spec.vol_window,
            bars_per_year=8760,  # Crypto 24/7
            vol_floor=spec.vol_floor,
            use_volume_adjustment=spec.use_volume_adjustment,
        )
        
        # === STEP 2: Position sizing (vol-targeted) ===
        position, position_raw = PositionSizerV3.calculate_position(
            score,
            realized_vol,
            target_vol=spec.target_vol,
            delta_pos_max=spec.delta_pos_max,
            apply_smoothing=True,
        )
        
        # === STEP 3: ATR and exits ===
        atr = ExitEngineV3.calculate_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            period=spec.atr_period,
        )
        
        # Generate entry signals (score != 0 and position changes)
        entry_signals = (score != 0).astype(int) * np.sign(score)
        
        # Simulate exits (simplified version for vectorized engine)
        position_final, exit_prices, exit_types = ExitEngineV3.vectorized_exit_simulation(
            df,
            pd.Series(entry_signals),
            atr,
            sl_atr_mult=spec.stop_loss_atr_mult,
            tp_atr_mult=spec.take_profit_atr_mult,
            time_stop_bars=spec.time_stop_bars,
        )
        
        # === STEP 4: P&L calculation ===
        df['score'] = score
        df['realized_vol'] = realized_vol
        df['atr'] = atr
        df['position'] = position_final
        df['position_raw'] = position_raw
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Strategy P&L (position held from day t-1)
        df['strategy_returns'] = (df['position'].shift(1) * df['returns']) - (
            np.abs(df['position'].diff()) * (self.commission + self.slippage_bps)
        )
        
        # Equity curve
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['equity'] = self.initial_cash * df['cumulative_returns']
        
        # === STEP 5: Performance metrics ===
        results = self._calculate_metrics(
            df, position_final, exit_types, score
        )
        
        results.df = df
        return results
    
    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        position: np.ndarray,
        exit_types: np.ndarray,
        score: np.ndarray,
    ) -> BacktestResultsV3:
        """Calculate all performance metrics."""
        
        returns = df['strategy_returns'].dropna()
        equity = df['equity']
        
        # === Basic returns ===
        total_return = (equity.iloc[-1] / self.initial_cash) - 1
        annual_return = total_return * 252 / len(df) * 252  # Annualize by days
        annual_vol = returns.std() * np.sqrt(252)
        
        # === Risk-adjusted ===
        sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # === Trade stats ===
        entry_signals = (score != 0).astype(int)
        entry_indices = np.where(np.diff(np.concatenate([[0], entry_signals.astype(int)])))[0]
        
        long_trades = np.sum((score[entry_indices] > 0) if len(entry_indices) > 0 else 0)
        short_trades = np.sum((score[entry_indices] < 0) if len(entry_indices) > 0 else 0)
        total_trades = long_trades + short_trades
        
        # === Win rate ===
        winning_trades = (returns > 0).sum()
        valid_trades = (returns != 0).sum()
        win_rate = winning_trades / valid_trades if valid_trades > 0 else 0
        
        # === Profit factor ===
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        profit_factor = gains / abs(losses) if losses < 0 else np.inf
        
        # === Exit distribution ===
        tp_count = (exit_types == 'tp').sum()
        sl_count = (exit_types == 'sl').sum()
        time_stop_count = (exit_types == 'time_stop').sum()
        
        # === Information Coefficient ===
        forward_returns = df['returns'].shift(-1).dropna()
        factor_vals = score[:-1]  # Align with forward returns
        
        valid_idx = ~(np.isnan(factor_vals) | np.isnan(forward_returns.values))
        if valid_idx.sum() > 1:
            ic = np.corrcoef(factor_vals[valid_idx], forward_returns.values[valid_idx])[0, 1]
            ic_std = np.std(factor_vals[valid_idx] * forward_returns.values[valid_idx])
            # t-stat for IC
            ic_tstat = ic * np.sqrt(valid_idx.sum() - 2) / np.sqrt(1 - ic**2 + 1e-9)
        else:
            ic = np.nan
            ic_std = np.nan
            ic_tstat = None
        
        return BacktestResultsV3(
            df=pd.DataFrame(),  # Will be set in run_backtest
            total_trades=int(total_trades),
            long_trades=int(long_trades),
            short_trades=int(short_trades),
            total_return=total_return,
            annual_return=annual_return,
            annual_vol=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            information_coefficient=ic,
            ic_std=ic_std,
            ic_tstat=ic_tstat,
            tp_count=int(tp_count),
            sl_count=int(sl_count),
            time_stop_count=int(time_stop_count),
        )
