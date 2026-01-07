"""Vectorized Backtesting Engine.

Jim Simons principle: Fast, realistic backtesting with proper risk metrics.

Features:
- Vectorized operations (NumPy-based, not for-loops)
- Realistic transaction costs (10-20 bps)
- Slippage modeling
- Proper sharpe ratio calculation
- Bootstrap confidence intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """Fast, vectorized backtest engine."""
    
    def __init__(
        self,
        initial_cash: float = 10000,
        commission: float = 0.001,  # 0.1% per trade
        slippage_bps: float = 10,    # 10 basis points
    ):
        """Initialize backtest engine.
        
        Args:
            initial_cash: Starting capital
            commission: Commission per trade (0.1%)
            slippage_bps: Slippage in basis points (0.1%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        factor_df: pd.DataFrame,
        holding_period: int = 4,
    ) -> Dict:
        """Run vectorized backtest.
        
        Args:
            df: DataFrame with OHLCV data
            factor_df: DataFrame with signals (signal column: 1=BUY, -1=SELL, 0=HOLD)
            holding_period: How many bars to hold (default 4 hours)
            
        Returns:
            Dict with backtest results
        """
        df = df.copy()
        df['signal'] = factor_df['signal']
        
        # Generate entry/exit signals
        df['position'] = 0
        df.loc[df['signal'] == 1, 'position'] = 1   # Long
        df.loc[df['signal'] == -1, 'position'] = -1 # Short
        
        # Forward-fill position for holding period
        df['position'] = df['position'].replace(0, np.nan).fillna(method='ffill', limit=holding_period)
        df['position'] = df['position'].fillna(0)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Apply slippage on entry
        df['entry_price'] = df['close'] * (1 + df['position'] * self.slippage_bps)
        
        # Strategy returns (position * returns - costs)
        df['trade_cost'] = 0
        entry_signals = (df['position'].diff() != 0) & (df['position'] != 0)
        df.loc[entry_signals, 'trade_cost'] = self.commission + self.slippage_bps
        
        df['strategy_returns'] = (df['position'] * df['returns']) - df['trade_cost']
        
        # Equity curve
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['equity'] = self.initial_cash * df['cumulative_returns']
        
        # Statistics
        stats = self._calculate_stats(df)
        
        return {
            "df": df,
            "stats": stats,
            "signals": {
                "total_trades": entry_signals.sum(),
                "long_count": (df['position'] == 1).sum(),
                "short_count": (df['position'] == -1).sum(),
            }
        }
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate performance statistics.
        
        Args:
            df: DataFrame with equity curve
            
        Returns:
            Dict with statistics
        """
        returns = df['strategy_returns'].dropna()
        equity = df['equity']
        
        # Basic stats
        total_return = (equity.iloc[-1] / self.initial_cash) - 1
        annual_return = total_return * 252 / len(df) * 252  # Annualize
        
        # Volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        profit_factor = gains / abs(losses) if losses < 0 else 0
        
        stats = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": total_trades,
            "final_equity": equity.iloc[-1],
        }
        
        return stats
    
    def print_stats(self, stats: Dict):
        """Print backtest statistics."""
        logger.info(f"\n" + "="*50)
        logger.info(f"BACKTEST RESULTS")
        logger.info(f"="*50)
        logger.info(f"Total Return:    {stats['total_return']:>10.2%}")
        logger.info(f"Annual Return:   {stats['annual_return']:>10.2%}")
        logger.info(f"Volatility:      {stats['volatility']:>10.2%}")
        logger.info(f"Sharpe Ratio:    {stats['sharpe_ratio']:>10.2f}")
        logger.info(f"Max Drawdown:    {stats['max_drawdown']:>10.2%}")
        logger.info(f"Win Rate:        {stats['win_rate']:>10.1%}")
        logger.info(f"Profit Factor:   {stats['profit_factor']:>10.2f}")
        logger.info(f"Num Trades:      {stats['num_trades']:>10.0f}")
        logger.info(f"Final Equity:    ${stats['final_equity']:>10,.0f}")
        logger.info(f"="*50)
    
    @staticmethod
    def bootstrap_ci(returns: pd.Series, num_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Sharpe ratio.
        
        Args:
            returns: Strategy returns
            num_bootstrap: Number of bootstrap samples
            ci: Confidence interval (0.95 = 95%)
            
        Returns:
            (lower_bound, upper_bound) for Sharpe ratio
        """
        annual_sharpes = []
        
        for _ in range(num_bootstrap):
            # Sample with replacement
            sampled = returns.sample(n=len(returns), replace=True)
            sharpe = (sampled.mean() * 252) / (sampled.std() * np.sqrt(252))
            annual_sharpes.append(sharpe)
        
        alpha = (1 - ci) / 2
        lower = np.percentile(annual_sharpes, alpha * 100)
        upper = np.percentile(annual_sharpes, (1 - alpha) * 100)
        
        return lower, upper


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range("2024-01-01", periods=252, freq="1H")
    prices = 40000 + np.cumsum(np.random.normal(0, 50, 252))
    
    df = pd.DataFrame({
        "open": prices + np.random.normal(0, 10, 252),
        "high": prices + np.random.uniform(0, 100, 252),
        "low": prices - np.random.uniform(0, 100, 252),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, 252),
    }, index=dates)
    
    # Simple signals
    signals_df = pd.DataFrame({
        "signal": np.random.choice([1, -1, 0], 252, p=[0.3, 0.3, 0.4])
    }, index=dates)
    
    # Run backtest
    backtester = VectorizedBacktester(initial_cash=10000)
    result = backtester.run_backtest(df, signals_df, holding_period=4)
    
    backtester.print_stats(result['stats'])
