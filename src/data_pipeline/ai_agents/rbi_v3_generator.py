"""RBI v3 Generator: Convert RBI factor descriptions → StrategySpecV3 → Backtest results.

Workflow:
1. User describes factor (text/YouTube/PDF via RBI agent)
2. Extract entry logic → generate score function
3. Create StrategySpecV3 with sensible defaults
4. Auto-backtest with VectorizedEngineV3
5. Return professional backtest report

This bridges the gap between RBI's rapid screening and v3's rigorous backtesting.
"""

import json
import logging
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd

from ..backtest.strategy_spec_v3 import StrategySpecV3, VolMethod
from ..backtest.vectorized_engine_v3 import VectorizedBacktesterV3, BacktestResultsV3
from ..backtest.volatility_estimators import estimate_vol

logger = logging.getLogger(__name__)


class RBIV3Generator:
    """Generate v3 strategies from RBI factor descriptions."""
    
    # Default spec templates for common factor types
    TEMPLATES = {
        "mean_reversion": StrategySpecV3(
            strategy_name="Mean Reversion (v3)",
            entry_factor_name="mean_reversion",
            entry_parameters={"period": 20, "threshold": 1.5},
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=2.5,
            time_stop_bars=15,
            target_vol=0.12,
            vol_window=20,
            vol_floor=0.001,
            delta_pos_max=0.1,
        ),
        "momentum": StrategySpecV3(
            strategy_name="Momentum (v3)",
            entry_factor_name="momentum",
            entry_parameters={"fast_period": 10, "slow_period": 30},
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=3.5,
            time_stop_bars=30,
            target_vol=0.15,
            vol_window=20,
            vol_floor=0.001,
            delta_pos_max=0.12,
        ),
        "breakout": StrategySpecV3(
            strategy_name="Breakout (v3)",
            entry_factor_name="breakout",
            entry_parameters={"lookback": 20, "breakout_thresh": 0.02},
            stop_loss_atr_mult=1.5,
            take_profit_atr_mult=3.0,
            time_stop_bars=25,
            target_vol=0.10,
            vol_window=20,
            vol_floor=0.001,
            delta_pos_max=0.1,
        ),
        "pairs_trade": StrategySpecV3(
            strategy_name="Pairs Trade (v3)",
            entry_factor_name="pairs_zscore",
            entry_parameters={"correlation_window": 60, "zscore_threshold": 2.0},
            stop_loss_atr_mult=2.5,
            take_profit_atr_mult=2.0,
            time_stop_bars=40,
            target_vol=0.08,
            vol_window=30,
            vol_floor=0.001,
            delta_pos_max=0.1,
        ),
    }
    
    def __init__(self):
        """Initialize RBI v3 generator."""
        self.backtester = VectorizedBacktesterV3(initial_cash=10000)
        logger.info("✅ RBIV3Generator initialized")
    
    def create_spec_from_description(
        self,
        factor_description: str,
        factor_type: str = "mean_reversion",
        strategy_name: Optional[str] = None,
        custom_params: Optional[Dict] = None,
    ) -> StrategySpecV3:
        """
        Create StrategySpecV3 from factor description.
        
        Args:
            factor_description: Human-readable factor description
            factor_type: Type of factor (mean_reversion, momentum, breakout, etc.)
            strategy_name: Override strategy name
            custom_params: Override specific parameters
            
        Returns:
            StrategySpecV3 instance
        """
        # Start with template
        if factor_type in self.TEMPLATES:
            spec = self.TEMPLATES[factor_type]
        else:
            logger.warning(f"Unknown factor type: {factor_type}, using mean_reversion")
            spec = self.TEMPLATES["mean_reversion"]
        
        # Convert to dict for modification
        spec_dict = asdict(spec)
        
        # Override with custom params
        if custom_params:
            spec_dict.update(custom_params)
        
        # Override name if provided
        if strategy_name:
            spec_dict["strategy_name"] = strategy_name
        
        # Ensure vol_method is enum
        if isinstance(spec_dict.get("vol_method"), str):
            spec_dict["vol_method"] = VolMethod(spec_dict["vol_method"])
        
        spec_dict["description"] = factor_description
        
        # Create new spec
        new_spec = StrategySpecV3(**spec_dict)
        
        logger.info(f"✅ Created StrategySpecV3: {new_spec.strategy_name}")
        logger.info(f"   Factor: {new_spec.entry_factor_name}")
        logger.info(f"   Target Vol: {new_spec.target_vol:.1%}")
        
        return new_spec
    
    def backtest_spec(
        self,
        df: pd.DataFrame,
        score: np.ndarray,
        spec: StrategySpecV3,
        long_short_separate: bool = False,
    ) -> BacktestResultsV3:
        """
        Run backtest for a StrategySpecV3.
        
        Args:
            df: OHLCV DataFrame
            score: Factor score array (-1 to 1)
            spec: StrategySpecV3 specification
            long_short_separate: Separate long/short position sizing
            
        Returns:
            BacktestResultsV3 with full metrics
        """
        logger.info(f"\n🔄 Backtesting: {spec.strategy_name}")
        logger.info(f"   Symbols: {len(df)} bars")
        logger.info(f"   Score range: [{score.min():.2f}, {score.max():.2f}]")
        
        # Run backtest
        results = self.backtester.run_backtest(
            df=df,
            score=score,
            spec=spec,
            long_short_separate=long_short_separate,
        )
        
        # Log results
        logger.info(f"\n📊 Backtest Results:")
        logger.info(f"   Total Return: {results.total_return:.2%}")
        logger.info(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"   Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"   Win Rate: {results.win_rate:.1%}")
        logger.info(f"   IC (factor): {results.information_coefficient:.3f}")
        logger.info(f"   Trades: {results.total_trades}")
        
        return results
    
    def optimize_spec_parameters(
        self,
        df: pd.DataFrame,
        score: np.ndarray,
        base_spec: StrategySpecV3,
        param_ranges: Optional[Dict] = None,
    ) -> Tuple[StrategySpecV3, pd.DataFrame]:
        """
        Grid search over spec parameters (for exploration only, not optimization!).
        
        ⚠️  CRITICAL: This is for understanding parameter sensitivity, NOT for optimization.
        To avoid data snooping:
        - Use only on training set (60% of data)
        - Always validate on separate test set
        - Report out-of-sample metrics as final result
        
        Args:
            df: OHLCV DataFrame (TRAINING SET ONLY)
            score: Factor score array
            base_spec: Base StrategySpecV3
            param_ranges: Dict of parameter -> list of values to try
                         E.g., {"stop_loss_atr_mult": [1.5, 2.0, 2.5]}
            
        Returns:
            (best_spec, results_dataframe)
        """
        if param_ranges is None:
            param_ranges = {
                "stop_loss_atr_mult": [1.5, 2.0, 2.5],
                "take_profit_atr_mult": [2.0, 2.5, 3.0, 3.5],
                "target_vol": [0.08, 0.10, 0.12],
            }
        
        logger.info(f"\n🔍 Parameter Search (TRAINING SET ONLY)")
        logger.info(f"   Ranges: {param_ranges}")
        logger.warning("   ⚠️  REMEMBER: Always validate on separate test set!")
        
        results_list = []
        
        # Generate all combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            # Create spec with this combination
            spec_dict = asdict(base_spec)
            for param_name, param_val in zip(param_names, combination):
                spec_dict[param_name] = param_val
            
            if isinstance(spec_dict.get("vol_method"), str):
                spec_dict["vol_method"] = VolMethod(spec_dict["vol_method"])
            
            spec = StrategySpecV3(**spec_dict)
            
            # Backtest
            results = self.backtester.run_backtest(df, score, spec)
            
            # Store result
            result_row = asdict(base_spec)
            result_row.update({
                param_name: param_val
                for param_name, param_val in zip(param_names, combination)
            })
            result_row.update({
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown,
                "win_rate": results.win_rate,
                "total_return": results.total_return,
                "ic": results.information_coefficient,
            })
            results_list.append(result_row)
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)
        
        # Best spec
        best_row = results_df.iloc[0]
        best_spec_dict = asdict(base_spec)
        for param_name in param_names:
            best_spec_dict[param_name] = best_row[param_name]
        
        if isinstance(best_spec_dict.get("vol_method"), str):
            best_spec_dict["vol_method"] = VolMethod(best_spec_dict["vol_method"])
        
        best_spec = StrategySpecV3(**best_spec_dict)
        
        logger.info(f"\n🏆 Best Configuration (on TRAINING set):")
        logger.info(f"   {best_spec.strategy_name}")
        logger.info(f"   Sharpe: {best_row['sharpe_ratio']:.2f}")
        logger.info(f"   Max DD: {best_row['max_drawdown']:.2%}")
        logger.info(f"   Parameters: {dict(zip(param_names, combination))}")
        logger.warning(f"\n   ⚠️  THIS IS TRAINING SET METRIC! Test on holdout data!")
        
        return best_spec, results_df
    
    def generate_report(
        self,
        spec: StrategySpecV3,
        results: BacktestResultsV3,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Generate professional backtest report.
        
        Args:
            spec: StrategySpecV3
            results: BacktestResultsV3
            output_dir: Directory to save report
            
        Returns:
            Dict with report data
        """
        report = {
            "strategy_name": spec.strategy_name,
            "description": spec.description,
            "specification": asdict(spec),
            "backtest_results": {
                "total_return": results.total_return,
                "annual_return": results.annual_return,
                "annual_vol": results.annual_vol,
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown,
                "calmar_ratio": results.calmar_ratio,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "information_coefficient": results.information_coefficient,
                "ic_std": results.ic_std,
                "total_trades": results.total_trades,
                "long_trades": results.long_trades,
                "short_trades": results.short_trades,
                "exit_distribution": {
                    "take_profit": results.tp_count,
                    "stop_loss": results.sl_count,
                    "time_stop": results.time_stop_count,
                },
            },
            "validation_checklist": {
                "sharpe_gt_1": results.sharpe_ratio > 1.0,
                "win_rate_gt_45": results.win_rate > 0.45,
                "mdd_lt_20": results.max_drawdown > -0.2,
                "ic_significant": results.ic_std > 0.02 if results.ic_std else False,
                "min_trades": results.total_trades >= 30,
            },
            "ready_for_deployment": all([
                results.sharpe_ratio > 1.0,
                results.win_rate > 0.45,
                results.max_drawdown > -0.2,
                results.total_trades >= 30,
            ]),
        }
        
        # Save report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_file = output_path / f"{spec.strategy_name.replace(' ', '_')}_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Report saved: {report_file}")
        
        return report
    
    def walkforward_validate(
        self,
        df: pd.DataFrame,
        score: np.ndarray,
        spec: StrategySpecV3,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
    ) -> Dict[str, BacktestResultsV3]:
        """
        Walk-forward validation: Train/Val/Test split.
        
        Returns scores to compare against data snooping.
        
        Args:
            df: Full OHLCV DataFrame
            score: Full factor score array
            spec: StrategySpecV3
            train_pct: Training set percentage
            val_pct: Validation set percentage
            
        Returns:
            Dict with "train", "val", "test" BacktestResultsV3
        """
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        logger.info(f"\n🔄 Walk-Forward Validation")
        logger.info(f"   Train: 0 → {train_end} ({train_pct:.0%})")
        logger.info(f"   Val:   {train_end} → {val_end} ({val_pct:.0%})")
        logger.info(f"   Test:  {val_end} → {n} ({(1-train_pct-val_pct):.0%})")
        
        # Train
        train_results = self.backtest_spec(
            df.iloc[:train_end].reset_index(drop=True),
            score[:train_end],
            spec,
        )
        
        # Val (no optimization, just evaluation)
        val_results = self.backtest_spec(
            df.iloc[train_end:val_end].reset_index(drop=True),
            score[train_end:val_end],
            spec,
        )
        
        # Test
        test_results = self.backtest_spec(
            df.iloc[val_end:].reset_index(drop=True),
            score[val_end:],
            spec,
        )
        
        logger.info(f"\n📊 Walk-Forward Results:")
        logger.info(f"   Train Sharpe: {train_results.sharpe_ratio:.2f}")
        logger.info(f"   Val Sharpe:   {val_results.sharpe_ratio:.2f}")
        logger.info(f"   Test Sharpe:  {test_results.sharpe_ratio:.2f}")
        
        # Check for data snooping
        sharpe_decay = train_results.sharpe_ratio - test_results.sharpe_ratio
        if sharpe_decay > 0.5:
            logger.warning(f"⚠️  High Sharpe decay ({sharpe_decay:.2f}) - possible snooping!")
        else:
            logger.info(f"✅ Sharpe decay acceptable ({sharpe_decay:.2f})")
        
        return {
            "train": train_results,
            "val": val_results,
            "test": test_results,
        }
