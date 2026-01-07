"""Moon Dev AI Agent Integration.

Bridge between Jan2025 data download pipeline and Moon Dev's powerful agents:
1. RBI Agent: Rapid factor screening from ideas (YouTube, PDFs, text)
2. Swarm Agent: Multi-model consensus voting
3. Strategy Agent: Deploy validated strategies

Workflow:
  Data Download → Validators → Factor Pipeline → RBI Screening → 
  Backtest Results → Strategy Selection → Moon Dev Deployment
"""

import os
import json
import sys
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MoonDevBridge:
    """Bridge to Moon Dev agents.
    
    This class:
    1. Exports clean data to Moon Dev format
    2. Triggers RBI agent for rapid screening
    3. Feeds backtest results to swarm agent
    4. Prepares strategy code for deployment
    """
    
    def __init__(self, moondev_root: str = "./"):
        """Initialize bridge to Moon Dev.
        
        Args:
            moondev_root: Root path of moon-dev-ai-agents repo
        """
        self.moondev_root = Path(moondev_root)
        self.rbi_agent_path = self.moondev_root / "src" / "agents" / "rbi_agent_pp_multi.py"
        self.data_export_dir = self.moondev_root / "src" / "data" / "jan2025_data_export"
        
        # Create export directory
        self.data_export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔗 Moon Dev bridge initialized at {moondev_root}")
    
    def export_data_for_rbi(
        self,
        data_dict: Dict,
        output_format: str = "csv",
    ) -> Dict[str, str]:
        """Export clean data in RBI-friendly format.
        
        Args:
            data_dict: Dict of (symbol, timeframe) -> DataFrame
            output_format: "csv" or "parquet"
            
        Returns:
            Dict of exported files
        """
        exported = {}
        
        for (symbol, timeframe), df in data_dict.items():
            filename = f"{symbol.replace('/', '-')}_{timeframe}_rbi.{output_format}"
            filepath = self.data_export_dir / filename
            
            # Save data
            if output_format == "csv":
                df.to_csv(filepath)
            else:
                df.to_parquet(filepath, compression="snappy")
            
            exported[f"{symbol}_{timeframe}"] = str(filepath)
            logger.info(f"💾 Exported {symbol} {timeframe} for RBI screening")
        
        return exported
    
    def generate_rbi_prompt(
        self,
        factor_description: str,
        symbols: List[str],
        timeframes: List[str],
        data_paths: Dict[str, str],
    ) -> str:
        """Generate a prompt for RBI agent screening.
        
        Args:
            factor_description: Description of factor to test
            symbols: List of symbols to test
            timeframes: List of timeframes
            data_paths: Dict of data file paths
            
        Returns:
            Prompt string for RBI agent
        """
        prompt = f"""
🚀 RBI FACTOR SCREENING REQUEST

**Factor Description:**
{factor_description}

**Assets to Screen:**
Symbols: {', '.join(symbols)}
Timeframes: {', '.join(timeframes)}
Date Range: 2020-01-01 to present (institutional era)

**Data Location:**
{json.dumps(data_paths, indent=2)}

**Expected Output:**
1. Backtest code for each factor variation
2. Performance metrics (Sharpe, Win Rate, Max DD)
3. Best-performing parameters
4. Risk assessment

**Jim Simons Filter:**
- Only report Sharpe > 1.0
- Win rate > 45%
- Max drawdown < 20%
- Out-of-sample confirmed

**Next Steps:**
After screening:
1. Mathematical formulation of winning factors
2. Statistical significance testing (bootstrap)
3. Cross-validation (60/20/20 split)
4. Regime analysis (when does it work?)
5. Live deployment via strategy_agent.py
        """
        
        return prompt.strip()
    
    def prepare_strategy_for_deployment(
        self,
        backtest_results: Dict,
        strategy_code: str,
        strategy_name: str,
    ) -> Dict:
        """Prepare validated strategy for Moon Dev deployment.
        
        Args:
            backtest_results: Backtest statistics
            strategy_code: Python code for strategy
            strategy_name: Name of strategy
            
        Returns:
            Dict with deployment info
        """
        deployment_info = {
            "strategy_name": strategy_name,
            "validation_date": pd.Timestamp.now().isoformat(),
            "backtest_stats": backtest_results,
            "code": strategy_code,
            "deployment_readiness": self._check_deployment_readiness(backtest_results),
            "risk_limits": self._calculate_risk_limits(backtest_results),
            "deployment_path": f"src/data/strategies/{strategy_name}.py",
        }
        
        # Save to JSON
        deployment_file = self.data_export_dir / f"{strategy_name}_deployment.json"
        with open(deployment_file, "w") as f:
            json.dump(deployment_info, f, indent=2, default=str)
        
        logger.info(f"✅ Strategy {strategy_name} prepared for deployment")
        
        return deployment_info
    
    @staticmethod
    def _check_deployment_readiness(stats: Dict) -> Dict[str, bool]:
        """Check if strategy is ready for live trading.
        
        Args:
            stats: Backtest statistics
            
        Returns:
            Dict of readiness checks
        """
        return {
            "sharpe_ok": stats.get("sharpe_ratio", 0) > 1.0,
            "drawdown_ok": stats.get("max_drawdown", 0) > -0.2,
            "win_rate_ok": stats.get("win_rate", 0) > 0.45,
            "trades_sufficient": stats.get("num_trades", 0) > 30,
            "volatility_ok": stats.get("volatility", 1) < 0.5,
        }
    
    @staticmethod
    def _calculate_risk_limits(stats: Dict) -> Dict:
        """Calculate risk limits based on backtest.
        
        Args:
            stats: Backtest statistics
            
        Returns:
            Dict with risk limits for deployment
        """
        max_drawdown = abs(stats.get("max_drawdown", -0.1))
        
        return {
            "max_position_usd": 1000,  # $1k per position
            "max_loss_daily": 500,     # $500 max loss/day
            "max_leverage": 1.0,       # Start unlevered
            "stop_loss_pct": min(max_drawdown * 0.5, 0.1),  # Half backtest drawdown
            "take_profit_pct": 0.05,   # 5% TP
        }
    
    def run_rbi_screening(
        self,
        factor_description: str,
        symbols: List[str],
        quick_test: bool = True,
    ) -> Dict:
        """Trigger RBI agent for rapid factor screening.
        
        Args:
            factor_description: Description of factor
            symbols: Symbols to test
            quick_test: If True, use 6-month data; if False, use all data
            
        Returns:
            Dict with RBI results (when ready)
        """
        logger.info(f"\n🤖 Triggering RBI Agent...")
        logger.info(f"Factor: {factor_description}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Quick test: {quick_test}")
        
        prompt = self.generate_rbi_prompt(
            factor_description=factor_description,
            symbols=symbols,
            timeframes=["1h", "4h"],
            data_paths={},
        )
        
        logger.info(f"\n{prompt}")
        
        # In production, this would call the actual RBI agent via:
        # from src.agents.rbi_agent_pp_multi import main as rbi_main
        # return rbi_main(prompt)
        
        logger.info("\n⚡️  RBI screening initiated (manual trigger needed)")
        logger.info("\nTo run RBI screening:")
        logger.info(f"  python src/agents/rbi_agent_pp_multi.py")
        logger.info(f"  Input the above prompt when asked.")
        
        return {"status": "awaiting_rbi_execution", "prompt": prompt}
    
    def integrate_rbi_results(
        self,
        rbi_output_dir: str,
    ) -> Dict:
        """Integrate RBI agent results back into pipeline.
        
        Args:
            rbi_output_dir: Directory where RBI agent saved results
            
        Returns:
            Dict with integrated results
        """
        rbi_dir = Path(rbi_output_dir)
        
        if not rbi_dir.exists():
            logger.error(f"❌ RBI output directory not found: {rbi_output_dir}")
            return {}
        
        # Load RBI results
        results = []
        for file in rbi_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                results.append(df)
                logger.info(f"✅ Loaded RBI result: {file.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {file}: {e}")
        
        if results:
            combined = pd.concat(results, ignore_index=True)
            # Filter by Jim Simons criteria
            filtered = combined[
                (combined['sharpe_ratio'] > 1.0) &
                (combined['win_rate'] > 0.45) &
                (combined['max_drawdown'] > -0.2)
            ].sort_values('sharpe_ratio', ascending=False)
            
            logger.info(f"\n🌟 RBI Results Summary")
            logger.info(f"Total variations tested: {len(combined)}")
            logger.info(f"Qualified (Sharpe > 1): {len(filtered)}")
            logger.info(f"\nTop 5 Strategies:")
            logger.info(filtered[['strategy', 'sharpe_ratio', 'win_rate', 'max_drawdown']].head())
            
            return {
                "all_results": combined,
                "qualified": filtered,
                "top_strategy": filtered.iloc[0] if len(filtered) > 0 else None,
            }
        
        return {}


if __name__ == "__main__":
    import pandas as pd
    
    # Initialize bridge
    bridge = MoonDevBridge(moondev_root="./")
    
    # Example: Trigger RBI screening for a simple momentum factor
    prompt = bridge.run_rbi_screening(
        factor_description="""RSI-based mean reversion:
        - Buy when RSI < 30 (oversold)
        - Sell when RSI > 70 (overbought)
        - Hold for 4 bars
        - Test RSI periods: 7, 14, 21
        """,
        symbols=["BTC", "ETH", "SOL"],
        quick_test=True,
    )
    
    logger.info(f"\n{prompt}")
