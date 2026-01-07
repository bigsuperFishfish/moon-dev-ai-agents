"""Complete RBI v3 Pipeline Example

Demonstrates the full workflow:
1. Describe a factor (RBI-style)
2. Create StrategySpecV3
3. Generate entry signals (factor scores)
4. Backtest with VectorizedEngineV3
5. Walk-forward validate
6. Generate professional report
7. Deploy if ready

Run this to understand the complete pipeline.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline.ai_agents.rbi_v3_generator import RBIV3Generator
from data_pipeline.ai_agents.factor_extractors import create_factor_from_description


def generate_sample_data(
    n_bars: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Args:
        n_bars: Number of bars to generate
        seed: Random seed
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    # Start price
    close = np.zeros(n_bars)
    close[0] = 100
    
    # Generate returns
    returns = np.random.normal(0.0001, 0.01, n_bars)
    for i in range(1, n_bars):
        close[i] = close[i-1] * (1 + returns[i])
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.003, n_bars))
    
    # Volume
    volume = np.random.uniform(1e6, 5e6, n_bars)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=n_bars, freq="1H"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    
    logger.info(f"✓ Generated {n_bars} bars of synthetic OHLCV data")
    return df


def pipeline_1_simple_mean_reversion():
    """Pipeline 1: Simple RSI mean reversion."""
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE 1: RSI Mean Reversion")
    logger.info("="*80)
    
    # Step 1: Generate data
    df = generate_sample_data(n_bars=2000)
    
    # Step 2: Create spec from description
    generator = RBIV3Generator()
    
    spec = generator.create_spec_from_description(
        factor_description="""RSI-based mean reversion strategy:
        - Buy when RSI < 30 (oversold)
        - Sell when RSI > 70 (overbought)
        - Hold for max 15 bars
        - Target 12% annualized volatility
        """,
        factor_type="mean_reversion",
        strategy_name="RSI Mean Reversion v1",
        custom_params={
            "entry_parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70},
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 2.5,
            "time_stop_bars": 15,
            "target_vol": 0.12,
        },
    )
    
    # Step 3: Generate entry signals
    score = create_factor_from_description(
        factor_type="rsi_mean_reversion",
        df=df,
        period=14,
        oversold_threshold=30,
        overbought_threshold=70,
    )
    
    logger.info(f"✓ Generated factor scores (range: [{score.min():.2f}, {score.max():.2f}])")
    
    # Step 4: Backtest
    results = generator.backtest_spec(df, score, spec)
    
    # Step 5: Walk-forward validation
    wf_results = generator.walkforward_validate(df, score, spec)
    
    # Step 6: Generate report
    report = generator.generate_report(
        spec, 
        results, 
        output_dir="./backtest_reports"
    )
    
    logger.info(f"\n📊 Final Assessment:")
    if report["ready_for_deployment"]:
        logger.info("✅ Strategy is ready for deployment!")
    else:
        logger.info("❌ Strategy needs refinement before deployment")
        missing = [k for k, v in report["validation_checklist"].items() if not v]
        logger.info(f"   Failed checks: {missing}")
    
    return spec, results, wf_results


def pipeline_2_momentum_with_parameter_search():
    """Pipeline 2: Momentum factor with parameter optimization."""
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE 2: Momentum with Parameter Search (TRAINING SET)")
    logger.info("="*80)
    
    # Step 1: Generate data
    df = generate_sample_data(n_bars=2000)
    
    # Split: 60% train, 20% val, 20% test
    train_end = int(len(df) * 0.6)
    val_end = int(len(df) * 0.8)
    
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"✓ Split data: Train {len(df_train)}, Val {len(df_val)}, Test {len(df_test)}")
    
    # Step 2: Create base spec
    generator = RBIV3Generator()
    
    base_spec = generator.create_spec_from_description(
        factor_description="""Momentum-based trend following:
        - Buy when price momentum > threshold
        - Sell when price momentum < -threshold
        - Target 15% annualized volatility
        """,
        factor_type="momentum",
        strategy_name="Momentum Trend Follower",
    )
    
    # Step 3: Generate signals on TRAINING set
    score_train = create_factor_from_description(
        factor_type="momentum",
        df=df_train,
        period=20,
        threshold=0.01,
    )
    
    # Step 4: Parameter search (only on training!)
    logger.info(f"\n🔍 Searching parameters on TRAINING set...")
    best_spec, search_results = generator.optimize_spec_parameters(
        df_train,
        score_train,
        base_spec,
        param_ranges={
            "stop_loss_atr_mult": [1.5, 2.0, 2.5],
            "take_profit_atr_mult": [2.5, 3.0, 3.5],
            "target_vol": [0.12, 0.15],
        },
    )
    
    logger.info(f"\n✓ Top 5 configurations on TRAINING set:")
    logger.info(search_results[["stop_loss_atr_mult", "take_profit_atr_mult", "target_vol", "sharpe_ratio"]].head())
    
    # Step 5: Validate on TEST set (holdout, no optimization!)
    logger.info(f"\n📊 Validating BEST configuration on TEST set...")
    
    score_test = create_factor_from_description(
        factor_type="momentum",
        df=df_test,
        period=20,
        threshold=0.01,
    )
    
    test_results = generator.backtest_spec(df_test, score_test, best_spec)
    
    logger.info(f"\n⚠️  CRITICAL COMPARISON:")
    logger.info(f"   Train Sharpe (best config): {search_results.iloc[0]['sharpe_ratio']:.2f}")
    logger.info(f"   Test Sharpe (same config):  {test_results.sharpe_ratio:.2f}")
    logger.info(f"   Decay: {search_results.iloc[0]['sharpe_ratio'] - test_results.sharpe_ratio:.2f}")
    
    if test_results.sharpe_ratio > search_results.iloc[0]["sharpe_ratio"] * 0.8:
        logger.info("   ✅ Reasonable decay, strategy likely not overfit")
    else:
        logger.warning("   ⚠️  High decay, possible overfitting!")
    
    return best_spec, test_results, search_results


def pipeline_3_multi_factor_comparison():
    """Pipeline 3: Compare multiple factors, select best."""
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE 3: Multi-Factor Comparison")
    logger.info("="*80)
    
    # Generate data
    df = generate_sample_data(n_bars=2000)
    
    generator = RBIV3Generator()
    
    # Test multiple factors
    factors = [
        ("rsi_mean_reversion", {"period": 14, "oversold_threshold": 30}),
        ("momentum", {"period": 20, "threshold": 0.01}),
        ("ma_crossover", {"fast_period": 10, "slow_period": 30}),
        ("zscore_mean_reversion", {"period": 20, "zscore_threshold": 2.0}),
    ]
    
    results_comparison = []
    
    for factor_type, factor_params in factors:
        logger.info(f"\n🔬 Testing {factor_type}...")
        
        # Generate score
        score = create_factor_from_description(
            factor_type=factor_type,
            df=df,
            **factor_params,
        )
        
        # Create spec
        spec = generator.create_spec_from_description(
            factor_description=f"Factor: {factor_type}",
            factor_type="mean_reversion" if "reversion" in factor_type else "momentum",
            strategy_name=f"{factor_type.title()} v1",
        )
        
        # Backtest
        results = generator.backtest_spec(df, score, spec)
        
        results_comparison.append({
            "factor": factor_type,
            "sharpe": results.sharpe_ratio,
            "return": results.total_return,
            "drawdown": results.max_drawdown,
            "win_rate": results.win_rate,
            "ic": results.information_coefficient,
        })
    
    # Compare
    comparison_df = pd.DataFrame(results_comparison).sort_values("sharpe", ascending=False)
    
    logger.info(f"\n🏆 Factor Comparison Results:")
    logger.info(comparison_df.to_string())
    
    best_factor = comparison_df.iloc[0]
    logger.info(f"\n✅ Best factor: {best_factor['factor']} (Sharpe: {best_factor['sharpe']:.2f})")
    
    return comparison_df


if __name__ == "__main__":
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       RBI v3 COMPLETE PIPELINE DEMO                          ║
║                                                                              ║
║  This demonstrates the full workflow for Moon Dev algo trading:             ║
║  1. Factor description → StrategySpecV3                                     ║
║  2. Factor score generation                                                 ║
║  3. Vectorized v3 backtesting                                               ║
║  4. Walk-forward validation                                                 ║
║  5. Parameter optimization (training only!)                                 ║
║  6. Statistical validation (IC, Sharpe decay)                               ║
║  7. Professional reporting                                                  ║
║                                                                              ║
║  Key Principles:                                                             ║
║  - Single-factor units only                                                 ║
║  - Vol-targeted position sizing                                             ║
║  - ATR-based exits (SL/TP/time)                                             ║
║  - Objective backtest rules (no account equity logic)                        ║
║  - Walk-forward validation (60/20/20 split)                                 ║
║  - Information Coefficient tracking                                          ║
║  - Test on HOLDOUT data only                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run pipelines
    try:
        # Pipeline 1: Simple strategy
        spec1, results1, wf1 = pipeline_1_simple_mean_reversion()
        
        # Pipeline 2: With parameter search and test validation
        spec2, test_results2, search2 = pipeline_2_momentum_with_parameter_search()
        
        # Pipeline 3: Compare multiple factors
        comparison = pipeline_3_multi_factor_comparison()
        
        logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ALL PIPELINES COMPLETE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
