# RBI v3 Deployment Checklist

**Goal**: Modify existing RBI agent to output StrategySpecV3 + auto-backtest with v3  
**Timeline**: Can be done incrementally  
**Difficulty**: Low-to-medium (mostly copy-paste + wiring)

---

## Pre-Flight Checks

- [x] v3 framework built (vectorized engine, specs, validators)
- [x] RBIV3Generator ready (creates specs from descriptions)
- [x] FactorExtractors ready (10+ common factors)
- [x] Example pipeline complete (shows all features)
- [ ] Your RBI agent identified (know the file to modify)
- [ ] Test data ready (crypto OHLCV, 24/7)

---

## 3-Step Implementation

### STEP 1: Create v3 Output Validator (30 mins)

**File to create**: `src/data_pipeline/ai_agents/rbi_output_validator.py`

**What it does**: Takes RBI output → validates → converts to StrategySpecV3

**Code template** (copy-paste ready):

```python
# src/data_pipeline/ai_agents/rbi_output_validator.py

from typing import Dict
from dataclasses import asdict
from .rbi_v3_generator import RBIV3Generator
import logging

logger = logging.getLogger(__name__)

class RBIOutputValidator:
    """Validate RBI factor output and convert to v3 spec."""
    
    # Maps RBI factor descriptions to types
    FACTOR_TYPE_MAPPING = {
        "rsi": "rsi_mean_reversion",
        "oversold": "rsi_mean_reversion",
        "moving average": "ma_crossover",
        "crossover": "ma_crossover",
        "momentum": "momentum",
        "bollinger": "bollinger_breakout",
        "volatility": "atr_expansion",
        "zscore": "zscore_mean_reversion",
        "breakout": "high_low_breakout",
        "volume": "volume_weighted_momentum",
    }
    
    VALID_FACTOR_TYPES = [
        "rsi_mean_reversion",
        "ma_crossover",
        "momentum",
        "bollinger_breakout",
        "atr_expansion",
        "zscore_mean_reversion",
        "high_low_breakout",
        "volume_weighted_momentum",
    ]
    
    @staticmethod
    def auto_detect_factor_type(description: str) -> str:
        """Detect factor type from description.
        
        Args:
            description: Factor description from RBI
            
        Returns:
            Factor type (e.g., "rsi_mean_reversion")
        """
        desc_lower = description.lower()
        
        for keyword, factor_type in RBIOutputValidator.FACTOR_TYPE_MAPPING.items():
            if keyword in desc_lower:
                logger.info(f"✅ Auto-detected factor type: {factor_type}")
                return factor_type
        
        logger.warning(f"Could not auto-detect factor type, defaulting to mean_reversion")
        return "rsi_mean_reversion"
    
    @staticmethod
    def validate_rbi_output(rbi_output: Dict) -> bool:
        """Validate RBI output has required fields.
        
        Args:
            rbi_output: Dict from RBI agent
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_fields = [
            "factor_description",  # Human-readable description
            "factor_parameters",    # Dict of parameters
        ]
        
        missing = [f for f in required_fields if f not in rbi_output]
        if missing:
            raise ValueError(f"RBI output missing fields: {missing}")
        
        if not isinstance(rbi_output["factor_parameters"], dict):
            raise ValueError("factor_parameters must be a dict")
        
        logger.info("✅ RBI output validation passed")
        return True
    
    @staticmethod
    def rbi_to_spec_v3(rbi_output: Dict) -> Dict:
        """Convert RBI output to StrategySpecV3.
        
        Args:
            rbi_output: Dict from RBI agent with keys:
                - factor_description: str
                - factor_parameters: dict
                - [optional] factor_type: str (if not provided, will auto-detect)
                - [optional] strategy_name: str
                
        Returns:
            Dict with:
                - spec: StrategySpecV3 instance
                - factor_type: str
                - rbi_output: original input
        """
        # Validate
        RBIOutputValidator.validate_rbi_output(rbi_output)
        
        # Auto-detect factor type if not provided
        if "factor_type" not in rbi_output:
            rbi_output["factor_type"] = RBIOutputValidator.auto_detect_factor_type(
                rbi_output["factor_description"]
            )
        else:
            # Validate if provided
            if rbi_output["factor_type"] not in RBIOutputValidator.VALID_FACTOR_TYPES:
                raise ValueError(
                    f"Invalid factor type: {rbi_output['factor_type']}. "
                    f"Valid: {RBIOutputValidator.VALID_FACTOR_TYPES}"
                )
        
        # Create spec
        generator = RBIV3Generator()
        spec = generator.create_spec_from_description(
            factor_description=rbi_output["factor_description"],
            factor_type="mean_reversion" if "reversion" in rbi_output["factor_type"] else "momentum",
            strategy_name=rbi_output.get("strategy_name", f"{rbi_output['factor_type']} (from RBI)"),
        )
        
        logger.info(f"✅ Created StrategySpecV3: {spec.strategy_name}")
        
        return {
            "spec": spec,
            "factor_type": rbi_output["factor_type"],
            "factor_parameters": rbi_output["factor_parameters"],
            "rbi_output": rbi_output,
        }
```

**Checklist for Step 1:**
- [ ] File created at `src/data_pipeline/ai_agents/rbi_output_validator.py`
- [ ] Test import works: `from src.data_pipeline.ai_agents.rbi_output_validator import RBIOutputValidator`
- [ ] Ran basic test:
  ```python
  rbi_output = {
      "factor_description": "Buy when RSI < 30",
      "factor_parameters": {"period": 14},
  }
  result = RBIOutputValidator.rbi_to_spec_v3(rbi_output)
  print(result["spec"].strategy_name)  # Should work
  ```

---

### STEP 2: Create RBI v3 Pipeline (45 mins)

**File to create**: `src/data_pipeline/ai_agents/rbi_v3_pipeline.py`

**What it does**: End-to-end pipeline (RBI output → generate score → backtest → report)

**Code template**:

```python
# src/data_pipeline/ai_agents/rbi_v3_pipeline.py

from typing import Dict
import pandas as pd
import logging

from .rbi_output_validator import RBIOutputValidator
from .rbi_v3_generator import RBIV3Generator
from .factor_extractors import create_factor_from_description

logger = logging.getLogger(__name__)

class RBIV3Pipeline:
    """End-to-end RBI → v3 Backtest pipeline."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.generator = RBIV3Generator()
        self.validator = RBIOutputValidator
    
    def process_factor(
        self,
        rbi_output: Dict,
        df: pd.DataFrame,
        enable_walkforward: bool = True,
        output_dir: str = "./backtest_reports",
    ) -> Dict:
        """
        Process RBI factor through complete v3 pipeline.
        
        Args:
            rbi_output: Dict from RBI agent with:
                - factor_description: str
                - factor_parameters: dict
                - [optional] factor_type: str
                - [optional] strategy_name: str
            df: OHLCV DataFrame (required columns: open, high, low, close, volume)
            enable_walkforward: Run 60/20/20 walk-forward validation
            output_dir: Where to save report
            
        Returns:
            Dict with:
                - status: "success" or "error"
                - spec: StrategySpecV3
                - backtest_results: Dict with Sharpe, return, MDD, etc.
                - walkforward: Dict with train/val/test results
                - report: Professional report
                - deployment_ready: bool
                - error: str (if status is "error")
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("🚀 RBI v3 PIPELINE START")
            logger.info("="*80)
            
            # Step 1: Validate and convert
            logger.info("\n🔍 Step 1: Validate RBI output and create spec...")
            converted = self.validator.rbi_to_spec_v3(rbi_output)
            spec = converted["spec"]
            factor_type = converted["factor_type"]
            factor_params = converted["factor_parameters"]
            
            # Step 2: Generate factor scores
            logger.info("\n📊 Step 2: Generate factor scores...")
            score = create_factor_from_description(
                factor_type=factor_type,
                df=df,
                **factor_params,
            )
            logger.info(f"✅ Score range: [{score.min():.2f}, {score.max():.2f}]")
            logger.info(f"✅ Score distribution: {(score != 0).sum()} non-zero signals")
            
            # Step 3: Backtest
            logger.info("\n🔄 Step 3: Run backtest...")
            results = self.generator.backtest_spec(df, score, spec)
            
            # Step 4: Walk-forward (if enabled)
            walkforward_results = None
            if enable_walkforward:
                logger.info("\n🔄 Step 4: Walk-forward validation (60/20/20)...")
                walkforward_results = self.generator.walkforward_validate(df, score, spec)
            
            # Step 5: Generate report
            logger.info("\n📌 Step 5: Generate report...")
            report = self.generator.generate_report(
                spec, results, output_dir=output_dir
            )
            
            # Build final output
            output = {
                "status": "success",
                "spec": spec,
                "backtest_results": {
                    "sharpe_ratio": results.sharpe_ratio,
                    "total_return": results.total_return,
                    "annual_return": results.annual_return,
                    "annual_vol": results.annual_vol,
                    "max_drawdown": results.max_drawdown,
                    "calmar_ratio": results.calmar_ratio,
                    "win_rate": results.win_rate,
                    "profit_factor": results.profit_factor,
                    "information_coefficient": results.information_coefficient,
                    "total_trades": results.total_trades,
                    "exit_distribution": {
                        "take_profit": results.tp_count,
                        "stop_loss": results.sl_count,
                        "time_stop": results.time_stop_count,
                    },
                },
                "report": report,
                "deployment_ready": report["ready_for_deployment"],
            }
            
            # Add walkforward if available
            if walkforward_results:
                output["walkforward"] = {
                    "train_sharpe": walkforward_results["train"].sharpe_ratio,
                    "val_sharpe": walkforward_results["val"].sharpe_ratio,
                    "test_sharpe": walkforward_results["test"].sharpe_ratio,
                    "sharpe_decay_train_to_test": (
                        walkforward_results["train"].sharpe_ratio - 
                        walkforward_results["test"].sharpe_ratio
                    ),
                }
            
            # Final summary
            logger.info("\n" + "="*80)
            logger.info("✅ RBI v3 PIPELINE COMPLETE")
            logger.info("="*80)
            logger.info(f"Deployment ready: {output['deployment_ready']}")
            logger.info(f"Sharpe: {output['backtest_results']['sharpe_ratio']:.2f}")
            logger.info(f"Max DD: {output['backtest_results']['max_drawdown']:.2%}")
            logger.info(f"Win rate: {output['backtest_results']['win_rate']:.1%}")
            
            return output
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }
```

**Checklist for Step 2:**
- [ ] File created at `src/data_pipeline/ai_agents/rbi_v3_pipeline.py`
- [ ] Test import works: `from src.data_pipeline.ai_agents.rbi_v3_pipeline import RBIV3Pipeline`
- [ ] Quick test:
  ```python
  import pandas as pd
  from src.data_pipeline.ai_agents.rbi_v3_pipeline import RBIV3Pipeline
  
  # Generate test data
  df = pd.DataFrame({
      "open": [100]*100, "high": [101]*100, "low": [99]*100,
      "close": [100]*100, "volume": [1e6]*100
  })
  
  # Test pipeline
  pipeline = RBIV3Pipeline()
  rbi_output = {
      "factor_description": "Buy when RSI < 30",
      "factor_parameters": {"period": 14},
  }
  result = pipeline.process_factor(rbi_output, df, enable_walkforward=False)
  print(result["status"])  # Should be "success"
  ```

---

### STEP 3: Integrate into Your RBI Agent (1-2 hours)

**File to modify**: `src/agents/rbi_agent_pp_multi.py` (or wherever your RBI lives)

**What to change**: After RBI generates a factor, instead of returning scattered results, use RBIV3Pipeline

**Code modification**:

```python
# In your RBI main function, AFTER the RBI analysis is done:

from src.data_pipeline.ai_agents.rbi_v3_pipeline import RBIV3Pipeline
import pandas as pd

def rbi_main(user_input: str, df: pd.DataFrame):
    """
    RBI Agent main with v3 integration.
    
    Args:
        user_input: User's factor description
        df: OHLCV data
        
    Returns:
        Dict with v3 backtest results
    """
    
    # ... existing RBI analysis code ...
    # This part analyzes the user input, identifies the factor, etc.
    
    # NEW: Instead of manual backtesting, use v3 pipeline
    rbi_output = {
        "factor_description": "Buy when RSI < 30, sell when RSI > 70",  # From RBI analysis
        "factor_type": "rsi_mean_reversion",  # Detected or specified by RBI
        "factor_parameters": {
            "period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
        "strategy_name": "RSI Mean Reversion (from RBI)",
    }
    
    # Use v3 pipeline
    pipeline = RBIV3Pipeline()
    results = pipeline.process_factor(rbi_output, df)
    
    # Return results
    if results["status"] == "success":
        print(f"✅ Strategy ready for deployment: {results['deployment_ready']}")
        print(f"Sharpe: {results['backtest_results']['sharpe_ratio']:.2f}")
        return results
    else:
        print(f"❌ Error: {results['error']}")
        return results
```

**Checklist for Step 3:**
- [ ] Import added at top of RBI file
- [ ] RBIV3Pipeline instantiated
- [ ] RBI output structured as dict (see format above)
- [ ] pipeline.process_factor() called
- [ ] Results returned to user
- [ ] Tested with sample factor

---

## Testing Checklist

After all 3 steps, test the integration:

```python
# test_rbi_v3_integration.py

import pandas as pd
import numpy as np
from src.data_pipeline.ai_agents.rbi_v3_pipeline import RBIV3Pipeline

def test_integration():
    """Test complete RBI v3 integration."""
    
    # Generate test data
    n = 2000
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1H"),
        "open": np.random.uniform(95, 105, n),
        "high": np.random.uniform(100, 110, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(95, 105, n),
        "volume": np.random.uniform(1e6, 5e6, n),
    })
    
    # Test cases
    test_factors = [
        {
            "factor_description": "RSI mean reversion strategy",
            "factor_parameters": {"period": 14, "oversold_threshold": 30},
        },
        {
            "factor_description": "Moving average crossover",
            "factor_type": "ma_crossover",
            "factor_parameters": {"fast_period": 10, "slow_period": 30},
        },
    ]
    
    pipeline = RBIV3Pipeline()
    
    for i, factor in enumerate(test_factors):
        print(f"\nTest {i+1}: {factor['factor_description']}")
        result = pipeline.process_factor(
            factor, df, 
            enable_walkforward=False,  # Skip for speed
        )
        
        # Checks
        assert result["status"] == "success", f"Status should be success, got {result['status']}"
        assert "sharpe_ratio" in result["backtest_results"], "Missing sharpe_ratio"
        assert "deployment_ready" in result, "Missing deployment_ready"
        
        print(f"  ✅ Sharpe: {result['backtest_results']['sharpe_ratio']:.2f}")
        print(f"  ✅ Ready: {result['deployment_ready']}")
    
    print("\n✅ All integration tests passed!")

if __name__ == "__main__":
    test_integration()
```

**Run test**:
```bash
python test_rbi_v3_integration.py
```

**Checklist:**
- [ ] Test runs without errors
- [ ] All factors pass validation
- [ ] Results include all required fields
- [ ] Sharpe ratios are reasonable (between -10 and 10)
- [ ] deployment_ready is bool

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: No module named 'src'" | Add `sys.path.insert(0, str(Path(__file__).parent.parent))` |
| "factor_parameters is not a dict" | Check RBI output format - must be `{"period": 14}` |
| "Unknown factor type" | Add to FactorExtractors or specify in rbi_output |
| "Empty score array" | Check OHLCV data has no NaN values |
| "Sharpe is -inf or nan" | Data might be all same price, or too short |

---

## Final Checklist

- [ ] Step 1 complete: RBIOutputValidator created
- [ ] Step 2 complete: RBIV3Pipeline created
- [ ] Step 3 complete: RBI agent modified to use v3
- [ ] Testing complete: Integration tests pass
- [ ] Documentation updated: In your RBI repo
- [ ] Ready to commit: `git push origin jan2025-data-download`

---

## Next: Live Execution

Once RBI v3 integration is working, next phase is **live execution bridge**:

- Entry: When score != 0, place order at market price
- Position: Sized by vol target (from StrategySpecV3)
- Exit: Triggered by ATR SL/TP or time stop
- Reporting: Log all trades to database

---

**Timeline**: With 3 parts (30 min + 45 min + 1.5 hour) = ~3 hours for full integration.  
**Questions**: Refer to v3 docs or see examples/rbi_v3_complete_pipeline.py
