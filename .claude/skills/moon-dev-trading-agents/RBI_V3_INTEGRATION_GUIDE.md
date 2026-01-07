# RBI v3 Integration Guide

**Status**: Implementation roadmap for RBI agent → v3 strategy auto-generation  
**Date**: January 7, 2026  
**Target**: Fully automated RBI → StrategySpecV3 → Backtest → Report pipeline

---

## Overview

Currently, RBI agent generates scattered backtest results. v3 integration means:

**RBI Agent** describes a factor (text/video/PDF)  
 ↓  
**RBIV3Generator** converts to StrategySpecV3  
 ↓  
**VectorizedEngineV3** backtest (vol-targeted, ATR exits, IC tracking)  
 ↓  
**Professional Report** (Sharpe, MDD, IC, walk-forward validation)  
 ↓  
**Auto-Deploy** if passes Jim Simons criteria

---

## File Structure (New v3 Integration)

```
src/data_pipeline/ai_agents/
├── __init__.py
├── moondev_integration.py          (existing)
├── rbi_v3_generator.py             ✅ NEW - Main v3 generator
├── factor_extractors.py            ✅ NEW - 10+ common factors
└── ...

src/data_pipeline/backtest/
├── strategy_spec_v3.py             ✅ (from v3 framework)
├── vectorized_engine_v3.py         ✅ (from v3 framework)
├── volatility_estimators.py        ✅ (from v3 framework)
├── exit_engine_v3.py               ✅ (from v3 framework)
├── position_sizer_v3.py            ✅ (from v3 framework)
└── ...

examples/
├── rbi_v3_complete_pipeline.py     ✅ NEW - Full demo
└── ...

.claude/skills/moon-dev-trading-agents/
├── V3_SPECIFICATION.md             ✅ (v3 framework spec)
├── RBI_V3_INTEGRATION_GUIDE.md     ✅ NEW - This file
└── ...
```

---

## Step-by-Step Implementation

### Phase 1: Core RBI v3 (DONE ✅)

**Components already created:**
1. ✅ `RBIV3Generator` class - Main orchestrator
2. ✅ `FactorExtractors` - 10+ pre-built factors
3. ✅ `rbi_v3_complete_pipeline.py` - Full demo

**What these do:**
- Convert factor descriptions → StrategySpecV3
- Generate factor scores (-1 to 1)
- Run vectorized backtests
- Parameter search (on training set only)
- Walk-forward validation (60/20/20)
- Generate professional reports

---

### Phase 2: Integrate with Existing RBI Agent (TO DO)

**Current RBI workflow:**
```
Youtube/PDF/Text
  ↓
RBI Agent (analyzes)
  ↓
"Buy when RSI < 30" → Python code (unstructured)
  ↓
Backtest code (scattered parameters)
  ↓
Metrics (Sharpe, return, MDD)
```

**New RBI v3 workflow:**
```
Youtube/PDF/Text
  ↓
RBI Agent (analyzes)
  ↓
"Buy when RSI < 30" → Structured JSON
  ↓
RBIV3Generator.create_spec_from_description()
  ↓
StrategySpecV3 (type-safe contract)
  ↓
RBIV3Generator.backtest_spec()
  ↓
VectorizedEngineV3 (professional metrics)
  ↓
Walk-forward validation + IC
  ↓
JSON Report + Deployment readiness
```

**Implementation steps:**

#### 2.1 Modify RBI Agent Output Format

**Currently**, RBI returns something like:
```python
{
    "factor_name": "rsi_oversold",
    "description": "Buy when RSI < 30",
    "code": "...python code...",
    "parameters": {"period": 14, "threshold": 30},
    "backtest_sharpe": 1.45,
}
```

**New format** should be structured for v3:
```python
{
    "factor_type": "rsi_mean_reversion",  # Must match FactorExtractors
    "description": "RSI-based mean reversion...",
    "factor_parameters": {
        "period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
    },
    "suggested_spec": {
        "strategy_name": "RSI Mean Reversion",
        "stop_loss_atr_mult": 2.0,
        "take_profit_atr_mult": 2.5,
        "time_stop_bars": 15,
        "target_vol": 0.12,
    }
}
```

#### 2.2 Create RBI Output Validator

```python
# src/data_pipeline/ai_agents/rbi_output_validator.py

from typing import Dict
from .factor_extractors import FactorExtractors
from .rbi_v3_generator import RBIV3Generator

class RBIOutputValidator:
    """Validate and convert RBI output to v3 spec."""
    
    VALID_FACTOR_TYPES = [
        "rsi_mean_reversion",
        "ma_crossover",
        "momentum",
        "bollinger_breakout",
        "atr_expansion",
        # ... etc
    ]
    
    @staticmethod
    def validate_rbi_output(rbi_output: Dict) -> Dict:
        """Validate RBI output format.
        
        Args:
            rbi_output: Dict from RBI agent
            
        Returns:
            Validated output ready for v3
            
        Raises:
            ValueError if validation fails
        """
        # Check required fields
        required = ["factor_type", "description", "factor_parameters"]
        missing = [k for k in required if k not in rbi_output]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        
        # Check factor type is valid
        factor_type = rbi_output["factor_type"]
        if factor_type not in RBIOutputValidator.VALID_FACTOR_TYPES:
            raise ValueError(
                f"Unknown factor type: {factor_type}. "
                f"Valid: {RBIOutputValidator.VALID_FACTOR_TYPES}"
            )
        
        # Check factor parameters are reasonable
        params = rbi_output["factor_parameters"]
        if not isinstance(params, dict):
            raise ValueError("factor_parameters must be dict")
        
        return rbi_output
    
    @staticmethod
    def rbi_to_spec(rbi_output: Dict) -> Dict:
        """Convert validated RBI output to StrategySpecV3.
        
        Args:
            rbi_output: Validated RBI output
            
        Returns:
            Dict with StrategySpecV3 and backtest results
        """
        # Validate first
        rbi_output = RBIOutputValidator.validate_rbi_output(rbi_output)
        
        # Create spec
        generator = RBIV3Generator()
        spec = generator.create_spec_from_description(
            factor_description=rbi_output["description"],
            factor_type="mean_reversion" if "reversion" in rbi_output["factor_type"] else "momentum",
            strategy_name=rbi_output.get("strategy_name", f"{rbi_output['factor_type']} v1"),
            custom_params=rbi_output.get("suggested_spec"),
        )
        
        return {
            "status": "validated",
            "rbi_output": rbi_output,
            "spec": spec,
        }
```

#### 2.3 Create RBI Pipeline Endpoint

```python
# src/data_pipeline/ai_agents/rbi_v3_pipeline.py

from typing import Dict, Optional
from .rbi_output_validator import RBIOutputValidator
from .rbi_v3_generator import RBIV3Generator
from .factor_extractors import create_factor_from_description
import pandas as pd

class RBIV3Pipeline:
    """End-to-end RBI → v3 Backtest pipeline."""
    
    def __init__(self):
        self.generator = RBIV3Generator()
    
    def process_rbi_factor(
        self,
        rbi_output: Dict,
        df: pd.DataFrame,
    ) -> Dict:
        """Process RBI output and return backtest results.
        
        Args:
            rbi_output: Output from RBI agent
            df: OHLCV data
            
        Returns:
            Dict with backtest report ready for deployment
        """
        # Step 1: Validate RBI output
        validated = RBIOutputValidator.rbi_to_spec(rbi_output)
        spec = validated["spec"]
        
        # Step 2: Generate factor scores
        score = create_factor_from_description(
            factor_type=rbi_output["factor_type"],
            df=df,
            **rbi_output["factor_parameters"],
        )
        
        # Step 3: Backtest
        results = self.generator.backtest_spec(df, score, spec)
        
        # Step 4: Walk-forward validation
        wf_results = self.generator.walkforward_validate(df, score, spec)
        
        # Step 5: Generate report
        report = self.generator.generate_report(spec, results)
        
        return {
            "rbi_factor": rbi_output,
            "spec": spec,
            "backtest_results": {
                "train": self._results_to_dict(wf_results["train"]),
                "val": self._results_to_dict(wf_results["val"]),
                "test": self._results_to_dict(wf_results["test"]),
            },
            "report": report,
            "deployment_ready": report["ready_for_deployment"],
        }
    
    @staticmethod
    def _results_to_dict(results):
        """Convert BacktestResultsV3 to dict."""
        return {
            "sharpe_ratio": results.sharpe_ratio,
            "total_return": results.total_return,
            "max_drawdown": results.max_drawdown,
            "win_rate": results.win_rate,
            "ic": results.information_coefficient,
        }
```

---

### Phase 3: Update RBI Agent Main Loop (TO DO)

**File**: `src/agents/rbi_agent_pp_multi.py` (or equivalent)

**Current flow**:
```python
def rbi_main():
    # Parse factor description
    factor = parse_factor_description(user_input)
    
    # Write code
    code = generate_backtest_code(factor)
    
    # Run backtest
    results = backtest(code, data)
    
    # Return results
    return results
```

**New flow**:
```python
def rbi_main():
    # Parse factor description
    factor = parse_factor_description(user_input)
    
    # Convert to structured RBI output
    rbi_output = {
        "factor_type": factor.type,  # rsi_mean_reversion, momentum, etc.
        "description": factor.description,
        "factor_parameters": factor.parameters,
    }
    
    # USE v3 PIPELINE
    pipeline = RBIV3Pipeline()
    v3_results = pipeline.process_rbi_factor(rbi_output, data)
    
    # Return v3 results (structured, validated, deployment-ready)
    return v3_results
```

---

## How to Use (For You)

### Quick Test

```bash
# Clone and checkout
git clone https://github.com/bigsuperFishfish/moon-dev-ai-agents
cd moon-dev-ai-agents
git checkout jan2025-data-download

# Run complete demo
python examples/rbi_v3_complete_pipeline.py
```

This will:
1. Generate synthetic data
2. Test 3 complete pipelines
3. Show parameter search + walk-forward validation
4. Compare 4 different factors
5. Generate reports

### Use in Your RBI

```python
from src.data_pipeline.ai_agents.rbi_v3_pipeline import RBIV3Pipeline
from src.data_pipeline.ai_agents.factor_extractors import create_factor_from_description

# Your RBI output
rbi_output = {
    "factor_type": "rsi_mean_reversion",
    "description": "Buy RSI < 30",
    "factor_parameters": {"period": 14, "oversold_threshold": 30},
}

# Load data
df = pd.read_csv("btc_1h.csv")

# Process through v3 pipeline
pipeline = RBIV3Pipeline()
results = pipeline.process_rbi_factor(rbi_output, df)

# Check if ready for deployment
if results["deployment_ready"]:
    print("✅ Strategy passes all criteria!")
    print(f"Sharpe: {results['backtest_results']['test']['sharpe_ratio']:.2f}")
else:
    print("❌ Needs refinement")
```

---

## Validation Criteria (Jim Simons Standard)

Strategy passes if:
- ✅ Sharpe > 1.0 (consistent return per unit risk)
- ✅ Win rate > 45% (more wins than losses)
- ✅ Max DD > -20% (drawdown acceptable)
- ✅ 30+ trades (sufficient sample size)
- ✅ Test Sharpe decay < 50% from train (not overfit)
- ✅ IC > 0.02 (factor has predictive power)

---

## Key Differences from Current RBI

| Aspect | Old RBI | RBI v3 |
|--------|---------|--------|
| **Output Format** | Scattered parameters | StrategySpecV3 (structured) |
| **Position Sizing** | Fixed %  | Vol-targeted (dynamic) |
| **Exits** | Manual/fixed | ATR-based (objective) |
| **Validation** | Single backtest | Walk-forward 60/20/20 |
| **Metrics** | Sharpe, return, MDD | + IC, Calmar, profit factor |
| **Data Snooping** | Not checked | Train/val/test splits |
| **Deployment** | Manual review | Auto-check readiness |

---

## Common Questions

**Q: What if my factor isn't in FactorExtractors?**  
A: Add it to `factor_extractors.py`. Must return score in [-1, 1].

**Q: Can I combine multiple factors?**  
A: No. v3 enforces single-factor units. Use portfolio aggregation instead.

**Q: How do I avoid data snooping?**  
A: Use walk-forward validation. v3 does 60/20/20 split automatically.

**Q: My test Sharpe is much lower than train. What now?**  
A: Possible overfitting. Try:
  1. Fewer parameters in spec
  2. Larger regularization (vol_floor, delta_pos_max)
  3. Different factor altogether

---

## Next Steps

1. ✅ **Phase 1 complete**: Core v3 infrastructure ready
2. 🚧 **Phase 2 (your task)**: Integrate with existing RBI agent
   - Add RBIOutputValidator
   - Add RBIV3Pipeline
   - Modify RBI main loop to use v3
3. 🚧 **Phase 3**: Live deployment bridge (execution).

---

## Files to Modify (For RBI Integration)

| File | Action | Priority |
|------|--------|----------|
| `src/agents/rbi_agent_pp_multi.py` | Use RBIV3Pipeline | High |
| `src/data_pipeline/ai_agents/rbi_output_validator.py` | Create | High |
| `src/data_pipeline/ai_agents/rbi_v3_pipeline.py` | Create | High |
| `src/data_pipeline/ai_agents/factor_extractors.py` | Extend w/ new factors | Medium |
| `tests/test_rbi_v3.py` | Add tests | Medium |

---

## Contact

For questions on RBI v3 integration, refer to this guide or open issues in repo.
