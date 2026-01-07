# Moon Dev v3 Strategy Framework

**Status**: LIVE on `jan2025-data-download` branch  
**Date**: January 7, 2026  
**Author**: Moon Dev + Jim Simons Principles  

---

## Executive Summary

v3 formalizes the concept of a **complete strategy unit** = entry + exit + position sizing, enforced through:

1. **StrategySpecV3**: Type-safe contract for strategy specifications
2. **Volatility Estimators**: Garman-Klass (default) + Parkinson (fallback)
3. **Position Sizer v3**: Vol-targeted sizing with delta smoothing
4. **Exit Engine v3**: ATR-based SL/TP + time stop with intrabar fill logic
5. **Vectorized Engine v3**: Complete backtest integrating all components

---

## Core Principles

### 1. Single-Factor Unit
- **One strategy = one factor** (e.g., RSI oversold, MA crossover, Z-score)
- No mixing of multiple logics in one unit
- Long and short can be separate units if behavior differs

### 2. Objective Rules Only
- Entry: **score** (continuous -1 to 1) from market data
- Exit: **ATR multiples, time stop, price levels**—never account equity or psychology
- Sizing: **vol-targeted**—position scales with volatility

### 3. Backtestable Specification
- Every parameter in StrategySpecV3 must be…  
  - **Explicit**: No hidden assumptions
  - **Reproducible**: Same spec = same backtest
  - **Validatable**: Can compute IC, Sharpe, MDD

---

## StrategySpecV3 Contract

```python
@dataclass
class StrategySpecV3:
    # Entry: single factor
    entry_factor_name: str              # e.g., "rsi_oversold"
    entry_parameters: Dict[str, Any]    # e.g., {"rsi_period": 14, "threshold": 30}
    
    # Exit: objective market-based criteria
    stop_loss_atr_mult: float = 2.0     # SL = entry ± (2.0 * atr_at_entry)
    take_profit_atr_mult: float = 3.0   # TP = entry ± (3.0 * atr_at_entry)
    time_stop_bars: int = 20            # Max hold = 20 bars
    atr_period: int = 14                # ATR lookback
    
    # Position sizing: vol-targeted
    target_vol: float = 0.10            # Target 10% annualized volatility
    vol_method: VolMethod = GK          # Garman-Klass estimator
    vol_window: int = 20                # 20-bar rolling window
    vol_floor: float = 0.001            # Min vol = 0.1% annualized
    delta_pos_max: float = 0.1          # Max ±0.1 position change/bar
    
    # Metadata
    strategy_name: str
    description: str
```

### Why This Contract Matters

1. **No Ambiguity**: Every parameter has explicit meaning
2. **Type Safety**: Can't accidentally use account PnL in exit logic
3. **Composability**: Multiple units aggregate easily with known correlations
4. **Auditability**: Each backtest is reproducible from spec

---

## Volatility Estimation (Garman-Klass)

### Formula

$$\sigma^2_{GK,t} = 0.5 \cdot (\ln(H/L))^2 - (2\ln2 - 1) \cdot (\ln(C/O))^2$$

### Advantages
- Uses OHLC efficiently (not just close-to-close)
- Captures both range and drift
- Parkinson fallback if GK goes negative
- Works well for crypto (24/7, no overnight gaps)

### Implementation

```python
from src.data_pipeline.backtest.volatility_estimators import estimate_vol

realized_vol = estimate_vol(
    df,
    method="gk",           # Garman-Klass
    window=20,             # 20-bar rolling
    bars_per_year=8760,    # 365*24 for crypto hourly
    vol_floor=0.001,       # 0.1% minimum
)
```

---

## Position Sizing (Vol-Targeted)

### Formula

$$\text{pos}_{raw} = \frac{\text{target\_vol}}{\text{realized\_vol}} \times \text{sign}(\text{score})$$

Then: $\text{pos} = \text{clip}(\text{pos}_{raw}, -1, 1)$ with delta smoothing

### How It Works

1. **Low vol period** → larger position (to maintain consistent risk)
2. **High vol period** → smaller position (prevent excessive leverage)
3. **Delta smoothing** → prevent excessive rebalancing from vol noise

### Code

```python
from src.data_pipeline.backtest.position_sizer_v3 import PositionSizerV3

position, position_raw = PositionSizerV3.calculate_position(
    score=score_array,
    realized_vol=vol_array,
    target_vol=0.10,          # 10% target vol
    delta_pos_max=0.1,        # Max ±0.1 per bar
    apply_smoothing=True,
)
```

---

## Exit Rules (ATR-Based)

### Components

1. **Stop Loss**: `SL = entry ± (sl_atr_mult * atr_at_entry)`
2. **Take Profit**: `TP = entry ± (tp_atr_mult * atr_at_entry)`
3. **Time Stop**: Exit if not hit TP/SL within `time_stop_bars`

### Intrabar Fill Logic

Crypto bars may span significant moves. v3 uses **HIGH/LOW** to determine fills:

**Long position**:
- If `high >= tp_level` → fill at TP (profit takes priority)
- Else if `low <= sl_level` → fill at SL
- Else if `bars_held >= time_stop_bars` → fill at close

**Short position**:
- If `low <= tp_level` → fill at TP
- Else if `high >= sl_level` → fill at SL
- Else if `bars_held >= time_stop_bars` → fill at close

### Code

```python
from src.data_pipeline.backtest.exit_engine_v3 import ExitEngineV3

exit_signal = ExitEngineV3.check_exit(
    current_idx=i,
    entry_idx=entry_idx,
    entry_price=entry_price,
    entry_atr=atr_at_entry,
    direction=1,  # Long
    sl_atr_mult=2.0,
    tp_atr_mult=3.0,
    time_stop_bars=20,
    high=df['high'].iloc[i],
    low=df['low'].iloc[i],
    close=df['close'].iloc[i],
)
```

---

## Complete Backtest (VectorizedEngineV3)

### Usage

```python
from src.data_pipeline.backtest.vectorized_engine_v3 import VectorizedBacktesterV3
from src.data_pipeline.backtest.strategy_spec_v3 import RSI_MEAN_REVERSION_V3

backtester = VectorizedBacktesterV3(initial_cash=10000)

results = backtester.run_backtest(
    df=price_data,
    score=rsi_scores,  # -1 to 1 from your entry factor
    spec=RSI_MEAN_REVERSION_V3,
)

print(results)
# Output:
# Total Return:      45.32%
# Sharpe Ratio:      1.82
# Max Drawdown:      -12.5%
# Win Rate:          58.3%
# ...
```

### Output Metrics

| Metric | Meaning |
|--------|----------|
| `total_return` | Total P&L % |
| `annual_return` | Annualized return |
| `annual_vol` | Realized volatility |
| `sharpe_ratio` | Return / Vol |
| `max_drawdown` | Worst equity drop |
| `calmar_ratio` | Return / MDD |
| `win_rate` | % of profitable trades |
| `information_coefficient` | Factor correlation w/ forward returns |
| `ic_std` / `ic_tstat` | IC statistical significance |

---

## How to Use v3

### 1. Define Your Strategy

```python
from src.data_pipeline.backtest.strategy_spec_v3 import StrategySpecV3

my_strategy = StrategySpecV3(
    strategy_name="Custom RSI Breakout",
    entry_factor_name="rsi_breakout",
    entry_parameters={"rsi_period": 14, "breakout_level": 70},
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=2.5,
    time_stop_bars=30,
    target_vol=0.12,
)
```

### 2. Generate Score

```python
# Your entry factor logic (should output -1 to 1)
def generate_rsi_score(df, period=14, threshold=30):
    rsi = ta.RSI(df['close'], period)
    score = np.where(rsi < threshold, 1.0, np.where(rsi > 100 - threshold, -1.0, 0))
    return score

score = generate_rsi_score(df)
```

### 3. Backtest

```python
from src.data_pipeline.backtest.vectorized_engine_v3 import VectorizedBacktesterV3

backtester = VectorizedBacktesterV3()
results = backtester.run_backtest(df, score, my_strategy)

print(results)
df_with_trades = results.df  # Full backtest dataframe
```

### 4. Validate (Critical!)

```python
# Check Information Coefficient
if results.information_coefficient > 0.02 and results.ic_tstat > 1.96:
    print("Factor has statistically significant edge")
else:
    print("Factor may be data-snooped, refine or discard")

# Check win rate
if results.win_rate > 0.5 and results.sharpe_ratio > 1.0:
    print("Looks promising, do walk-forward test")
```

---

## Walk-Forward Validation (Must Do!)

v3 includes support for robust validation:

```python
# Split data: 60% train, 20% val, 20% test
train_end = int(len(df) * 0.6)
val_end = int(len(df) * 0.8)

# Train on train set
train_results = backtester.run_backtest(
    df.iloc[:train_end], 
    score[:train_end], 
    my_strategy
)

# Validate on val set (no optimization)
val_results = backtester.run_backtest(
    df.iloc[train_end:val_end], 
    score[train_end:val_end], 
    my_strategy
)

# Test on holdout set
test_results = backtester.run_backtest(
    df.iloc[val_end:], 
    score[val_end:], 
    my_strategy
)

print(f"Train Sharpe: {train_results.sharpe_ratio:.2f}")
print(f"Val Sharpe:   {val_results.sharpe_ratio:.2f}")
print(f"Test Sharpe:  {test_results.sharpe_ratio:.2f}")
```

If Test Sharpe is significantly lower than Train Sharpe → **data snooping**

---

## Key Files

| File | Purpose |
|------|----------|
| `volatility_estimators.py` | GK + Parkinson vol |
| `strategy_spec_v3.py` | StrategySpecV3 contract |
| `position_sizer_v3.py` | Vol-targeted sizing |
| `exit_engine_v3.py` | ATR SL/TP + exits |
| `vectorized_engine_v3.py` | Complete backtest engine |

---

## Common Pitfalls

### ❌ Don't
- Use account equity / PnL in exit logic
- Mix multiple factors in one unit
- Optimize parameters on test set
- Trust backtest without walk-forward validation
- Use close-only vol estimation (use GK instead)

### ✅ Do
- Keep entry **single-factor**
- Use ATR exits (not fixed %)
- Set position sizing via target_vol
- Run 60/20/20 or walk-forward tests
- Track Information Coefficient

---

## FAQ

**Q: Can I use multiple factors in one strategy?**  
A: No. v3 enforces single-factor units for clarity. Combine units via aggregation/portfolio optimization.

**Q: What if my data has gaps or missing bars?**  
A: Crypto 24/7, but if you add other assets, resample to consistent interval first.

**Q: How do I choose ATR multipliers?**  
A: Start with (2.0, 3.0), walk-forward test nearby values (1.5-3.0 range).

**Q: Can I turn off vol targeting?**  
A: Not recommended. Vol targeting prevents leverage creep in low-vol periods.

**Q: How do I detect data snooping?**  
A: Compare test set Sharpe to train/val. If test is much lower, your strategy overfit.

---

## Contact

For questions on v3 framework: Refer to `.claude/skills/moon-dev-trading-agents/` folder.  
For issues: Submit in the moon-dev-ai-agents repo.
