# Jan 2025: Data Download Pipeline + Moon Dev AI Integration

## 🎯 Mission

**Implement Jim Simons quant trading methodology using Moon Dev AI agents as rapid screening tool.**

```
Raw Ideas (YouTube, PDFs)
    ↓
  [Data Download & Validation]
    ↓
  [Factor Implementation]
    ↓
  [Vectorized Backtesting]
    ↓
  [Moon Dev RBI Agent: Rapid Screening]
    ↓
  [Statistical Validation: Bootstrap, Cross-Validation]
    ↓
  [Regime Analysis: When does it work?]
    ↓
  [Moon Dev Strategy Agent: Live Deployment]
    ↓
  [Risk Agent: Circuit Breakers]
```

---

## 📊 Quick Start (5 Minutes)

### 1. Download & Test

```bash
cd moon-dev-ai-agents

# Install dependencies
pip install -r requirements.txt

# Run quick start (downloads BTC/ETH, runs 3 factor backtests)
python scripts/quick_start.py
```

**Expected Output:**
- 4 Parquet files (BTC/ETH × 1h/4h)
- 6 backtest results (2 pairs × 3 factors)
- Moon Dev RBI Agent prompt (copy-paste to RBI agent)

**Runtime:** ~5-10 minutes (first time includes download)  
**Cost:** $0 (free APIs)

---

## 🏗️ Architecture

### Data Pipeline Components

```
src/data_pipeline/
├── config.py                    # Configuration (symbols, timeframes, dates)
├── downloaders/
│   ├── ccxt_ohlcv.py           # OHLCV downloader (CCXT)
│   ├── funding_rate.py         # Binance funding rates
│   └── liquidations.py         # Coinglass liquidations
├── validators/
│   └── ohlcv_validator.py      # Data quality checks
├── factors/
│   └── momentum_factor.py      # Simple factors (RSI, BB, SMA)
├── backtest/
│   └── vectorized_engine.py    # Fast backtesting with costs
└── ai_agents/
    └── moondev_integration.py  # Bridge to Moon Dev agents
```

### Design Principles

1. **Data-First**: Bad data in → bad models out (Jim Simons)
2. **Realistic Costs**: 10bps slippage + 0.1% commission (not backtester fantasy)
3. **Fast Screening**: RBI agent tests 50 ideas in 6 hours ($1.50)
4. **Statistical Rigor**: Bootstrap, cross-validation, regime analysis
5. **Production-Ready**: Risk management from day one

---

## 🔧 Configuration

### File: `src/data_pipeline/config.py`

```python
# Assets to download
TOP_10_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", ...]

# Timeframes (Jim Simons uses 15m + 1h for signal/noise balance)
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Date range (2020/1/1 = start of institutional era)
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

# Quick testing (last 6 months only)
TEST_START_DATE = datetime(2024, 1, 1)
TEST_END_DATE = datetime.now()

# Storage: Parquet (10x compression vs CSV)
STORAGE_FORMAT = "parquet"

# Backtest settings: realistic costs
BACKTEST_CONFIG = {
    "initial_cash": 10000,
    "commission": 0.001,   # 0.1%
    "slippage_bps": 10,    # 10 basis points
}
```

**To customize:**
```python
# Edit these values to change:
VALID_SYMBOLS = ["BTC", "ETH", "SOL"]  # Your symbols
TIMEFRAMES = ["1h", "4h"]               # Your timeframes
TEST_START_DATE = datetime(2023, 1, 1) # Your test period
```

---

## 📥 Download OHLCV Data

### Full Dataset (All 10 Symbols, 2020-Present)

```python
from src.data_pipeline.downloaders.ccxt_ohlcv import download_main
from src.data_pipeline.config import VALID_SYMBOLS, TIMEFRAMES, START_DATE, END_DATE

files = download_main(
    symbols=VALID_SYMBOLS,  # BTC, ETH, SOL, BNB, ...
    timeframes=TIMEFRAMES,  # 15m, 1h, 4h, 1d
    start_date=START_DATE,
    end_date=END_DATE,
)
```

**Estimated Size:**
- 10 symbols × 4 timeframes × 5 years = 40 datasets
- ~500MB total (Parquet compression)
- **Time: 30-60 minutes** (rate-limited downloads)

### Quick Test (2 Symbols, Last 6 Months)

```python
from src.data_pipeline.downloaders.ccxt_ohlcv import download_main
from src.data_pipeline.config import TEST_START_DATE, TEST_END_DATE

files = download_main(
    symbols=["BTC", "ETH"],
    timeframes=["1h", "4h"],
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
)
# Time: 5 minutes, Size: 50MB
```

### Output Format

**File: `src/data/jan2025_download/ohlcv/BTC-USDT_1h.parquet`**

```
Columns: timestamp, open, high, low, close, volume, symbol, timeframe
Index:   timestamp (DatetimeIndex)
Rows:    ~9000 (1 year of hourly data = 8760 hours)
```

**Metadata: `src/data/jan2025_download/ohlcv/metadata.json`**

```json
{
  "download_date": "2026-01-07T12:00:00",
  "exchange": "binance",
  "symbols": ["BTC", "ETH"],
  "timeframes": ["1h", "4h"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2026-01-07"
  },
  "files_count": 8,
  "total_candles": 35000
}
```

---

## ✅ Validate Data Quality

```python
from src.data_pipeline.validators.ohlcv_validator import OHLCVValidator

# Check single dataset
result = OHLCVValidator.comprehensive_check(
    df=df_btc_1h,
    symbol="BTC/USDT",
    timeframe="1h"
)

print(result)
# {
#   "is_clean": True,
#   "issues": [],
#   "stats": {"missing_bars": 0, "outliers": 0, ...}
# }
```

**Validation Checks:**
1. ✅ OHLC logic (High ≥ max(Open, Close), Low ≤ min(Open, Close))
2. ✅ No missing timestamps (fills gaps)
3. ✅ Outlier detection (price moves > 5σ)
4. ✅ Volume patterns (zero-volume bars)
5. ✅ Gap detection (overnight jumps)

---

## 📊 Simple Factors (Ready for RBI Screening)

### Momentum Factor (RSI-Based)

```python
from src.data_pipeline.factors.momentum_factor import MomentumFactor

factor = MomentumFactor(period=14)
df_signals, stats = factor.generate_signals(df_ohlcv)

print(stats)
# {
#   "factor": "Momentum (RSI)",
#   "buy_signals": 45,
#   "sell_signals": 42,
#   "signal_frequency": 8.5,  # % of bars with signal
# }
```

**Signals:**
- BUY: RSI < 30 (oversold)
- SELL: RSI > 70 (overbought)
- HOLD: 30 ≤ RSI ≤ 70

### Mean Reversion Factor (Bollinger Bands)

```python
from src.data_pipeline.factors.momentum_factor import MeanReversionFactor

factor = MeanReversionFactor(period=20, num_std=2.0)
df_signals, stats = factor.generate_signals(df_ohlcv)
```

### Trend Following Factor (SMA Crossover)

```python
from src.data_pipeline.factors.momentum_factor import TrendFollowingFactor

factor = TrendFollowingFactor(fast_period=5, slow_period=20)
df_signals, stats = factor.generate_signals(df_ohlcv)
```

---

## 🔄 Backtest with Realistic Costs

```python
from src.data_pipeline.backtest.vectorized_engine import VectorizedBacktester
from src.data_pipeline.factors.momentum_factor import MomentumFactor

# Create factor signals
factor = MomentumFactor(period=14)
df_signals, _ = factor.generate_signals(df_ohlcv)

# Run backtest
backtester = VectorizedBacktester(
    initial_cash=10000,
    commission=0.001,    # 0.1% per trade
    slippage_bps=10,     # 10 basis points
)

result = backtest er.run_backtest(
    df=df_ohlcv,
    factor_df=df_signals,
    holding_period=4,  # Hold for 4 bars
)

backtester.print_stats(result['stats'])
```

**Output:**
```
==================================================
BACKTEST RESULTS
==================================================
Total Return:       15.32%
Annual Return:      18.5%
Volatility:         21.3%
Sharpe Ratio:       0.87
Max Drawdown:      -12.5%
Win Rate:          48.3%
Profit Factor:      1.24
Num Trades:        157
Final Equity:     $11,532
==================================================
```

**Key Metrics:**
- **Sharpe Ratio** > 1.0: Good (Jim Simons: > 2.0)
- **Max Drawdown** > -20%: Acceptable
- **Win Rate** > 45%: OK (not primary metric)
- **Profit Factor** > 1.2: Acceptable edge

---

## 🤖 Moon Dev AI Integration

### Step 1: Export Data for RBI Agent

```python
from src.data_pipeline.ai_agents.moondev_integration import MoonDevBridge

bridge = MoonDevBridge(moondev_root="./")

# Export to RBI-friendly format
exported = bridge.export_data_for_rbi(
    data_dict=data,  # Dict of (symbol, timeframe) -> DataFrame
    output_format="csv"
)
```

### Step 2: Trigger RBI Screening

```python
# Generate prompt for RBI agent
prompt = bridge.run_rbi_screening(
    factor_description="""RSI mean reversion with:
    - Periods: 7, 14, 21
    - Buy: RSI < 30
    - Sell: RSI > 70
    - Hold: 4 bars
    """,
    symbols=["BTC", "ETH", "SOL"],
    quick_test=True,
)

print(prompt)  # Copy to RBI agent
```

### Step 3: RBI Agent Tests 50+ Variations

The RBI agent will:
1. Generate backtest code for each parameter combination
2. Run backtests (cost: $0.027 each)
3. Filter by Jim Simons criteria (Sharpe > 1.0)
4. Return top performers

**Cost: $1-2 per screening**  
**Time: 6 hours computation**

### Step 4: Integrate Results

```python
# Load RBI output
results = bridge.integrate_rbi_results(
    rbi_output_dir="src/data/rbi_agent/"
)

# View top strategies
print(results['qualified'])  # Strategies meeting criteria
print(results['top_strategy'])  # Best performer
```

---

## 🧪 Cross-Validation (Jim Simons Standard)

**After RBI screening, validate with:**

### 1. Time-Series Split (60/20/20)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)

for train_idx, test_idx in tscv.split(df):
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    
    # Backtest on train, evaluate on test
    result_train = backtest(df_train)
    result_test = backtest(df_test)
```

### 2. Bootstrap Confidence Intervals

```python
from src.data_pipeline.backtest.vectorized_engine import VectorizedBacktester

# Calculate 95% CI for Sharpe ratio
lower, upper = VectorizedBacktester.bootstrap_ci(
    returns=result['df']['strategy_returns'],
    num_bootstrap=1000,
    ci=0.95
)

print(f"Sharpe 95% CI: [{lower:.2f}, {upper:.2f}]")
```

### 3. Regime Analysis

```python
from hmmlearn import hmm

# Detect bull/bear/sideways regimes
model = hmm.GaussianHMM(n_components=3)
model.fit(df[['returns']].values)

df['regime'] = model.predict(df[['returns']].values)

# Analyze returns by regime
for regime in [0, 1, 2]:
    regime_returns = df[df['regime'] == regime]['strategy_returns']
    print(f"Regime {regime}: Sharpe = {(regime_returns.mean()*252)/regime_returns.std()/np.sqrt(252):.2f}")
```

---

## 🚀 Deployment with Moon Dev

### Export Strategy to Deployment

```python
deployment_info = bridge.prepare_strategy_for_deployment(
    backtest_results=result['stats'],
    strategy_code=strategy_code,
    strategy_name="RSI_Momentum_v1",
)

print(deployment_info)
# {
#   "strategy_name": "RSI_Momentum_v1",
#   "backtest_stats": {...},
#   "deployment_readiness": {
#     "sharpe_ok": True,
#     "drawdown_ok": True,
#     ...
#   },
#   "risk_limits": {
#     "max_position_usd": 1000,
#     "max_loss_daily": 500,
#     "max_leverage": 1.0,
#   }
# }
```

### Deploy via strategy_agent.py

```bash
# From Moon Dev repo
python src/agents/strategy_agent.py
# Select: RSI_Momentum_v1
# Exchange: Hyperliquid
# Position size: $1000
# Start live trading
```

### Monitor with risk_agent.py

```bash
# Risk management circuit breaker (runs first)
python src/agents/risk_agent.py
# Checks: Balance, Loss limits, Position sizes
# Runs before any trading
```

---

## 📈 Performance Goals

| Metric | Target | Comments |
|--------|--------|----------|
| Sharpe Ratio | > 1.0 | Good (Simons: > 2.0) |
| Max Drawdown | < 20% | Risk control |
| Win Rate | > 45% | Not primary metric |
| Profit Factor | > 1.2 | Positive expectation |
| Out-of-Sample | > 80% | Avoid overfitting |
| Bootstrap CI | Positive | Sharpe CI doesn't cross 0 |

---

## 🐛 Troubleshooting

### Download Issues

**Problem: "RateLimitExceeded"**
```
Solution: Built-in retry with exponential backoff
Max retries: 5
Wait time: 2^retry seconds
```

**Problem: "Exchange not responding"**
```
Solution: Check internet connection
Fallback: Use cached data from previous run
```

### Backtest Issues

**Problem: "Sharpe ratio unrealistically high"**
```
Solution: Increase transaction costs
commission = 0.002  # Try 0.2%
slippage_bps = 20   # Try 20 bps
```

**Problem: "Zero trades generated"**
```
Solution: Adjust factor thresholds
RSI_threshold = 35  # Instead of 30
BB_std_dev = 1.5   # Instead of 2.0
```

---

## 📚 References

- **Jim Simons**: "The Man Who Solved the Market" (Gregory Zuckerman)
- **Renaissance Technologies**: CTA + Statistical Arbitrage blend
- **Moon Dev Docs**: `.claude/skills/moon-dev-trading-agents/`
- **CCXT**: https://docs.ccxt.com/

---

## ✅ Checklist: Path to One Validated Strategy

- [ ] Download clean data (OHLCV + funding rates)
- [ ] Validate data quality (outliers, gaps, volume)
- [ ] Implement 3-5 simple factors
- [ ] Backtest with realistic costs (10bps)
- [ ] Run RBI screening (50 variations)
- [ ] Select top performers (Sharpe > 1.0)
- [ ] Cross-validate (60/20/20 split)
- [ ] Bootstrap confidence intervals
- [ ] Regime analysis (when does it work?)
- [ ] Deploy to Moon Dev strategy_agent
- [ ] Monitor with risk_agent
- [ ] Track live P&L
- [ ] Iterate and improve

**Total Time: 2-3 weeks**  
**Total Cost: $50-100 (APIs + screening)**

---

## 🚀 Quick Links

- **Quick Start**: `python scripts/quick_start.py`
- **Configuration**: `src/data_pipeline/config.py`
- **Full Download**: `python scripts/download_all.py` (coming soon)
- **RBI Integration**: `src/data_pipeline/ai_agents/moondev_integration.py`
- **Backtest Engine**: `src/data_pipeline/backtest/vectorized_engine.py`

---

**Built for quantitative traders pursuing Jim Simons methodology using Moon Dev AI agents** 🌙📊
