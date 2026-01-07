# 🚀 Jan 2025 Data Pipeline: 5-Minute Quick Start

## What You Get

A **production-grade quant trading pipeline** that:
- 📄 Downloads clean crypto data (OHLCV, funding rates, liquidations)
- ✅ Validates data quality (outliers, gaps, OHLC logic)
- 📊 Tests 3 simple factors with realistic costs
- 🤖 Integrates with Moon Dev RBI Agent for rapid screening
- 🔄 Produces backtest results with Sharpe ratio, drawdown, win rate
- 🔗 Bridges to Moon Dev agents for live deployment

## Prerequisites

```bash
# Have these installed
python --version          # 3.9+
pip --version            # Latest

# Install dependencies
cd moon-dev-ai-agents
pip install -r requirements.txt

# If missing CCXT
pip install ccxt
```

## ⚡ 5-Minute Execute

### Run Quick Start

```bash
python scripts/quick_start.py
```

**What happens:**
1. 📄 Downloads BTC + ETH (1h, 4h timeframes) - **~3 min**
2. ✅ Validates data quality - **~30 sec**
3. 📊 Runs 3 factor backtests (Momentum, Mean Reversion, Trend) - **~1 min**
4. 🤖 Generates Moon Dev RBI Agent prompt - **~30 sec**

**Total time: 5-10 minutes**

### Expected Output

```
============================================================
JAN 2025 DATA DOWNLOAD + FACTOR BACKTEST
============================================================

📄 STEP 1: Download OHLCV Data
Symbols: BTC, ETH
Timeframes: 1h, 4h
Date Range: 2024-01-01 to 2026-01-07
📄 Downloading BTC/USDT 1h from 2024-01-01 00:00:00
...
✅ Downloaded 8 files

📄 STEP 2: Run Factor Backtests
🏗️ Testing BTC/USDT 1h
  🤕 Factor 1: Momentum (RSI)
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

  ... (2 more factors)

📄 STEP 3: Moon Dev AI Agent Integration
  💾 Exporting data for RBI agent...
  ✅ Exported 8 datasets

  🤖 Preparing RBI Agent prompt...

👉 Next Steps:
  1. Run RBI Agent screening (copy prompt above)
  2. Validate top factors with cross-validation
  3. Deploy via Moon Dev strategy_agent.py
  4. Monitor live trading with risk_agent.py

============================================================
✅ Quick start complete!
============================================================
```

## 📊 Where's My Data?

After running `quick_start.py`:

```
src/data/jan2025_download/
├── ohlcv/
│   ├── BTC-USDT_1h.parquet
│   ├── BTC-USDT_4h.parquet
│   ├── ETH-USDT_1h.parquet
│   ├── ETH-USDT_4h.parquet
│   └── metadata.json
└── backtest_results.csv  (if enabled)
```

**File sizes:**
- BTC 1h (1 year): ~20MB
- BTC 4h (1 year): ~5MB
- ETH 1h (1 year): ~18MB
- ETH 4h (1 year): ~5MB
- **Total: ~50MB** (Parquet compression)

## 🚀 Next: Full Download (30 minutes)

Want to download ALL 10 top cryptocurrencies?

```python
# Edit src/data_pipeline/config.py
VALID_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "SUI", "OP"]

# Then run
python scripts/download_all.py
```

**Time: 30-60 minutes**  
**Size: 500MB-1GB**  
**Cost: $0**

## 🤖 Next: RBI Agent Screening (6 hours)

Use Moon Dev's RBI Agent for rapid factor testing.

The `quick_start.py` script outputs a prompt. Copy it:

```bash
# From moon-dev-ai-agents root
python src/agents/rbi_agent_pp_multi.py

# Paste the prompt from quick_start output
# RBI Agent will test 50+ variations automatically
```

**Cost: $1-2 (6 minutes of DeepSeek-R1)**  
**Output: CSV with Sharpe ratios for each variation**

## 🔄 Next: Cross-Validation (2 hours)

Validate winning factors with Jim Simons methodology.

```python
from sklearn.model_selection import TimeSeriesSplit
from src.data_pipeline.backtest.vectorized_engine import VectorizedBacktester

# 60/20/20 train/val/test split
tscv = TimeSeriesSplit(n_splits=3)

for train_idx, test_idx in tscv.split(df):
    result_train = backtest(df.iloc[train_idx])
    result_test = backtest(df.iloc[test_idx])
    print(f"Train Sharpe: {result_train['stats']['sharpe_ratio']:.2f}")
    print(f"Test Sharpe:  {result_test['stats']['sharpe_ratio']:.2f}")
```

## 🔗 Next: Deploy to Moon Dev (1 hour)

Once validated, deploy to Moon Dev agents:

```bash
# Export strategy
from src.data_pipeline.ai_agents.moondev_integration import MoonDevBridge
bridge = MoonDevBridge()
bridge.prepare_strategy_for_deployment(
    backtest_results=result['stats'],
    strategy_code=strategy_code,
    strategy_name="RSI_Momentum_v1",
)

# Deploy via Moon Dev
python src/agents/strategy_agent.py
# Select: RSI_Momentum_v1
# Exchange: Hyperliquid
# Start trading
```

## 💵 Cost Breakdown

| Component | Cost | When |
|-----------|------|------|
| **Data Download** | $0 | Once (via free CCXT) |
| **RBI Screening** | $1-2 | Per 50 factor variations |
| **Live Trading** | Variable | Your position size |
| **Risk Management** | $0 | Built-in (risk_agent) |
| **Total for one strategy** | $2-5 | 2 weeks development |

## 📄 Key Files to Know

| File | Purpose | Edit? |
|------|---------|-------|
| `scripts/quick_start.py` | Execute in 5 min | No |
| `src/data_pipeline/config.py` | Configure symbols/timeframes | Yes |
| `src/data_pipeline/downloaders/ccxt_ohlcv.py` | Download data | No |
| `src/data_pipeline/factors/momentum_factor.py` | Test factors | Yes |
| `src/data_pipeline/backtest/vectorized_engine.py` | Backtest | No |
| `src/data_pipeline/ai_agents/moondev_integration.py` | RBI integration | No |
| `JAN2025_DATA_PIPELINE_README.md` | Full docs | No |

## 🔧 Troubleshooting

### "ModuleNotFoundError: ccxt"

```bash
pip install ccxt
```

### "RateLimitExceeded" during download

Normal! Built-in retry logic handles this.
Wait 2-5 minutes, it will auto-resume.

### "No data after download"

Check:
1. Internet connection
2. Binance API is accessible (try in browser)
3. File permissions: `ls src/data/jan2025_download/ohlcv/`

### "Backtest Sharpe looks unrealistic"

Adjust transaction costs:
```python
backtester = VectorizedBacktester(
    commission=0.002,  # Increase from 0.1%
    slippage_bps=20,   # Increase from 10
)
```

## 🗓️ Common Customizations

### Change symbols

```python
# In quick_start.py
files = download_main(
    symbols=["BTC", "ETH", "SOL"],  # Your symbols
    ...
)
```

### Change timeframes

```python
# In src/data_pipeline/config.py
TIMEFRAMES = ["1h", "4h"]  # Remove 15m and 1d
```

### Change date range

```python
# In src/data_pipeline/config.py
TEST_START_DATE = datetime(2023, 1, 1)  # Longer history
TEST_END_DATE = datetime(2026, 1, 7)
```

### Change backtest settings

```python
# In scripts/quick_start.py
backtester = VectorizedBacktester(
    initial_cash=50000,      # Larger account
    commission=0.002,        # Higher costs
    slippage_bps=20,        # More slippage
)
```

## 🔏 How It Works: Under the Hood

### Data Download
```
Binance REST API (via CCXT)
    ↓
Rate-limited requests (1200/min)
    ↓
Retry logic (exponential backoff)
    ↓
Parquet compression (10x)
    ↓
Local storage with metadata
```

### Backtesting
```
OHLCV data + Factor signals
    ↓
Vectorized operations (NumPy)
    ↓
Transaction costs (realistic)
    ↓
Equity curve calculation
    ↓
Sharpe, Sortino, Max DD metrics
```

### Moon Dev Integration
```
Backtest results
    ↓
Export to RBI format
    ↓
Generate RBI agent prompt
    ↓
RBI tests 50+ variations
    ↓
Filter by Jim Simons criteria
    ↓
Deploy best performers
```

## 📈 Success Criteria

Your factor is **good** if:
- ✅ Sharpe Ratio > 1.0 (>2.0 is excellent)
- ✅ Max Drawdown < 20% (< 10% is excellent)
- ✅ Win Rate > 45% (not the primary metric)
- ✅ Out-of-sample Sharpe > 70% of in-sample
- ✅ Bootstrap CI doesn't cross zero
- ✅ Works in multiple market regimes

Your factor is **problematic** if:
- ❌ Sharpe < 0.5 (no edge)
- ❌ Max Drawdown > 30% (too risky)
- ❌ Out-of-sample Sharpe collapses (overfitted)
- ❌ Deteriorates over time (alpha decay)
- ❌ Breaks in specific regimes (regime-dependent)

## 📦 Full Pipeline Timeline

| Phase | Time | Cost | Checkpoint |
|-------|------|------|------------|
| **1. Data Download** | 5-60 min | $0 | BTC/ETH working |
| **2. Quick Backtest** | 5 min | $0 | Factors generate signals |
| **3. RBI Screening** | 6 hours | $1-2 | 5+ qualified factors |
| **4. Cross-Validation** | 2 hours | $0 | Out-of-sample confirms |
| **5. Regime Analysis** | 2 hours | $0 | Understand when/why |
| **6. Deployment** | 1 hour | $0 | Strategy agent ready |
| **7. Live Testing** | 2-4 weeks | Variable | Real P&L feedback |
| **8. Optimization** | 2-4 weeks | Variable | Parameter tuning |
|
| **TOTAL** | **3-5 weeks** | **$2-5** | **One validated strategy** |

## 🚀 You're Ready!

```bash
python scripts/quick_start.py
```

**Then:**
1. Read full docs: `JAN2025_DATA_PIPELINE_README.md`
2. Copy RBI prompt output
3. Run `python src/agents/rbi_agent_pp_multi.py`
4. Paste prompt and wait 6 hours
5. Get results with 50+ factor variations tested
6. Select top performers
7. Validate with cross-validation
8. Deploy via Moon Dev
9. Monitor with risk_agent
10. Iterate and improve

**Questions?**
See `JAN2025_DATA_PIPELINE_README.md` for comprehensive docs.

---

**Built with Jim Simons methodology × Moon Dev AI agents** 🌙🚀
