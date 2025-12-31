# 🌊 Liquidation Agent V2 - Quick Start Guide

## Overview

**Luna the Liquidation Monitor V2** integrates **websearch_agent_v2 improvements** into the liquidation monitoring system:

- 🤖 **Local Qwen2.5-7B LLM** via HPC Server (192.168.30.158:8000)
- 📊 **Structured CSV Logging** (liquidation events, analysis results, deduplication logs)
- 📈 **Similarity Detection** (avoid duplicate signals)
- 📁 **Content Hash Deduplication** (MD5-based)
- ✅ **Production-Grade Error Handling** (exponential backoff, timeouts)
- 📝 **Detailed Metrics Tracking** (quality scores, confidence levels)

## Files

| File | Purpose | Status |
|------|---------|--------|
| `src/agents/liquidation_agent_v2.py` | Main agent implementation | ✅ Ready |
| `src/agents/liquidation_agent.py` | Original v1 (kept for reference) | ✅ Available |

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/bigsuperFishfish/moon-dev-ai-agents.git
cd moon-dev-ai-agents
```

### 2. Checkout Feature Branch

```bash
git checkout feature/improved-websearch-agent-v2
```

### 3. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Key packages needed:
# - pandas>=1.0.0
# - requests>=2.25.0
# - termcolor>=1.1.0
# - python-dotenv>=0.19.0
# - openai (optional)
# - anthropic (optional)
```

### 4. Configure Environment

```bash
# Copy example and configure
cp .env_example .env

# Edit .env and set:
# - OPENAI_KEY (optional)
# - ANTHROPIC_KEY (optional)
# - API keys for Moon Dev API
```

## Running in HPC/Apptainer Environment

### Method 1: Direct Python Execution

```bash
# From project root directory
cd /path/to/moon-dev-ai-agents

# Run the agent
python src/agents/liquidation_agent_v2.py
```

### Method 2: With PYTHONPATH Explicitly Set

```bash
cd /path/to/moon-dev-ai-agents

PYTHONPATH=/path/to/moon-dev-ai-agents:/path/to/moon-dev-ai-agents/src \
  python src/agents/liquidation_agent_v2.py
```

### Method 3: Using Apptainer/Singularity

```bash
# If using Apptainer container
apptainer run --bind /puhome/YOUR_USER:/home \
  /path/to/container.sif \
  python /home/moon-dev-ai-agents/src/agents/liquidation_agent_v2.py
```

### Method 4: In HPC Job Script

```bash
#!/bin/bash
#SBATCH --job-name=liquidation_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

module load python/3.10

cd /puhome/${USER}/moon-dev-ai-agents

python src/agents/liquidation_agent_v2.py
```

## Output & Data Storage

All data automatically saved to: `src/data/liquidation_v2/`

### CSV Files

- **`liquidation_events.csv`** - Raw liquidation events
  - timestamp, long_size, short_size, total_size
  - long_change_pct, short_change_pct, total_change_pct
  - event_hash, processed flag

- **`analysis_results.csv`** - LLM analysis outputs
  - timestamp, event_hash, signal (BUY/SELL/NOTHING)
  - confidence (0.0-1.0), reason, liquidation amounts
  - market_context, similarity_score

- **`deduplication_log.csv`** - Deduplication decisions
  - timestamp, event_hash, signal, similarity_score
  - decision (SAVED/SKIPPED), reason, similar_to

### Logs

- **`logs/liquidation_agent_YYYYMMDD_HHMMSS.log`** - Detailed logs
  - All INFO, DEBUG, WARNING, ERROR messages
  - Timestamps and module names

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution 1: Run from project root**
```bash
cd /path/to/moon-dev-ai-agents  # Must be project root
python src/agents/liquidation_agent_v2.py
```

**Solution 2: Set PYTHONPATH**
```bash
PYTHONPATH=/path/to/moon-dev-ai-agents python src/agents/liquidation_agent_v2.py
```

**Solution 3: Install as package**
```bash
cd /path/to/moon-dev-ai-agents
pip install -e .
```

### Issue: `ConnectionError: Cannot connect to LLM server`

**Checklist:**
1. Verify HPC server is running: `ping 192.168.30.158`
2. Check port 8000 is accessible: `curl http://192.168.30.158:8000/health`
3. Verify network connection from Apptainer: `--nv` flag for GPU support
4. Check firewall rules (if applicable)

### Issue: `pandas not found` / `requests not found`

**Solution:**
```bash
pip install pandas requests termcolor python-dotenv
```

### Issue: CSV files not created

**Solution:**
Check write permissions in `src/data/liquidation_v2/`
```bash
mkdir -p src/data/liquidation_v2
chmod 755 src/data/liquidation_v2
```

## Configuration Options

Edit the `LiquidationConfig` class in `liquidation_agent_v2.py`:

```python
class LiquidationConfig:
    # Check interval (minutes)
    CHECK_INTERVAL_MINUTES = 10
    
    # Liquidation data points
    LIQUIDATION_ROWS = 10000
    
    # Alert threshold (% increase)
    LIQUIDATION_THRESHOLD = 0.5
    
    # LLM settings
    LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
    LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_TIMEOUT_SECONDS = 120
    LLM_MAX_RETRIES = 2
```

## Monitoring the Agent

### Real-time Logs

```bash
# Watch logs as they're written
tail -f src/data/liquidation_v2/logs/liquidation_agent_*.log

# Or use less for pagination
less +F src/data/liquidation_v2/logs/liquidation_agent_*.log
```

### Check CSV Data

```bash
# View latest liquidation events
head -20 src/data/liquidation_v2/liquidation_events.csv

# View analysis results
head -20 src/data/liquidation_v2/analysis_results.csv

# View deduplication decisions
head -20 src/data/liquidation_v2/deduplication_log.csv
```

### With Python

```python
import pandas as pd

# Load and analyze data
events = pd.read_csv('src/data/liquidation_v2/liquidation_events.csv')
results = pd.read_csv('src/data/liquidation_v2/analysis_results.csv')

print(f"Total events: {len(events)}")
print(f"Buy signals: {len(results[results['signal'] == 'BUY'])}")
print(f"Sell signals: {len(results[results['signal'] == 'SELL'])}")
print(f"Average confidence: {results['confidence'].mean():.2%}")
```

## Key Improvements Over V1

| Feature | V1 | V2 |
|---------|----|---------|
| LLM | OpenAI/Anthropic API | ✅ Local Qwen (HPC) |
| Data Logging | Minimal | ✅ Comprehensive CSV |
| Similarity Detection | None | ✅ TF-IDF + Cosine |
| Content Dedup | None | ✅ MD5 Hashes |
| Error Handling | Basic | ✅ Exponential Backoff |
| Metrics Tracking | Basic | ✅ Detailed Scores |
| Reproducibility | Limited | ✅ Full Audit Trail |

## Architecture

```
🌊 Liquidation Agent V2
├─📡 LiquidationEvent (data model)
├─📊 AnalysisResult (data model)
├─📈 LocalLLMProvider (Qwen via HPC)
├─📈 EventSimilarityDetector (dedup)
├─📈 ContentHasher (MD5 hashing)
├─📈 CSVManager (CSV initialization)
├─📈 DataLogger (CSV writing)
├─🧪 LiquidationAgent (main orchestrator)
└─🚀 main() (entry point)
```

## Next Steps

1. **Test LLM Connection**
   ```bash
   python -c "from src.agents.liquidation_agent_v2 import LocalLLMProvider; p = LocalLLMProvider(); print(p.generate_response('test', 'hello'))"
   ```

2. **Run First Cycle**
   ```bash
   python src/agents/liquidation_agent_v2.py
   # Press Ctrl+C after one cycle
   ```

3. **Check Output**
   ```bash
   ls -la src/data/liquidation_v2/
   head src/data/liquidation_v2/analysis_results.csv
   ```

4. **Integrate into Production**
   - Set up cron job or scheduler
   - Monitor log files
   - Analyze CSV data regularly

## Support & Issues

If you encounter issues:

1. Check logs: `src/data/liquidation_v2/logs/`
2. Verify HPC connectivity: `curl http://192.168.30.158:8000/health`
3. Check Python path: `python -c "import sys; print(sys.path)"`
4. Verify dependencies: `pip list | grep -E "pandas|requests|termcolor"`

## License

Moon Dev Trading Agents © 2025
