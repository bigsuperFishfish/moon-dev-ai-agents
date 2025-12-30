# WebSearch Agent V2 - Quick Start

## What Changed?

The new `websearch_agent_v2.py` is a complete rewrite with:

✅ **1,150+ lines** of production-grade code  
✅ **4 CSV logging systems** for data tracking  
✅ **Intelligent URL caching** to prevent duplicates  
✅ **Quality scoring** for strategies (0.0-1.0)  
✅ **Comprehensive error handling** with retry logic  
✅ **Professional logging** system  
✅ **Performance metrics** auto-calculated  
✅ **100% backward compatible** with original  

---

## Quick Start

### 1. Run the agent
```bash
cd /path/to/moon-dev-ai-agents
python src/agents/websearch_agent_v2.py
```

### 2. What you'll see
```
🌙 MOON DEV WEB SEARCH AGENT V2 (PRODUCTION-GRADE)
🤖 Local LLM: Qwen2.5-7B (via LM Studio)
🔄 Sleep between cycles: 120s
📁 Output directory: src/data/web_search_local/final_strategies

======================================================================
🌙 CYCLE #1
======================================================================
✨ Generated query: momentum RSI strategy 4h parameters
...
[Processing 8 search results]
...
======================================================================
🎉 CYCLE COMPLETE
======================================================================
✅ URLs processed: 6
✅ Strategies extracted: 12
✅ Success rate: 75.0%
✅ Duration: 120.4s
📁 Saved to: src/data/web_search_local/final_strategies
```

### 3. Check results

**Extracted strategies**:
```bash
ls -la src/data/web_search_local/final_strategies/
cat src/data/web_search_local/final_strategies/strategy_*.md
```

**View logs**:
```bash
tail -f src/data/web_search_local/logs/websearch_agent_*.log
```

**Analyze metrics**:
```bash
cat src/data/web_search_local/metrics.csv | head -5
```

---

## Output Locations

```
src/data/web_search_local/
├── final_strategies/              # 📍 Your extracted strategies (markdown files)
├── logs/                          # 📍 Detailed logs
├── cache/
│   └── url_cache.json             # Processed URLs (prevents duplicates)
├── search_results.csv             # All search operations
├── extraction_log.csv             # Strategy extraction details
├── strategy_quality.csv           # Quality metrics per strategy
└── metrics.csv                    # Cycle performance data
```

---

## Configuration (Optional)

Edit `src/agents/websearch_agent_v2.py` line 65-95 to customize:

```python
class SearchConfig:
    # How many results to fetch per search
    MAX_SEARCH_RESULTS = 8  # increase for more content
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2.0
    
    # Content length bounds
    MAX_CONTENT_LENGTH = 15000
    MIN_CONTENT_LENGTH = 300
    
    # Timing
    TIMEOUT_SECONDS = 20
    SLEEP_BETWEEN_SEARCHES = 120  # Cooldown between cycles
    
    # Quality threshold
    MIN_STRATEGY_QUALITY_SCORE = 0.6  # 60% minimum quality
```

---

## Key Features

### 🎯 Quality Scoring
Each extracted strategy gets a score 0.0-1.0 based on:
- **Completeness** (20%): All 8 fields filled
- **Specificity** (30%): Uses concrete parameters
- **Actionability** (25%): Can be implemented
- **Clarity** (25%): Well-formatted

Strategies below `MIN_STRATEGY_QUALITY_SCORE` are filtered out.

### 🔄 Smart Caching
- Persistent URL tracking (`url_cache.json`)
- Content hashing prevents re-processing
- Automatically skips already-processed URLs

### 📊 Auto-calculated Metrics
Every cycle tracks:
- Queries generated
- Search results found
- URLs processed
- Strategies extracted
- Success rate (strategies / URLs)
- Average quality score
- Total duration
- API calls made

### 🛡️ Error Handling
- Timeouts → Retry with exponential backoff
- HTTP errors → Log and continue
- Parsing errors → Graceful degradation
- LLM failures → Return empty results

### 📝 Professional Logging
- Dual output: file + colored console
- Emoji indicators for quick scanning
- Full timestamps and context
- Separate log file per run

---

## CSV Files Explained

### 1. `metrics.csv` - Overall performance
```csv
timestamp,cycle,queries_generated,search_results,urls_processed,strategies_extracted,success_rate,avg_quality_score,duration_seconds
2025-12-30T10:47:15,1,1,8,6,12,0.75,0.81,120.4
```
Best for: Tracking trends over time

### 2. `strategy_quality.csv` - Quality per strategy
```csv
timestamp,strategy_name,quality_score,completeness,specificity,actionability,source_url
2025-12-30T10:46:01,RSI Breakout Strategy,0.85,High,High,High,https://example.com
```
Best for: Filtering by quality

### 3. `extraction_log.csv` - Per-URL extraction
```csv
timestamp,url,num_strategies,average_quality,extraction_duration_ms
2025-12-30T10:45:45,https://example.com,3,0.78,3400
```
Best for: Analyzing content sources

### 4. `search_results.csv` - Search operations
```csv
timestamp,cycle,query,url,title,status,content_length
2025-12-30T10:45:32,1,momentum RSI strategy,https://example.com,Page Title,success,8234
```
Best for: Auditing search operations

---

## Common Tasks

### 📊 Analyze quality trends
```bash
python -c "
import pandas as pd
df = pd.read_csv('src/data/web_search_local/strategy_quality.csv')
print('Average quality:', df['quality_score'].mean().round(2))
print('High quality (>0.75):', (df['quality_score'] > 0.75).sum())
print('By completeness:')
print(df['completeness'].value_counts())
"
```

### 📈 Check cycle performance
```bash
python -c "
import pandas as pd
df = pd.read_csv('src/data/web_search_local/metrics.csv')
print('Total strategies:', df['strategies_extracted'].sum())
print('Average success rate:', df['success_rate'].mean().round(1))
print('Total duration:', df['duration_seconds'].sum() / 60, 'minutes')
"
```

### 🔍 Find best sources
```bash
python -c "
import pandas as pd
df = pd.read_csv('src/data/web_search_local/extraction_log.csv')
df_sorted = df.sort_values('num_strategies', ascending=False)
print(df_sorted[['url', 'num_strategies', 'average_quality']].head(10))
"
```

---

## Troubleshooting

### ❓ Agent not finding strategies
- Check logs: `tail -f src/data/web_search_local/logs/websearch_agent_*.log`
- Lower `MIN_STRATEGY_QUALITY_SCORE` to 0.4 temporarily
- Increase `MAX_SEARCH_RESULTS` to 12

### ❓ Timeout errors
- Increase `TIMEOUT_SECONDS` to 30
- Increase `SLEEP_BETWEEN_FETCHES` to 5
- Check internet connection

### ❓ LLM connection issues
- Ensure LM Studio is running on port 8000
- Check: `curl http://127.0.0.1:8000/v1/chat/completions`
- Verify Qwen2.5-7B model is loaded

### ❓ Duplicate URLs in logs
- Cache might be corrupted: Delete `src/data/web_search_local/cache/url_cache.json`
- Agent will rebuild cache on next run

---

## Next Steps

1. **Run for 5 cycles** (~10 minutes) to get baseline data
2. **Analyze metrics.csv** to see performance
3. **Review top strategies** in final_strategies/
4. **Adjust MIN_STRATEGY_QUALITY_SCORE** based on results
5. **Run as scheduled task** using cron/scheduler

---

## Comparison: V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Lines of code | ~250 | 1,150+ |
| URL deduplication | CSV check | Persistent cache |
| Error handling | Minimal | Comprehensive |
| Logging system | Print statements | Professional logging |
| Quality scoring | None | Weighted algorithm |
| CSV files | 2 | 4 |
| Performance tracking | None | Full metrics |
| Type hints | No | Yes |
| Data models | Dicts | Dataclasses |
| Retry logic | None | Exponential backoff |

---

## Files

- **Main agent**: `src/agents/websearch_agent_v2.py` (1,150 lines)
- **Full docs**: `docs/websearch_agent_v2_improvements.md`
- **Original agent**: `src/agents/websearch_agent.py` (still available)

---

## Questions?

Check the detailed documentation:
```bash
cat docs/websearch_agent_v2_improvements.md
```

Or review the code with inline comments:
```bash
less src/agents/websearch_agent_v2.py
```

---

**Built with 🌙 by Moon Dev**

*Never over-engineer, always ship real trading systems.*
