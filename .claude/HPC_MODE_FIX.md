# 🌙 HPC-Only Mode Fix (No Groq/Cloud APIs Needed)

## 問題
當 `USE_LOCAL_HPC_LLM = True` 時，不需要任何 Cloud API 或 Groq，但是 `model_factory` 導致 import 錯誤。

## 解決方案

### 選項 1: 快速修復 (推薦)

在 `src/agents/rbi_agent_pp_multi_v2.py` 中，將：

```python
# OLD (line ~70):
try:
    from src.models import model_factory
    print("✅ Successfully imported model_factory")
except ImportError as e:
    print(f"⚠️ Could not import model_factory: {e}")
    sys.exit(1)
```

替換為：

```python
# NEW: Only import model_factory if NOT using HPC LLM
if not USE_LOCAL_HPC_LLM:
    try:
        from src.models import model_factory
        print("✅ Successfully imported model_factory")
    except ImportError as e:
        print(f"❌ Could not import model_factory: {e}")
        print("❌ Cannot use cloud API fallback without model_factory")
        sys.exit(1)
else:
    cprint("⏭️  Skipping model_factory import (HPC-only mode)", "cyan")
    model_factory = None
```

### 選項 2: 安裝 Groq (不推薦)

如果您想保留雲端 fallback 功能：

```bash
pip install groq==0.16.0
```

## 修復後確認

運行時應該看到：

```
✅ Environment variables loaded

======================================================================
🌙 HPC LLM CONFIGURATION 🌙
======================================================================
✅ LOCAL HPC LLM ENABLED
   URL: http://192.168.30.158:8000/v1/chat/completions
   Model: Qwen/Qwen2.5-7B-Instruct
   Timeout: 120s
   Max Retries: 2
======================================================================

⏭️  Skipping model_factory import (HPC-only mode)  <-- 👍 正確！
```

## 功能說明

### HPC Mode (USE_LOCAL_HPC_LLM = True)
- ✅ 使用 Qwen 2.5 7B 本地 LLM
- ✅ 不需要任何 Cloud API keys
- ✅ 不需要 model_factory
- ✅ 自動 retry + timeout 處理

### Fallback Mode (USE_LOCAL_HPC_LLM = False)
- 📡 使用 DeepSeek/OpenRouter/其他 cloud APIs
- ✅ 需要 model_factory
- ✅ 需要 API keys 在 `.env`

## 相關配置

在 `rbi_agent_pp_multi_v2.py` 中：

```python
# ============================================
# 🌙 HPC LLM CONFIGURATION
# ============================================
USE_LOCAL_HPC_LLM = True  # 設為 False 使用 Cloud APIs
LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_LLM_TIMEOUT = 120  # seconds
LOCAL_LLM_MAX_RETRIES = 2
```

## 確認 HPC LLM 運行中

```bash
# 測試 HPC LLM 是否可用
curl -X POST http://192.168.30.158:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }'
```

應該返回 JSON response，而不是連接錯誤。

---

🌙 **Moon Dev's Trading Agents - HPC Optimized**
