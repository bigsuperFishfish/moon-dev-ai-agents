# 🌙 Quick Fix: Groq Import Error

## 問題
```
⚠️ Could not import model_factory: No module named 'groq'
```

## 解決方案

### Step 1: 安裝缺失的依賴
```bash
# 方法 A: 重新安裝所有依賴
pip install -r requirements.txt

# 或者方法 B: 只安裝 groq
pip install groq==0.16.0
```

### Step 2: 驗證安裝
```bash
python -c "import groq; print(f'Groq {groq.__version__} installed successfully')"
```

### Step 3: 重新運行
```bash
python src/agents/rbi_agent_pp_multi_v2.py
```

---

## 修復說明

我已經修復了 `src/models/model_factory.py`，現在它使用 **lazy loading** 和 **proper error handling**：

```python
# 舊的方式 (會崩潰):
from .groq_model import GroqModel  # ❌ 如果 groq 沒安裝就會崩潰

# 新的方式 (安全):
try:
    from .groq_model import GroqModel
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    GroqModel = None
```

現在即使 groq 沒安裝，應用也能：
- ✅ 正常啟動
- ✅ 使用其他模型 (Claude, OpenAI, Gemini 等)
- ⚠️ 顯示警告說 Groq 不可用

---

## 檢查所有模型狀態

運行此命令來檢查所有模型的可用性：

```bash
python -c "
from src.models.model_factory import model_factory
print('Available models:')
for model_type in model_factory._models:
    print(f'  ✅ {model_type}')
if not model_factory._models:
    print('  ⚠️ No models available')
"
```

---

## 如果仍然失敗

檢查 Python 版本和環境：
```bash
python --version
pip --version
echo $VIRTUAL_ENV  # 確保你在虛擬環境中
```

如果不在虛擬環境中：
```bash
# 創建新的虛擬環境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 重新安裝
pip install -r requirements.txt
```
