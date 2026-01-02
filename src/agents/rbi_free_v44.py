"""
🌙 Moon Dev's RBI AI v4.4 - CSV FIX EDITION 🚀

Handles rate limits gracefully with automatic model switching.
Fixed CSV column naming for backtesting.py compatibility.

Features:
- ✅ Auto model fallback on rate limits
- ✅ Fixed CSV column naming (datetime → Datetime, open → Open, etc.)
- ✅ Correct data path to src/data/rbi/BTC-USD-15m.csv
- ✅ Multiple free model support
- ✅ Retry with exponential backoff
- ✅ 100% FREE with safety net

Built with ❤️ by Moon Dev
"""

import os
import sys
import time
import hashlib
import csv
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from termcolor import cprint
from threading import Lock, Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 🌙 LOAD ENVIRONMENT & SETUP
# ============================================

load_dotenv()
print("✅ Environment variables loaded")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rbi_free"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create output folders
CURRENT_DATE = datetime.now().strftime("%m_%d_%Y")
TODAY_DIR = DATA_DIR / CURRENT_DATE
BACKTEST_DIR = TODAY_DIR / "backtests"
FINAL_DIR = TODAY_DIR / "backtests_final"
RESULTS_DIR = TODAY_DIR / "results"

for d in [DATA_DIR, TODAY_DIR, BACKTEST_DIR, FINAL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# 📊 CONFIGURATION
# ============================================

TARGET_RETURN = 50.0
SAVE_IF_OVER_RETURN = 1.0
MAX_PARALLEL_THREADS = 4
MAX_STRATEGIES_PER_BATCH = 50
CONDA_ENV_PATH = "/puhome/22078349d/FYP-code-HPC/env"
API_TIMEOUT = 60
API_MAX_TOKENS = 2000

# 🆓 FREE MODELS (in priority order)
FREE_MODELS = [
    ("qwen/qwen3-coder:free", "Qwen3 Coder 480B (BEST for code)"),
    ("meta-llama/llama-3.1-8b:free", "Llama 3.1 8B (Fast fallback)"),
    ("mistralai/mistral-7b:free", "Mistral 7B (Alternative)"),
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    cprint("❌ OPENROUTER_API_KEY not found in .env", "red", attrs=['bold'])
    sys.exit(1)

cprint(f"✅ OpenRouter API Key loaded", "green")

# File paths
PROCESSED_IDEAS_LOG = DATA_DIR / "processed_ideas.log"
STATS_CSV = DATA_DIR / "backtest_stats.csv"
IDEAS_FILE = DATA_DIR / "ideas.txt"
DATA_FILE = PROJECT_ROOT / "src" / "data" / "rbi" / "BTC-USD-15m.csv"

# Strategy input configuration
STRATEGIES_FROM_FILES = True
STRATEGIES_FOLDER = PROJECT_ROOT / "src" / "data" / "web_search_local" / "final_strategies"

# Locks for thread safety
console_lock = Lock()
file_lock = Lock()
api_lock = Lock()

# Color mapping for threads
THREAD_COLORS = {
    0: "cyan", 1: "magenta", 2: "yellow", 3: "green", 4: "blue",
    5: "white", 6: "cyan", 7: "magenta", 8: "yellow", 9: "green"
}

# ============================================
# 🤖 OPENROUTER CLIENT & MODEL MANAGEMENT
# ============================================

current_model_idx = 0  # Track which model we're using
current_model_lock = Lock()

def get_current_model():
    """Get current active model"""
    with current_model_lock:
        return FREE_MODELS[current_model_idx][0]

def get_current_model_name():
    """Get current active model description"""
    with current_model_lock:
        return FREE_MODELS[current_model_idx][1]

def switch_model():
    """Switch to next available model"""
    global current_model_idx
    with current_model_lock:
        if current_model_idx < len(FREE_MODELS) - 1:
            current_model_idx += 1
            model, desc = FREE_MODELS[current_model_idx]
            cprint(f"\n⚠️ SWITCHED to: {desc}", "yellow", attrs=['bold'])
            return True
        else:
            cprint(f"\n❌ ALL MODELS RATE-LIMITED! No more fallbacks.", "red", attrs=['bold'])
            return False

try:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test connection with current model
    test_response = openrouter_client.chat.completions.create(
        model=get_current_model(),
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10
    )
    
    cprint(f"✨ Connected to {get_current_model_name()}", "green", attrs=['bold'])
    cprint(f"🆓 FREE models with auto-fallback enabled", "cyan")
    
except Exception as e:
    error_str = str(e)
    if "429" in error_str:
        cprint(f"⚠️ Initial model rate-limited, trying fallback...", "yellow", attrs=['bold'])
        if switch_model():
            try:
                test_response = openrouter_client.chat.completions.create(
                    model=get_current_model(),
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
                cprint(f"✨ Fallback successful: {get_current_model_name()}", "green", attrs=['bold'])
            except Exception as e2:
                cprint(f"❌ Fallback also failed: {str(e2)[:100]}", "red", attrs=['bold'])
                sys.exit(1)
        else:
            cprint(f"❌ Failed to initialize OpenRouter: {error_str[:150]}", "red", attrs=['bold'])
            sys.exit(1)
    else:
        cprint(f"❌ Failed to initialize OpenRouter: {error_str[:150]}", "red", attrs=['bold'])
        sys.exit(1)

# ============================================
# 🎨 THREAD-SAFE PRINTING
# ============================================

def thread_print(message, thread_id, color=None):
    """Thread-safe colored print"""
    if color is None:
        color = THREAD_COLORS.get(thread_id % 10, "white")
    
    with console_lock:
        prefix = f"[T{thread_id:02d}]"
        cprint(f"{prefix} {message}", color)

def thread_print_status(thread_id, phase, message):
    """Print status with phase"""
    color = THREAD_COLORS.get(thread_id % 10, "white")
    with console_lock:
        cprint(f"[T{thread_id:02d}] {phase}: {message}", color)

# ============================================
# 📝 AI PROMPTS
# ============================================

RESEARCH_PROMPT = """Analyze this trading strategy. Extract:
1. Strategy name (2 words, specific)
2. Entry rules
3. Exit rules
4. Risk management

Output format:
STRATEGY_NAME: [name]
ENTRY: [rules]
EXIT: [rules]
RISK: [rules]"""

BACKTEST_PROMPT = """Create backtesting.py code for this strategy.
REQUIREMENTS:
1. Use backtesting.py ONLY
2. Read CSV: src/data/rbi/BTC-USD-15m.csv
3. **CRITICAL - Fix CSV columns**:
   import pandas as pd
   df = pd.read_csv('src/data/rbi/BTC-USD-15m.csv')
   df.columns = df.columns.str.strip()  # Remove spaces
   df.rename(columns={
       'datetime': 'Datetime',
       'open': 'Open',
       'high': 'High', 
       'low': 'Low',
       'close': 'Close',
       'volume': 'Volume'
   }, inplace=True)
   df['Datetime'] = pd.to_datetime(df['Datetime'])
   df.set_index('Datetime', inplace=True)
4. Create Backtest(data=df, cash=10000, commission=.002)
5. Use self.I() for talib indicators
6. Size as fraction (0-1) or integer
7. Include if __name__ == "__main__" block
8. Print stats with Return line
9. NO plotting
ONLY CODE, NO TEXT"""

# ============================================
# 🤖 AI COMMUNICATION WITH AUTO-FALLBACK
# ============================================

def chat_with_qwen(system_prompt, user_content, thread_id, max_tokens=None, retry_count=0, model_switch_count=0):
    """OpenRouter API call with auto-fallback to different models"""
    
    if max_tokens is None:
        max_tokens = API_MAX_TOKENS
    
    if retry_count > 2:
        # Exhausted retries for current model, try switching
        if model_switch_count < len(FREE_MODELS) - 1:
            thread_print(f"🔄 Retries exhausted, switching model...", thread_id, "yellow")
            if switch_model():
                return chat_with_qwen(system_prompt, user_content, thread_id, max_tokens, 0, model_switch_count + 1)
            else:
                thread_print(f"❌ All models exhausted", thread_id, "red")
                return None
        else:
            thread_print(f"❌ Max retries and model switches exceeded", thread_id, "red")
            return None
    
    try:
        with api_lock:
            # Limit content
            if len(user_content) > 4000:
                user_content = user_content[:4000] + "..."
            
            response = openrouter_client.chat.completions.create(
                model=get_current_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            # Safe extraction
            if not response or not hasattr(response, 'choices') or not response.choices:
                thread_print(f"⚠️ Empty response", thread_id, "yellow")
                return None
            
            choice = response.choices[0]
            if not choice or not hasattr(choice, 'message'):
                thread_print(f"⚠️ No message", thread_id, "yellow")
                return None
            
            message = choice.message
            if not message or message.content is None:
                thread_print(f"⚠️ Empty content", thread_id, "yellow")
                return None
            
            content = message.content.strip()
            
            # Remove <think> tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if '<think>' in content:
                content = content.split('<think>')[0].strip()
            
            if not content:
                thread_print(f"⚠️ Content empty", thread_id, "yellow")
                return None
            
            time.sleep(0.2)
            return content
            
    except Exception as e:
        error_str = str(e)
        
        # 429 = Rate limited
        if "429" in error_str or "rate" in error_str.lower():
            thread_print(f"⚠️ Rate limited, trying next model...", thread_id, "yellow")
            
            # Try switching model
            if model_switch_count < len(FREE_MODELS) - 1:
                switch_model()
                time.sleep(5)
                return chat_with_qwen(system_prompt, user_content, thread_id, max_tokens, 0, model_switch_count + 1)
            else:
                thread_print(f"❌ All models rate-limited", thread_id, "red")
                return None
        
        # Timeout = retry same model
        elif "timeout" in error_str.lower() or "deadline" in error_str.lower():
            thread_print(f"⏱️ Timeout, retrying... ({retry_count+1}/2)", thread_id, "yellow")
            time.sleep(3)
            return chat_with_qwen(system_prompt, user_content, thread_id, max_tokens, retry_count + 1, model_switch_count)
        
        # 502/503 = Service error
        elif "502" in error_str or "503" in error_str:
            thread_print(f"⚠️ Service error, retrying...", thread_id, "yellow")
            time.sleep(5)
            return chat_with_qwen(system_prompt, user_content, thread_id, max_tokens, retry_count + 1, model_switch_count)
        
        else:
            thread_print(f"❌ API Error: {error_str[:120]}", thread_id, "red")
            return None

# ============================================
# 📄 IDEA PROCESSING
# ============================================

def get_idea_hash(idea):
    """Generate hash for idea"""
    return hashlib.md5(idea.encode('utf-8')).hexdigest()

def is_idea_processed(idea):
    """Check if idea already processed"""
    if not PROCESSED_IDEAS_LOG.exists():
        return False
    
    idea_hash = get_idea_hash(idea)
    with file_lock:
        try:
            with open(PROCESSED_IDEAS_LOG, 'r') as f:
                processed = [line.strip() for line in f if line.strip()]
                return idea_hash in processed
        except:
            return False

def log_processed_idea(idea, strategy_name, thread_id):
    """Log processed idea"""
    idea_hash = get_idea_hash(idea)
    
    with file_lock:
        try:
            with open(PROCESSED_IDEAS_LOG, 'a') as f:
                f.write(f"{idea_hash}\n")
        except Exception as e:
            thread_print(f"⚠️ Failed to log: {e}", thread_id, "yellow")

def read_strategies():
    """Read strategies from file or folder"""
    strategies = []
    
    try:
        if STRATEGIES_FROM_FILES and STRATEGIES_FOLDER.exists():
            # Read from folder
            for file in sorted(STRATEGIES_FOLDER.glob("*.md"))[:MAX_STRATEGIES_PER_BATCH]:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            strategies.append(content)
                except:
                    pass
            
            for file in sorted(STRATEGIES_FOLDER.glob("*.txt"))[:MAX_STRATEGIES_PER_BATCH]:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            strategies.append(content)
                except:
                    pass
        else:
            # Read from ideas.txt
            if IDEAS_FILE.exists():
                with open(IDEAS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    strategies = [line.strip() for line in f if line.strip() and not line.startswith('#')][:MAX_STRATEGIES_PER_BATCH]
    except Exception as e:
        cprint(f"⚠️ Error reading strategies: {e}", "yellow")
    
    return strategies

# ============================================
# 🔄 BACKTEST EXECUTION
# ============================================

def execute_backtest(backtest_file, strategy_name, thread_id):
    """Execute backtest code using custom conda env"""
    try:
        thread_print(f"🔨 Executing backtest...", thread_id, "cyan")
        
        # Use source activate for custom env
        result = subprocess.run(
            f"source {CONDA_ENV_PATH}/bin/activate && python {str(backtest_file)}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        stdout = result.stdout
        
        if result.returncode != 0:
            thread_print(f"❌ Execution failed", thread_id, "red")
            return {"success": False, "stdout": stdout}
        
        # Extract return percentage
        match = re.search(r'Return\s+([-\d.]+)%', stdout)
        if match:
            ret = float(match.group(1))
            thread_print(f"✅ Return: {ret:.2f}%", thread_id, "green", attrs=['bold'])
            return {"success": True, "return": ret, "stdout": stdout}
        else:
            thread_print(f"⚠️ No return found", thread_id, "yellow")
            return {"success": True, "return": 0, "stdout": stdout}
            
    except subprocess.TimeoutExpired:
        thread_print(f"⏱️ Timeout (5min)", thread_id, "yellow")
        return {"success": False}
    except Exception as e:
        thread_print(f"❌ Exec error: {str(e)[:80]}", thread_id, "red")
        return {"success": False}

# ============================================
# 💾 LOGGING
# ============================================

def log_stats_to_csv(strategy_name, thread_id, stats, backtest_file):
    """Log results to CSV"""
    try:
        with file_lock:
            file_exists = STATS_CSV.exists()
            
            with open(STATS_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'Timestamp', 'Strategy Name', 'Thread ID', 'Return %',
                        'File Path'
                    ])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    strategy_name[:50],
                    f"T{thread_id:02d}",
                    stats.get('return', 0),
                    str(backtest_file)
                ])
    except Exception as e:
        pass

# ============================================
# 🎯 STRATEGY PROCESSING
# ============================================

def process_strategy(idea, thread_id):
    """Full pipeline: Research → Code Gen → Execute → Log"""
    
    try:
        if is_idea_processed(idea):
            return
        
        idea_snippet = idea[:40].replace('\n', ' ')
        thread_print(f"🎯 {idea_snippet}...", thread_id, "cyan")
        
        # PHASE 1: RESEARCH
        thread_print_status(thread_id, "RESEARCH", "Analyzing...")
        research_output = chat_with_qwen(RESEARCH_PROMPT, idea, thread_id, max_tokens=1200)
        
        if not research_output:
            thread_print(f"❌ Research failed", thread_id, "red")
            return
        
        # Extract strategy name
        strategy_name = "Strategy"
        if "STRATEGY_NAME:" in research_output:
            try:
                name_part = research_output.split("STRATEGY_NAME:")[1].split('\n')[0].strip()
                strategy_name = re.sub(r'[^a-zA-Z0-9_]', '', name_part)[:30]
            except:
                pass
        
        thread_print(f"📝 {strategy_name}", thread_id, "green")
        
        # PHASE 2: CODE GENERATION
        thread_print_status(thread_id, "BACKTEST", "Generating...")
        backtest_code = chat_with_qwen(BACKTEST_PROMPT, idea, thread_id, max_tokens=3500)
        
        if not backtest_code:
            thread_print(f"❌ Code gen failed", thread_id, "red")
            return
        
        # Clean code
        if '```python' in backtest_code:
            backtest_code = backtest_code.split('```python')[1].split('```')[0].strip()
        elif '```' in backtest_code:
            backtest_code = backtest_code.split('```')[1].split('```')[0].strip()
        
        # Save code
        backtest_file = BACKTEST_DIR / f"T{thread_id:02d}_{strategy_name}_{datetime.now().strftime('%H%M%S')}.py"
        with file_lock:
            with open(backtest_file, 'w') as f:
                f.write(backtest_code)
        
        # PHASE 3: EXECUTION
        exec_result = execute_backtest(backtest_file, strategy_name, thread_id)
        
        if exec_result['success']:
            ret = exec_result.get('return', 0)
            
            if ret >= SAVE_IF_OVER_RETURN:
                final_file = FINAL_DIR / f"{strategy_name}_return_{ret:.2f}.py"
                with file_lock:
                    with open(final_file, 'w') as f:
                        f.write(backtest_code)
                thread_print(f"🎉 SAVED! Return: {ret:.2f}%", thread_id, "green", attrs=['bold'])
            
            log_stats_to_csv(strategy_name, thread_id, {'return': ret}, backtest_file)
        
        # Mark processed
        log_processed_idea(idea, strategy_name, thread_id)
        
    except Exception as e:
        thread_print(f"💥 Fatal: {str(e)[:80]}", thread_id, "red")

# ============================================
# 🚀 MAIN EXECUTION
# ============================================

def main():
    """Main execution"""
    
    print("\n" + "="*60)
    cprint("🌟 Moon Dev's RBI AI v4.4 - CSV FIX EDITION 🚀", "cyan", attrs=['bold'])
    print("="*60 + "\n")
    
    print(f"📅 Date: {CURRENT_DATE}")
    print(f"🎯 Target Return: {TARGET_RETURN}%")
    print(f"🔀 Max Threads: {MAX_PARALLEL_THREADS}")
    print(f"🆓 Current Model: {get_current_model_name()}")
    print(f"🔄 Fallback Models: {len(FREE_MODELS)} available")
    print(f"🐍 Conda Env: {CONDA_ENV_PATH}")
    print(f"📊 Data File: {DATA_FILE}")
    print(f"📂 Output: {DATA_DIR}\n")
    
    if STRATEGIES_FROM_FILES:
        print(f"📁 Reading from: {STRATEGIES_FOLDER}")
    else:
        print(f"📝 Reading from: {IDEAS_FILE}")
    
    # Check data file exists
    if not DATA_FILE.exists():
        cprint(f"❌ Data file not found: {DATA_FILE}", "red", attrs=['bold'])
        sys.exit(1)
    else:
        cprint(f"✅ Data file found: {DATA_FILE}", "green")
    
    print("\n" + "="*60 + "\n")
    
    processed_hashes = set()
    batch_count = 0
    
    while True:
        try:
            strategies = read_strategies()
            
            if not strategies:
                print("⏳ Waiting for strategies...")
                time.sleep(2)
                continue
            
            # Filter new
            new_strategies = []
            for strat in strategies:
                h = get_idea_hash(strat)
                if h not in processed_hashes:
                    new_strategies.append(strat)
                    processed_hashes.add(h)
            
            if not new_strategies:
                time.sleep(1)
                continue
            
            batch_count += 1
            cprint(f"\n📥 Batch #{batch_count}: {len(new_strategies)} new strategy(ies)", "green")
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_THREADS) as executor:
                futures = {
                    executor.submit(process_strategy, strat, i % MAX_PARALLEL_THREADS): strat
                    for i, strat in enumerate(new_strategies)
                }
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass
            
            cprint(f"✅ Batch complete! Total: {len(processed_hashes)}\n", "cyan")
            
        except KeyboardInterrupt:
            cprint(f"\n👋 Shutting down...", "yellow", attrs=['bold'])
            break
        except Exception as e:
            cprint(f"❌ Error: {str(e)[:100]}", "red")
            time.sleep(2)

if __name__ == "__main__":
    main()
