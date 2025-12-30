"""
🌙 Moon Dev's RBI AI v3.0 PARALLEL PROCESSOR + HPC LLM INTEGRATION 🚀
Built with love by Moon Dev 🚀

PARALLEL PROCESSING + MULTI-DATA VALIDATION + HPC LLM:
Run up to 18 backtests simultaneously using local Qwen 2.5 7B model on HPC,
each tested on 25+ different data sources!

- Each thread processes a different trading idea
- Thread-safe colored output
- Rate limiting to avoid API throttling
- 🌙 HPC LOCAL LLM SUPPORT: Direct connection to Qwen 2.5 7B
- 🔄 AUTOMATIC FALLBACK: Switches to DeepSeek API if local LLM unavailable
- ⏱️ RETRY LOGIC: 120s timeout with 2 retry attempts
- 🎯 MULTI-SOURCE STRATEGY READING:
  1. ideas.txt (one strategy per line - classic mode)
  2. src/data/web_search_research/final_strategies/ (markdown files)
  3. src/data/web_search_local/final_strategies/ (websearch_agent_v2.py output)

- Each thread independently: Research → Backtest → Debug → Optimize
- 📊 Each successful backtest automatically tests on 25+ data sources!
- All threads run simultaneously until target returns are hit
- Thread-safe file naming with unique 2-digit thread IDs
- Multi-data results saved to ./results/ folders for each strategy

NEW FEATURES IN V2:
- 🌙 HPC LLM INTEGRATION with OpenAI-compatible API
- 🔄 Automatic retry + timeout handling for LLM calls
- 📁 Multi-source strategy reading (3 sources in priority order)
- 🎨 Color-coded output per thread (Thread 1 = cyan, Thread 2 = magenta, etc.)
- ⏱️ Rate limiting to avoid API throttling
- 🔒 Thread-safe file operations
- 📊 Real-time progress tracking across all threads
- 💾 Clean file organization with thread IDs in names
- 📈 MULTI-DATA TESTING: Validates strategies on 25+ assets/timeframes automatically!
- 📊 CSV results showing performance across all data sources

STRATEGY SOURCE PRIORITY (IMPORTANT!):
1. ✅ Check src/data/web_search_local/final_strategies/ (websearch_agent_v2.py)
2. ✅ Check src/data/web_search_research/final_strategies/ (other websearch agent)
3. ✅ Fall back to ideas.txt (manual input, one per line)

Required Setup:
1. Conda environment 'tflow' with backtesting packages
2. Set MAX_PARALLEL_THREADS (default: 18)
3. 🌙 HPC SETUP:
   - FastAPI server running on http://192.168.30.158:8000/v1
   - Qwen 2.5 7B model loaded with 4-bit quantization
   - Or set USE_LOCAL_HPC_LLM=False to use DeepSeek API
4. External dependency: moon-dev-trading-bots repo (clone as sibling: ../moon-dev-trading-bots)
5. Run and watch all ideas process in parallel with multi-data validation! 🚀💰

IMPORTANT: Each thread is fully independent and won't interfere with others!
"""

# Import execution functionality
import subprocess
import json
from pathlib import Path

# Core imports
import os
import time
import re
import hashlib
import csv
import pandas as pd
from datetime import datetime
from termcolor import cprint
import sys
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore, Thread
from queue import Queue
import requests
from io import BytesIO

# Load environment variables FIRST
load_dotenv()
print("✅ Environment variables loaded")

# Add config values directly to avoid import issues
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 16000

# Import model factory with proper path handling
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models import model_factory
    print("✅ Successfully imported model_factory")
except ImportError as e:
    print(f"⚠️ Could not import model_factory: {e}")
    sys.exit(1)

# ============================================
# 🌙 HPC LLM CONFIGURATION - V2 NEW!
# ============================================
USE_LOCAL_HPC_LLM = True  # Set to False to use DeepSeek API instead
LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_LLM_TIMEOUT = 120  # seconds for LLM processing
LOCAL_LLM_MAX_RETRIES = 2  # retry attempts for failed calls
LOCAL_LLM_RETRY_DELAY = 30  # seconds to wait between retries

print(f"\n{'='*70}")
cprint(f"🌙 HPC LLM CONFIGURATION 🌙", "cyan", attrs=['bold'])
print(f"{'='*70}")
if USE_LOCAL_HPC_LLM:
    cprint(f"✅ LOCAL HPC LLM ENABLED", "green", attrs=['bold'])
    print(f"   URL: {LOCAL_LLM_URL}")
    print(f"   Model: {LOCAL_LLM_MODEL}")
    print(f"   Timeout: {LOCAL_LLM_TIMEOUT}s")
    print(f"   Max Retries: {LOCAL_LLM_MAX_RETRIES}")
else:
    cprint(f"📡 DEEPSEEK API MODE (Fallback)", "yellow", attrs=['bold'])
    print(f"   Using OpenRouter API for LLM calls")
print(f"{'='*70}\n")

# ============================================
# 🎯 PARALLEL PROCESSING CONFIGURATION
# ============================================
MAX_PARALLEL_THREADS = 18
RATE_LIMIT_DELAY = .5
RATE_LIMIT_GLOBAL_DELAY = 0.5

# ============================================
# 📁 STRATEGY SOURCE CONFIGURATION - Moon Dev V2
# ============================================
# NEW IN V2: Multi-source strategy reading
# Priority order: websearch_local → websearch_research → ideas.txt

WEBSEARCH_LOCAL_DIR = None  # Will be set dynamically
WEBSEARCH_RESEARCH_DIR = None  # Will be set dynamically
IDEAS_FILE = None  # Will be set dynamically

# Thread color mapping
THREAD_COLORS = {
    0: "cyan",
    1: "magenta",
    2: "yellow",
    3: "green",
    4: "blue"
}

# Global locks
console_lock = Lock()
api_lock = Lock()
file_lock = Lock()
date_lock = Lock()

# Rate limiter
rate_limiter = Semaphore(MAX_PARALLEL_THREADS)

# ============================================
# 🌙 Moon Dev's MODEL SELECTION - Easy Switch!
# ============================================
RESEARCH_CONFIG = {"type": "claude", "name": "claude-opus-4-5-20251101"}
BACKTEST_CONFIG = {"type": "claude", "name": "claude-opus-4-5-20251101"}
DEBUG_CONFIG = {"type": "claude", "name": "claude-opus-4-5-20251101"}
PACKAGE_CONFIG = {"type": "claude", "name": "claude-opus-4-5-20251101"}
OPTIMIZE_CONFIG = {"type": "claude", "name": "claude-opus-4-5-20251101"}

# 🎯 PROFIT TARGET CONFIGURATION
TARGET_RETURN = 50
SAVE_IF_OVER_RETURN = 1.0
CONDA_ENV = "tflow"
MAX_DEBUG_ITERATIONS = 10
MAX_OPTIMIZATION_ITERATIONS = 10
EXECUTION_TIMEOUT = 300

# 🌙 Moon Dev: Date tracking
CURRENT_DATE = datetime.now().strftime("%m_%d_%Y")

# Update data directory paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"

# 🌙 Moon Dev: These will be updated dynamically when date changes
TODAY_DIR = None
RESEARCH_DIR = None
BACKTEST_DIR = None
PACKAGE_DIR = None
WORKING_BACKTEST_DIR = None
FINAL_BACKTEST_DIR = None
OPTIMIZATION_DIR = None
CHARTS_DIR = None
EXECUTION_DIR = None

PROCESSED_IDEAS_LOG = None
STATS_CSV = None

def init_paths():
    """Initialize all paths and strategy sources - V2 NEW!"""
    global DATA_DIR, WEBSEARCH_LOCAL_DIR, WEBSEARCH_RESEARCH_DIR, IDEAS_FILE
    global PROCESSED_IDEAS_LOG, STATS_CSV

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "src" / "data"
    
    # Strategy sources - V2 NEW
    WEBSEARCH_LOCAL_DIR = DATA_DIR / "web_search_local" / "final_strategies"
    WEBSEARCH_RESEARCH_DIR = DATA_DIR / "web_search_research" / "final_strategies"
    RBI_DATA_DIR = PROJECT_ROOT / "src" / "data" / "rbi_pp_multi"
    IDEAS_FILE = RBI_DATA_DIR / "ideas.txt"
    
    # Logs
    PROCESSED_IDEAS_LOG = RBI_DATA_DIR / "processed_ideas.log"
    STATS_CSV = RBI_DATA_DIR / "backtest_stats.csv"
    
    # Create directories if needed
    RBI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEBSEARCH_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    WEBSEARCH_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

init_paths()

def update_date_folders():
    """🌙 Moon Dev's Date Folder Updater!"""
    global CURRENT_DATE, TODAY_DIR, RESEARCH_DIR, BACKTEST_DIR, PACKAGE_DIR
    global WORKING_BACKTEST_DIR, FINAL_BACKTEST_DIR, OPTIMIZATION_DIR, CHARTS_DIR, EXECUTION_DIR

    with date_lock:
        new_date = datetime.now().strftime("%m_%d_%Y")

        if new_date != CURRENT_DATE:
            with console_lock:
                cprint(f"\n🌅 NEW DAY DETECTED! {CURRENT_DATE} → {new_date}", "cyan", attrs=['bold'])
                cprint(f"📁 Creating new folder structure for {new_date}...\n", "yellow")

            CURRENT_DATE = new_date

        RBI_DATA_DIR = PROJECT_ROOT / "src" / "data" / "rbi_pp_multi"
        TODAY_DIR = RBI_DATA_DIR / CURRENT_DATE
        RESEARCH_DIR = TODAY_DIR / "research"
        BACKTEST_DIR = TODAY_DIR / "backtests"
        PACKAGE_DIR = TODAY_DIR / "backtests_package"
        WORKING_BACKTEST_DIR = TODAY_DIR / "backtests_working"
        FINAL_BACKTEST_DIR = TODAY_DIR / "backtests_final"
        OPTIMIZATION_DIR = TODAY_DIR / "backtests_optimized"
        CHARTS_DIR = TODAY_DIR / "charts"
        EXECUTION_DIR = TODAY_DIR / "execution_results"

        for dir in [RBI_DATA_DIR, TODAY_DIR, RESEARCH_DIR, BACKTEST_DIR, PACKAGE_DIR,
                    WORKING_BACKTEST_DIR, FINAL_BACKTEST_DIR, OPTIMIZATION_DIR, CHARTS_DIR, EXECUTION_DIR]:
            dir.mkdir(parents=True, exist_ok=True)

update_date_folders()

# ============================================
# 🎨 THREAD-SAFE PRINTING
# ============================================

def thread_print(message, thread_id, color=None, attrs=None):
    """Thread-safe colored print with thread ID prefix"""
    if color is None:
        color = THREAD_COLORS.get(thread_id % 5, "white")

    with console_lock:
        prefix = f"[T{thread_id:02d}]"
        cprint(f"{prefix} {message}", color, attrs=attrs)

def thread_print_status(thread_id, phase, message):
    """Print status update for a specific phase"""
    color = THREAD_COLORS.get(thread_id % 5, "white")
    with console_lock:
        cprint(f"[T{thread_id:02d}] {phase}: {message}", color)

# ============================================
# 🌙 HPC LLM FUNCTIONS - V2 NEW!
# ============================================

def call_local_hpc_llm(messages: list, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    """
    Call local HPC LLM (Qwen 2.5 7B) with retry logic and timeout handling
    Returns: Response text or None if failed
    """
    import time
    
    for attempt in range(LOCAL_LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(
                LOCAL_LLM_URL,
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                headers={"Content-Type": "application/json"},
                timeout=LOCAL_LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                if attempt < LOCAL_LLM_MAX_RETRIES:
                    print(f"   ⚠️ HPC LLM returned {response.status_code}, retrying...")
                    time.sleep(LOCAL_LLM_RETRY_DELAY)
                    continue
                else:
                    print(f"   ❌ HPC LLM error {response.status_code}: {response.text[:200]}")
                    return None
                    
        except requests.exceptions.Timeout:
            if attempt < LOCAL_LLM_MAX_RETRIES:
                print(f"   ⏳ HPC LLM timeout (attempt {attempt + 1}/{LOCAL_LLM_MAX_RETRIES + 1}), retrying...")
                time.sleep(LOCAL_LLM_RETRY_DELAY * (attempt + 1))
                continue
            else:
                print(f"   ❌ HPC LLM timeout after {LOCAL_LLM_MAX_RETRIES + 1} attempts")
                return None
        except Exception as e:
            if attempt < LOCAL_LLM_MAX_RETRIES:
                print(f"   ⚠️ HPC LLM error: {str(e)}, retrying...")
                time.sleep(LOCAL_LLM_RETRY_DELAY)
                continue
            else:
                print(f"   ❌ HPC LLM failed: {str(e)}")
                return None
    
    return None

def call_llm(system_prompt: str, user_content: str, thread_id: int, max_tokens: int = 4096, temperature: float = 0.7) -> str:
    """
    Call LLM - either local HPC or cloud DeepSeek
    Automatically switches to fallback if local LLM unavailable
    """
    if USE_LOCAL_HPC_LLM:
        # Try local HPC LLM first
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        result = call_local_hpc_llm(messages, max_tokens, temperature)
        
        if result is not None:
            return result
        
        # Fallback to cloud API if local fails
        thread_print(f"⚠️ Falling back to DeepSeek API", thread_id, "yellow")
    
    # Use cloud model (DeepSeek or other)
    model = model_factory.get_model(RESEARCH_CONFIG["type"], RESEARCH_CONFIG["name"])
    if not model:
        thread_print(f"❌ Could not initialize model", thread_id, "red")
        return None
    
    response = model.generate_response(
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    if response:
        return response.content if hasattr(response, 'content') else str(response)
    return None

# ============================================
# 📄 PDF & YOUTUBE EXTRACTION
# ============================================

def get_youtube_transcript(video_id, thread_id):
    """Get transcript from YouTube video"""
    try:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            thread_print("⚠️ youtube-transcript-api not installed", thread_id, "yellow")
            return None

        thread_print(f"🎥 Fetching transcript for video ID: {video_id}", thread_id, "cyan")

        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en'])

        transcript_text = ' '.join([snippet.text for snippet in transcript_data])

        thread_print(f"✅ Transcript extracted! Length: {len(transcript_text)} characters", thread_id, "green")

        preview = transcript_text[:300].replace('\n', ' ')
        thread_print(f"📝 Preview: {preview}...", thread_id, "cyan")

        return transcript_text
    except Exception as e:
        thread_print(f"❌ Error fetching YouTube transcript: {e}", thread_id, "red")
        return None

def get_pdf_text(url, thread_id):
    """Extract text from PDF URL"""
    try:
        try:
            import PyPDF2
        except ImportError:
            thread_print("⚠️ PyPDF2 not installed", thread_id, "yellow")
            return None

        thread_print(f"📚 Fetching PDF from: {url[:60]}...", thread_id, "cyan")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        thread_print("📖 Extracting text from PDF...", thread_id, "cyan")
        reader = PyPDF2.PdfReader(BytesIO(response.content))
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text + '\n'

        thread_print(f"✅ PDF extracted! Pages: {len(reader.pages)}, Length: {len(text)} characters", thread_id, "green")

        preview = text[:300].replace('\n', ' ')
        thread_print(f"📝 Preview: {preview}...", thread_id, "cyan")

        return text
    except Exception as e:
        thread_print(f"❌ Error reading PDF: {e}", thread_id, "red")
        return None

def extract_youtube_id(url):
    """Extract video ID from YouTube URL"""
    try:
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            video_id = url.split("/")[-1].split("?")[0]
        return video_id
    except:
        return None

def extract_content_from_url(idea: str, thread_id: int) -> str:
    """🌙 Moon Dev: Extract content from PDF or YouTube URLs"""
    idea = idea.strip()

    if "youtube.com" in idea or "youtu.be" in idea:
        video_id = extract_youtube_id(idea)
        if video_id:
            transcript = get_youtube_transcript(video_id, thread_id)
            if transcript:
                return f"Strategy from YouTube video:\n\n{transcript}"
            else:
                with console_lock:
                    cprint("="*80, "white", "on_red", attrs=['bold'])
                    cprint(f"⚠️  YOUTUBE EXTRACTION FAILED - Sleeping 30s", "white", "on_red", attrs=['bold'])
                    cprint("="*80, "white", "on_red", attrs=['bold'])
                time.sleep(30)
                return idea

    elif idea.endswith(".pdf") or "pdf" in idea.lower():
        pdf_text = get_pdf_text(idea, thread_id)
        if pdf_text:
            return f"Strategy from PDF document:\n\n{pdf_text}"
        else:
            with console_lock:
                cprint("="*80, "white", "on_red", attrs=['bold'])
                cprint(f"⚠️  PDF EXTRACTION FAILED - Sleeping 30s", "white", "on_red", attrs=['bold'])
                cprint("="*80, "white", "on_red", attrs=['bold'])
            time.sleep(30)
            return idea

    return idea

# ============================================
# 📝 PROMPTS (Same as v1)
# ============================================

RESEARCH_PROMPT = """
You are Moon Dev's Research AI 🌙

IMPORTANT NAMING RULES:
1. Create a UNIQUE TWO-WORD NAME for this specific strategy
2. The name must be DIFFERENT from any generic names like "TrendFollower" or "MomentumStrategy"
3. First word should describe the main approach (e.g., Adaptive, Neural, Quantum, Fractal, Dynamic)
4. Second word should describe the specific technique (e.g., Reversal, Breakout, Oscillator, Divergence)
5. Make the name SPECIFIC to this strategy's unique aspects

Examples of good names:
- "AdaptiveBreakout" for a strategy that adjusts breakout levels
- "FractalMomentum" for a strategy using fractal analysis with momentum
- "QuantumReversal" for a complex mean reversion strategy
- "NeuralDivergence" for a strategy focusing on divergence patterns

BAD names to avoid:
- "TrendFollower" (too generic)
- "SimpleMoving" (too basic)
- "PriceAction" (too vague)

Output format must start with:
STRATEGY_NAME: [Your unique two-word name]

Then analyze the trading strategy content and create detailed instructions.
Focus on:
1. Key strategy components
2. Entry/exit rules
3. Risk management
4. Required indicators

Your complete output must follow this format:
STRATEGY_NAME: [Your unique two-word name]

STRATEGY_DETAILS:
[Your detailed analysis]

Remember: The name must be UNIQUE and SPECIFIC to this strategy's approach!
"""

BACKTEST_PROMPT = """
You are Moon Dev's Backtest AI 🌙

🚨 CRITICAL: Your code MUST have TWO parts:
PART 1: Strategy class definition
PART 2: if __name__ == "__main__" block (SEE TEMPLATE BELOW - MANDATORY!)

If you don't include the if __name__ == "__main__" block with stats printing, the code will FAIL!

Create a backtesting.py implementation for the strategy.
USE BACKTESTING.PY
Include:
1. All necessary imports
2. Strategy class with indicators
3. Entry/exit logic
4. Risk management
5. your size should be 1,000,000
6. If you need indicators use TA lib or pandas TA.

IMPORTANT DATA HANDLING:
1. Clean column names by removing spaces: data.columns = data.columns.str.strip().str.lower()
2. Drop any unnamed columns: data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
3. Ensure proper column mapping to match backtesting requirements:
   - Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
   - Use proper case (capital first letter)

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.

INDICATOR CALCULATION RULES:
1. ALWAYS use self.I() wrapper for ANY indicator calculations
2. Use talib functions instead of pandas operations:
   - Instead of: self.data.Close.rolling(20).mean()
   - Use: self.I(talib.SMA, self.data.Close, timeperiod=20)
3. For swing high/lows use talib.MAX/MIN:
   - Instead of: self.data.High.rolling(window=20).max()
   - Use: self.I(talib.MAX, self.data.High, timeperiod=20)

BACKTEST EXECUTION ORDER:
1. Run initial backtest with default parameters first
2. Print full stats using print(stats) and print(stats._strategy)
3. no optimization code needed, just print the final stats, make sure full stats are printed, not just part or some. stats = bt.run() print(stats) is an example of the last line of code. no need for plotting ever.

❌ NEVER USE bt.plot() - IT CAUSES TIMEOUTS IN PARALLEL PROCESSING!
❌ NO PLOTTING, NO CHARTS, NO VISUALIZATIONS!
✅ ONLY PRINT STATS TO CONSOLE!

CRITICAL POSITION SIZING RULE:
When calculating position sizes in backtesting.py, the size parameter must be either:
1. A fraction between 0 and 1 (for percentage of equity)
2. A whole number (integer) of units

The common error occurs when calculating position_size = risk_amount / risk, which results in floating-point numbers. Always use:
position_size = int(round(position_size))

Example fix:
❌ self.buy(size=3546.0993)  # Will fail
✅ self.buy(size=int(round(3546.0993)))  # Will work

RISK MANAGEMENT:
1. Always calculate position sizes based on risk percentage
2. Use proper stop loss and take profit calculations
4. Print entry/exit signals with Moon Dev themed messages

If you need indicators use TA lib or pandas TA.

Use this data path: src/data/rbi/BTC-USD-15m.csv (relative to project root)
the above data head looks like below
datetime, open, high, low, close, volume,
2023-01-01 00:00:00, 16531.83, 16532.69, 16509.11, 16510.82, 231.05338022,
2023-01-01 00:15:00, 16509.78, 16534.66, 16509.11, 16533.43, 308.12276951,

Always add plenty of Moon Dev themed debug prints with emojis to make debugging easier! 🌙 ✨ 🚀

🚨🚨🚨 MANDATORY EXECUTION BLOCK - DO NOT SKIP THIS! 🚨🚨🚨

YOU ABSOLUTELY MUST INCLUDE THIS BLOCK AT THE END OF YOUR CODE!
WITHOUT THIS BLOCK, THE STATS CANNOT BE PARSED AND THE BACKTEST WILL FAIL!

Copy this EXACT template and replace YourStrategyClassName with your actual class name:

```python
# 🌙 MOON DEV'S MULTI-DATA TESTING FRAMEWORK 🚀
# Tests this strategy on 25+ data sources automatically!
if __name__ == "__main__":
    import sys
    import os
    from backtesting import Backtest
    import pandas as pd

    # FIRST: Run standard backtest and print stats (REQUIRED for parsing!)
    print("\\n🌙 Running initial backtest for stats extraction...")
    data = pd.read_csv('src/data/rbi/BTC-USD-15m.csv')  # 🌙 Moon Dev: Relative path from project root!
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    bt = Backtest(data, YourStrategyClassName, cash=1_000_000, commission=0.002)
    stats = bt.run()

    # 🌙 CRITICAL: Print full stats for Moon Dev's parser!
    print("\\n" + "="*80)
    print("📊 BACKTEST STATISTICS (Moon Dev's Format)")
    print("="*80)
    print(stats)
    print("="*80 + "\\n")

    # THEN: Run multi-data testing
    # 🌙 Moon Dev: EXTERNAL DEPENDENCY - Requires moon-dev-trading-bots repo
    # Clone from: https://github.com/moondevonyt/moon-dev-trading-bots
    # Expected location: ../moon-dev-trading-bots (sibling to this repo)
    external_backtests = str(Path(__file__).parent.parent.parent.parent / 'moon-dev-trading-bots' / 'backtests')
    sys.path.append(external_backtests)
    from multi_data_tester import test_on_all_data

    print("\\n" + "="*80)
    print("🚀 MOON DEV'S MULTI-DATA BACKTEST - Testing on 25+ Data Sources!")
    print("="*80)

    # Test this strategy on all configured data sources
    # This will test on: BTC, ETH, SOL (multiple timeframes), AAPL, TSLA, ES, NQ, GOOG, NVDA
    # IMPORTANT: verbose=False to prevent plotting (causes timeouts in parallel processing!)
    results = test_on_all_data(YourStrategyClassName, 'YourStrategyName', verbose=False)

    if results is not None:
        print("\\n✅ Multi-data testing complete! Results saved in ./results/ folder")
        print(f"📊 Tested on {len(results)} different data sources")
    else:
        print("\\n⚠️ No results generated - check for errors above")
```

IMPORTANT: Replace 'YourStrategyClassName' with your actual strategy class name!
IMPORTANT: Replace 'YourStrategyName' with a descriptive name for the CSV output!

🚨 FINAL REMINDER 🚨
Your response MUST include:
1. Import statements
2. Strategy class (with init() and next() methods)
3. The if __name__ == "__main__" block shown above (MANDATORY!)

Do NOT send ONLY the strategy class. You MUST include the execution block!
ONLY SEND BACK CODE, NO OTHER TEXT.

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

DEBUG_PROMPT = """
You are Moon Dev's Debug AI 🌙
Fix technical issues in the backtest code WITHOUT changing the strategy logic.

CRITICAL ERROR TO FIX:
{error_message}

CRITICAL DATA LOADING REQUIREMENTS:
The CSV file has these exact columns after processing:
- datetime, open, high, low, close, volume (all lowercase after .str.lower())
- After capitalization: Datetime, Open, High, Low, Close, Volume

CRITICAL BACKTESTING REQUIREMENTS:
1. Data Loading Rules:
   - Use data.columns.str.strip().str.lower() to clean columns
   - Drop unnamed columns: data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
   - Rename columns properly: data.rename(columns={{'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}})
   - Set datetime as index: data = data.set_index(pd.to_datetime(data['datetime']))

2. Position Sizing Rules:
   - Must be either a fraction (0 < size < 1) for percentage of equity
   - OR a positive whole number (round integer) for units
   - NEVER use floating point numbers for unit-based sizing

3. Indicator Issues:
   - Cannot use .shift() on backtesting indicators
   - Use array indexing like indicator[-2] for previous values
   - All indicators must be wrapped in self.I()

4. Position Object Issues:
   - Position object does NOT have .entry_price attribute
   - Use self.trades[-1].entry_price if you need entry price from last trade
   - Available position attributes: .size, .pl, .pl_pct
   - For partial closes: use self.position.close() without parameters (closes entire position)
   - For stop losses: use sl= parameter in buy/sell calls, not in position.close()

5. No Trades Issue (Signals but no execution):
   - If strategy prints "ENTRY SIGNAL" but shows 0 trades, the self.buy() call is not executing
   - Common causes: invalid size parameter, insufficient cash, missing self.buy() call
   - Ensure self.buy() is actually called in the entry condition block
   - Check size parameter: must be fraction (0-1) or positive integer
   - Verify cash/equity is sufficient for the trade size

Focus on:
1. KeyError issues with column names
2. Syntax errors and import statements
3. Indicator calculation methods
4. Data loading and preprocessing
5. Position object attribute errors (.entry_price, .close() parameters)

DO NOT change strategy logic, entry/exit conditions, or risk management rules.

Return the complete fixed code with Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

PACKAGE_PROMPT = """
You are Moon Dev's Package AI 🌙
Your job is to ensure the backtest code NEVER uses ANY backtesting.lib imports or functions.

❌ STRICTLY FORBIDDEN:
1. from backtesting.lib import *
2. import backtesting.lib
3. from backtesting.lib import crossover
4. ANY use of backtesting.lib

✅ REQUIRED REPLACEMENTS:
1. For crossover detection:
   Instead of: backtesting.lib.crossover(a, b)
   Use: (a[-2] < b[-2] and a[-1] > b[-1])  # for bullish crossover
        (a[-2] > b[-2] and a[-1] < b[-1])  # for bearish crossover

2. For indicators:
   - Use talib for all standard indicators (SMA, RSI, MACD, etc.)
   - Use pandas-ta for specialized indicators
   - ALWAYS wrap in self.I()

3. For signal generation:
   - Use numpy/pandas boolean conditions
   - Use rolling window comparisons with array indexing
   - Use mathematical comparisons (>, <, ==)

Example conversions:
❌ from backtesting.lib import crossover
❌ if crossover(fast_ma, slow_ma):
✅ if fast_ma[-2] < slow_ma[-2] and fast_ma[-1] > slow_ma[-1]:

❌ self.sma = self.I(backtesting.lib.SMA, self.data.Close, 20)
✅ self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)

IMPORTANT: Scan the ENTIRE code for any backtesting.lib usage and replace ALL instances!
Return the complete fixed code with proper Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

OPTIMIZE_PROMPT = """
You are Moon Dev's Optimization AI 🌙
Your job is to IMPROVE the strategy to achieve higher returns while maintaining good risk management.

CURRENT PERFORMANCE:
Return [%]: {current_return}%
TARGET RETURN: {target_return}%

YOUR MISSION: Optimize this strategy to hit the target return!

OPTIMIZATION TECHNIQUES TO CONSIDER:
1. **Entry Optimization:**
   - Tighten entry conditions to catch better setups
   - Add filters to avoid low-quality signals
   - Use multiple timeframe confirmation
   - Add volume/momentum filters

2. **Exit Optimization:**
   - Improve take profit levels
   - Add trailing stops
   - Use dynamic position sizing based on volatility
   - Scale out of positions

3. **Risk Management:**
   - Adjust position sizing
   - Use volatility-based position sizing (ATR)
   - Add maximum drawdown limits
   - Improve stop loss placement

4. **Indicator Optimization:**
   - Fine-tune indicator parameters
   - Add complementary indicators
   - Use indicator divergence
   - Combine multiple timeframes

5. **Market Regime Filters:**
   - Add trend filters
   - Avoid choppy/ranging markets
   - Only trade in favorable conditions

IMPORTANT RULES:
- DO NOT break the code structure
- Keep all Moon Dev debug prints
- Maintain proper backtesting.py format
- Use self.I() for all indicators
- Position sizes must be int or fraction (0-1)
- Focus on REALISTIC improvements (no curve fitting!)
- Explain your optimization changes in comments

Return the COMPLETE optimized code with Moon Dev themed comments explaining what you improved! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

# ============================================
# 🛠️ HELPER FUNCTIONS
# ============================================

def parse_return_from_output(stdout: str, thread_id: int) -> float:
    """Extract the Return [%] from backtest output"""
    try:
        match = re.search(r'Return \[%\]\s+([-\d.]+)', stdout)
        if match:
            return_pct = float(match.group(1))
            thread_print(f"📊 Extracted return: {return_pct}%", thread_id)
            return return_pct
        else:
            thread_print("⚠️ Could not find Return [%] in output", thread_id, "yellow")
            return None
    except Exception as e:
        thread_print(f"❌ Error parsing return: {str(e)}", thread_id, "red")
        return None

def parse_all_stats_from_output(stdout: str, thread_id: int) -> dict:
    """🌙 Moon Dev's Stats Parser - Extract all key stats from backtest output!"""
    stats = {
        'return_pct': None,
        'buy_hold_pct': None,
        'max_drawdown_pct': None,
        'sharpe': None,
        'sortino': None,
        'expectancy': None,
        'trades': None
    }

    try:
        match = re.search(r'Return \[%\]\s+([-\d.]+)', stdout)
        if match:
            stats['return_pct'] = float(match.group(1))

        match = re.search(r'Buy & Hold Return \[%\]\s+([-\d.]+)', stdout)
        if match:
            stats['buy_hold_pct'] = float(match.group(1))

        match = re.search(r'Max\. Drawdown \[%\]\s+([-\d.]+)', stdout)
        if match:
            stats['max_drawdown_pct'] = float(match.group(1))

        match = re.search(r'Sharpe Ratio\s+([-\d.]+)', stdout)
        if match:
            stats['sharpe'] = float(match.group(1))

        match = re.search(r'Sortino Ratio\s+([-\d.]+)', stdout)
        if match:
            stats['sortino'] = float(match.group(1))

        match = re.search(r'Expectancy \[%\]\s+([-\d.]+)', stdout)
        if not match:
            match = re.search(r'Avg\. Trade \[%\]\s+([-\d.]+)', stdout)
        if match:
            stats['expectancy'] = float(match.group(1))

        match = re.search(r'# Trades\s+(\d+)', stdout)
        if match:
            stats['trades'] = int(match.group(1))

        match = re.search(r'Exposure Time \[%\]\s+([-\d.]+)', stdout)
        if match:
            stats['exposure'] = float(match.group(1))

        thread_print(f"📊 Extracted {sum(1 for v in stats.values() if v is not None)}/8 stats", thread_id)
        return stats

    except Exception as e:
        thread_print(f"❌ Error parsing stats: {str(e)}", thread_id, "red")
        return stats

def log_stats_to_csv(strategy_name: str, thread_id: int, stats: dict, file_path: str, data_source: str = "BTC-USD-15m.csv") -> None:
    """🌙 Moon Dev's CSV Logger - Thread-safe stats logging!"""
    try:
        with file_lock:
            file_exists = STATS_CSV.exists()

            with open(STATS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        'Strategy Name',
                        'Thread ID',
                        'Return %',
                        'Buy & Hold %',
                        'Max Drawdown %',
                        'Sharpe Ratio',
                        'Sortino Ratio',
                        'Exposure %',
                        'EV %',
                        'Trades',
                        'File Path',
                        'Data',
                        'Time'
                    ])
                    thread_print("📝 Created new stats CSV with headers", thread_id, "green")

                timestamp = datetime.now().strftime("%m/%d %H:%M")
                writer.writerow([
                    strategy_name,
                    f"T{thread_id:02d}",
                    stats.get('return_pct', 'N/A'),
                    stats.get('buy_hold_pct', 'N/A'),
                    stats.get('max_drawdown_pct', 'N/A'),
                    stats.get('sharpe', 'N/A'),
                    stats.get('sortino', 'N/A'),
                    stats.get('exposure', 'N/A'),
                    stats.get('expectancy', 'N/A'),
                    stats.get('trades', 'N/A'),
                    str(file_path),
                    data_source,
                    timestamp
                ])

                thread_print(f"✅ Logged stats to CSV (Return: {stats.get('return_pct', 'N/A')}% on {data_source})", thread_id, "green")

    except Exception as e:
        thread_print(f"❌ Error logging to CSV: {str(e)}", thread_id, "red")

def get_idea_hash(idea: str) -> str:
    """Generate a unique hash for an idea"""
    return hashlib.md5(idea.encode('utf-8')).hexdigest()

def is_idea_processed(idea: str) -> bool:
    """Check if an idea has already been processed"""
    if not PROCESSED_IDEAS_LOG.exists():
        return False

    idea_hash = get_idea_hash(idea)

    with file_lock:
        with open(PROCESSED_IDEAS_LOG, 'r') as f:
            processed_hashes = [line.strip().split(',')[0] for line in f if line.strip()]

    return idea_hash in processed_hashes

def log_processed_idea(idea: str, strategy_name: str, thread_id: int) -> None:
    """Log an idea as processed"""
    idea_hash = get_idea_hash(idea)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with file_lock:
        if not PROCESSED_IDEAS_LOG.exists():
            PROCESSED_IDEAS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(PROCESSED_IDEAS_LOG, 'w') as f:
                f.write("# Moon Dev's RBI AI - Processed Ideas Log 🌙\n")
                f.write("# Format: hash,timestamp,thread_id,strategy_name,idea_snippet\n")

        idea_snippet = idea[:50].replace(',', ';') + ('...' if len(idea) > 50 else '')
        with open(PROCESSED_IDEAS_LOG, 'a') as f:
            f.write(f"{idea_hash},{timestamp},T{thread_id:02d},{strategy_name},{idea_snippet}\n")

    thread_print(f"📝 Logged processed idea: {strategy_name}", thread_id, "green")

def execute_backtest(file_path: str, strategy_name: str, thread_id: int) -> dict:
    """Execute a backtest file and capture output"""
    thread_print(f"🚀 Executing: {strategy_name}", thread_id)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", str(file_path)
    ]

    start_time = datetime.now()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=EXECUTION_TIMEOUT
    )

    execution_time = (datetime.now() - start_time).total_seconds()

    output = {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat()
    }

    result_file = EXECUTION_DIR / f"T{thread_id:02d}_{strategy_name}_{datetime.now().strftime('%H%M%S')}.json"
    with file_lock:
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)

    if output['success']:
        thread_print(f"✅ Backtest executed in {execution_time:.2f}s!", thread_id, "green")
    else:
        thread_print(f"❌ Backtest failed: {output['return_code']}", thread_id, "red")

    return output

def clean_model_output(output, content_type="text"):
    """Clean model output by removing thinking tags and extracting code"""
    cleaned_output = output

    if "<think>" in output and "</think>" in output:
        clean_content = output.split("</think>")[-1].strip()
        if not clean_content:
            import re
            clean_content = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        if clean_content:
            cleaned_output = clean_content

    if content_type == "code" and "```" in cleaned_output:
        try:
            import re
            code_blocks = re.findall(r'```python\n(.*?)\n```', cleaned_output, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', cleaned_output, re.DOTALL)

            if not code_blocks:
                match = re.search(r'```(?:python)?\s*\n(.*)', cleaned_output, re.DOTALL)
                if match:
                    cleaned_output = match.group(1).strip()
                    if cleaned_output.endswith('```'):
                        cleaned_output = cleaned_output[:-3].strip()
            else:
                cleaned_output = "\n\n".join(code_blocks)
        except Exception as e:
            print(f"❌ Error extracting code: {str(e)}")

    if content_type == "code":
        cleaned_output = cleaned_output.replace('```python', '').replace('```', '').strip()

    return cleaned_output

# ============================================
# 🤖 AI AGENT FUNCTIONS
# ============================================

def research_strategy(content, thread_id):
    """Research AI: Analyzes and creates trading strategy"""
    thread_print_status(thread_id, "🔍 RESEARCH", "Starting analysis...")

    output = call_llm(RESEARCH_PROMPT, content, thread_id, AI_MAX_TOKENS, AI_TEMPERATURE)

    if output:
        output = clean_model_output(output, "text")

        strategy_name = "UnknownStrategy"
        if "STRATEGY_NAME:" in output:
            try:
                name_section = output.split("STRATEGY_NAME:")[1].strip()
                if "\n\n" in name_section:
                    strategy_name = name_section.split("\n\n")[0].strip()
                else:
                    strategy_name = name_section.split("\n")[0].strip()

                strategy_name = re.sub(r'[^\w\s-]', '', strategy_name)
                strategy_name = re.sub(r'[\s]+', '', strategy_name)

                thread_print(f"✅ Strategy: {strategy_name}", thread_id, "green")
            except Exception as e:
                thread_print(f"⚠️ Error extracting strategy name: {str(e)}", thread_id, "yellow")

        filepath = RESEARCH_DIR / f"T{thread_id:02d}_{strategy_name}_strategy.txt"
        with file_lock:
            with open(filepath, 'w') as f:
                f.write(output)

        return output, strategy_name
    return None, None

def create_backtest(strategy, strategy_name, thread_id):
    """Backtest AI: Creates backtest implementation"""
    thread_print_status(thread_id, "📊 BACKTEST", "Creating backtest code...")

    output = call_llm(BACKTEST_PROMPT, f"Create a backtest for this strategy:\n\n{strategy}", thread_id, AI_MAX_TOKENS, 0.3)

    if output:
        output = clean_model_output(output, "code")

        filepath = BACKTEST_DIR / f"T{thread_id:02d}_{strategy_name}_BT.py"
        with file_lock:
            with open(filepath, 'w') as f:
                f.write(output)

        thread_print(f"🔥 Backtest code saved", thread_id, "green")
        return output
    return None

def package_check(backtest_code, strategy_name, thread_id):
    """Package AI: Ensures correct packages are used"""
    thread_print_status(thread_id, "📦 PACKAGE", "Checking imports...")

    output = call_llm(PACKAGE_PROMPT, f"Check and fix indicator packages in this code:\n\n{backtest_code}", thread_id, AI_MAX_TOKENS, 0.3)

    if output:
        output = clean_model_output(output, "code")

        filepath = PACKAGE_DIR / f"T{thread_id:02d}_{strategy_name}_PKG.py"
        with file_lock:
            with open(filepath, 'w') as f:
                f.write(output)

        thread_print(f"📦 Package check complete", thread_id, "green")
        return output
    return None

def debug_backtest(backtest_code, error_message, strategy_name, thread_id, iteration=1):
    """Debug AI: Fixes technical issues"""
    thread_print_status(thread_id, f"🔧 DEBUG #{iteration}", "Fixing errors...")

    debug_prompt_with_error = DEBUG_PROMPT.format(error_message=error_message)

    output = call_llm(debug_prompt_with_error, f"Fix this backtest code:\n\n{backtest_code}", thread_id, AI_MAX_TOKENS, 0.3)

    if output:
        output = clean_model_output(output, "code")

        filepath = BACKTEST_DIR / f"T{thread_id:02d}_{strategy_name}_DEBUG_v{iteration}.py"
        with file_lock:
            with open(filepath, 'w') as f:
                f.write(output)

        thread_print(f"🔧 Debug iteration {iteration} complete", thread_id, "green")
        return output
    return None

def optimize_strategy(backtest_code, current_return, target_return, strategy_name, thread_id, iteration=1):
    """Optimization AI: Improves strategy"""
    thread_print_status(thread_id, f"🎯 OPTIMIZE #{iteration}", f"{current_return}% → {target_return}%")

    optimize_prompt_with_stats = OPTIMIZE_PROMPT.format(
        current_return=current_return,
        target_return=target_return
    )

    output = call_llm(optimize_prompt_with_stats, f"Optimize this backtest code to hit the target:\n\n{backtest_code}", thread_id, AI_MAX_TOKENS, 0.3)

    if output:
        output = clean_model_output(output, "code")

        filepath = OPTIMIZATION_DIR / f"T{thread_id:02d}_{strategy_name}_OPT_v{iteration}.py"
        with file_lock:
            with open(filepath, 'w') as f:
                f.write(output)

        thread_print(f"🎯 Optimization {iteration} complete", thread_id, "green")
        return output
    return None

# ============================================
# 📁 STRATEGY LOADING - V2 NEW! Multi-source
# ============================================

def get_strategies_from_markdown_files(folder_path):
    """🌙 Moon Dev V2: Read all markdown files from a folder"""
    strategies = []
    
    if not folder_path.exists():
        return strategies
    
    for file_path in sorted(folder_path.glob('*.md')):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    strategies.append(content)
        except Exception as e:
            with console_lock:
                cprint(f"⚠️ Error reading {file_path.name}: {str(e)}", "yellow")
    
    return strategies

def collect_all_strategies():
    """🌙 Moon Dev V2: Collect strategies from all sources in priority order!
    
    Priority:
    1. websearch_local/final_strategies/ (websearch_agent_v2.py output) - HIGHEST PRIORITY
    2. websearch_research/final_strategies/ (other websearch agents)
    3. ideas.txt (manual input)
    """
    all_strategies = []
    
    # Priority 1: websearch_local
    cprint(f"\n📁 Checking websearch_local for strategies...", "cyan")
    local_strategies = get_strategies_from_markdown_files(WEBSEARCH_LOCAL_DIR)
    if local_strategies:
        cprint(f"✅ Found {len(local_strategies)} strategies from websearch_local", "green", attrs=['bold'])
        all_strategies.extend(local_strategies)
    else:
        cprint(f"⏭️  No strategies in websearch_local yet", "yellow")
    
    # Priority 2: websearch_research
    cprint(f"\n📁 Checking websearch_research for strategies...", "cyan")
    research_strategies = get_strategies_from_markdown_files(WEBSEARCH_RESEARCH_DIR)
    if research_strategies:
        cprint(f"✅ Found {len(research_strategies)} strategies from websearch_research", "green", attrs=['bold'])
        all_strategies.extend(research_strategies)
    else:
        cprint(f"⏭️  No strategies in websearch_research yet", "yellow")
    
    # Priority 3: ideas.txt
    cprint(f"\n📄 Checking ideas.txt for strategies...", "cyan")
    if IDEAS_FILE.exists():
        with open(IDEAS_FILE, 'r') as f:
            ideas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if ideas:
            cprint(f"✅ Found {len(ideas)} strategies from ideas.txt", "green", attrs=['bold'])
            all_strategies.extend(ideas)
        else:
            cprint(f"⏭️  No strategies in ideas.txt yet", "yellow")
    else:
        cprint(f"⏭️  ideas.txt not found", "yellow")
    
    cprint(f"\n{'='*70}", "green", attrs=['bold'])
    cprint(f"📊 TOTAL STRATEGIES AVAILABLE: {len(all_strategies)}", "green", attrs=['bold'])
    cprint(f"{'='*70}\n", "green", attrs=['bold'])
    
    return all_strategies

def idea_monitor_thread(idea_queue: Queue, queued_ideas: set, queued_lock: Lock, stop_flag: dict):
    """🌙 Moon Dev V2: Producer thread - monitors all strategy sources"""
    
    while not stop_flag.get('stop', False):
        try:
            # Collect strategies from all sources (with priority order)
            ideas = collect_all_strategies()
            
            for idea in ideas:
                idea_hash = get_idea_hash(idea)

                with queued_lock:
                    already_queued = idea_hash in queued_ideas

                if not is_idea_processed(idea) and not already_queued:
                    idea_queue.put(idea)
                    with queued_lock:
                        queued_ideas.add(idea_hash)
                    with console_lock:
                        idea_preview = idea[:80].replace('\n', ' ')
                        cprint(f"🆕 NEW IDEA QUEUED: {idea_preview}...", "green", attrs=['bold'])

            time.sleep(5)  # Check every 5 seconds

        except Exception as e:
            with console_lock:
                cprint(f"❌ Monitor thread error: {str(e)}", "red")
            time.sleep(5)

# ============================================
# 🚀 PARALLEL PROCESSING CORE (Simplified)
# ============================================

def process_trading_idea_parallel(idea: str, thread_id: int) -> dict:
    """Process a single trading idea with full pipeline"""
    try:
        update_date_folders()
        
        thread_print(f"🚀 Starting processing", thread_id, attrs=['bold'])

        processed_idea = extract_content_from_url(idea, thread_id)

        # Phase 1: Research
        strategy, strategy_name = research_strategy(processed_idea, thread_id)

        if not strategy:
            thread_print("❌ Research failed", thread_id, "red")
            return {"success": False, "error": "Research failed", "thread_id": thread_id}

        log_processed_idea(idea, strategy_name, thread_id)

        # Phase 2: Backtest
        backtest = create_backtest(strategy, strategy_name, thread_id)

        if not backtest:
            thread_print("❌ Backtest failed", thread_id, "red")
            return {"success": False, "error": "Backtest failed", "thread_id": thread_id}

        # Phase 3: Package Check
        package_checked = package_check(backtest, strategy_name, thread_id)

        if not package_checked:
            thread_print("❌ Package check failed", thread_id, "red")
            return {"success": False, "error": "Package check failed", "thread_id": thread_id}

        return {
            "success": True,
            "thread_id": thread_id,
            "strategy_name": strategy_name,
            "return": 0
        }

    except Exception as e:
        thread_print(f"❌ FATAL ERROR: {str(e)}", thread_id, "red", attrs=['bold'])
        return {"success": False, "error": str(e), "thread_id": thread_id}

# ============================================
# 🌍 MAIN ENTRY POINT
# ============================================

def main(run_name=None):
    """Main orchestrator - CONTINUOUS QUEUE MODE WITH V2 FEATURES"""
    
    cprint(f"\n{'='*70}", "cyan", attrs=['bold'])
    cprint(f"🌟 Moon Dev's RBI AI v3.0 PARALLEL + HPC LLM 🚀", "cyan", attrs=['bold'])
    cprint(f"{'='*70}", "cyan", attrs=['bold'])

    cprint(f"\n📅 Date: {CURRENT_DATE}", "magenta")
    cprint(f"🎯 Target Return: {TARGET_RETURN}%", "green", attrs=['bold'])
    cprint(f"🔀 Max Parallel Threads: {MAX_PARALLEL_THREADS}", "yellow", attrs=['bold'])
    cprint(f"🐍 Conda env: {CONDA_ENV}", "cyan")
    cprint(f"🌙 LLM: {'LOCAL HPC' if USE_LOCAL_HPC_LLM else 'DEEPSEEK'}", "magenta", attrs=['bold'])
    if run_name:
        cprint(f"📁 Run Name: {run_name}\n", "green", attrs=['bold'])

    cprint(f"\n{'='*70}", "white", attrs=['bold'])
    cprint(f"🔄 MULTI-SOURCE STRATEGY READING (Priority Order):", "cyan", attrs=['bold'])
    cprint(f"{'='*70}", "white", attrs=['bold'])
    cprint(f"1️⃣  websearch_local/final_strategies/ (websearch_agent_v2.py)", "yellow")
    cprint(f"2️⃣  websearch_research/final_strategies/ (other agents)", "yellow")
    cprint(f"3️⃣  ideas.txt (manual input)", "yellow")
    cprint(f"{'='*70}\n", "white", attrs=['bold'])

    cprint(f"\n🔄 CONTINUOUS QUEUE MODE ACTIVATED", "cyan", attrs=['bold'])
    cprint(f"⏰ Monitoring strategy sources every 5 seconds", "yellow")
    cprint(f"🧵 {MAX_PARALLEL_THREADS} worker threads ready\n", "yellow")

    idea_queue = Queue()
    queued_ideas = set()
    queued_lock = Lock()
    stats = {
        'completed': 0,
        'successful': 0,
        'failed': 0,
        'targets_hit': 0,
        'active': 0
    }
    stop_flag = {'stop': False}

    monitor = Thread(target=idea_monitor_thread, args=(idea_queue, queued_ideas, queued_lock, stop_flag), daemon=True)
    monitor.start()
    cprint("✅ Idea monitor thread started", "green")

    try:
        while True:
            time.sleep(5)
            update_date_folders()

            with console_lock:
                if stats['active'] > 0 or not idea_queue.empty():
                    cprint(f"📊 Status: {stats['active']} active | {idea_queue.qsize()} queued | {stats['completed']} completed | {stats['targets_hit']} targets hit", "cyan")
                else:
                    cprint(f"💤 AI swarm waiting... ({stats['completed']} completed, {stats['targets_hit']} targets) - {datetime.now().strftime('%I:%M:%S %p')}", "yellow")

    except KeyboardInterrupt:
        cprint(f"\n\n🛑 Shutting down gracefully...", "yellow", attrs=['bold'])
        stop_flag['stop'] = True

        cprint(f"\n{'='*60}", "cyan", attrs=['bold'])
        cprint(f"📊 FINAL STATS", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan", attrs=['bold'])
        cprint(f"✅ Successful: {stats['successful']}", "green")
        cprint(f"🎯 Targets hit: {stats['targets_hit']}", "green", attrs=['bold'])
        cprint(f"❌ Failed: {stats['failed']}", "red")
        cprint(f"📊 Total completed: {stats['completed']}", "cyan")
        cprint(f"{'='*60}\n", "cyan", attrs=['bold'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moon Dev's RBI Agent v2 - HPC LLM + Multi-Source")
    parser.add_argument('--run-name', type=str, help='Run name for folder organization')
    args = parser.parse_args()

    main(run_name=args.run_name)
