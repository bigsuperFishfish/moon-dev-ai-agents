# ... (keep all content before line 60 EXACTLY the same)

# Load environment variables FIRST
load_dotenv()
print("✅ Environment variables loaded")

# Add config values directly to avoid import issues
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 16000

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

# Import model factory ONLY if not using HPC LLM
sys.path.append(str(Path(__file__).parent.parent.parent))

if not USE_LOCAL_HPC_LLM:
    try:
        from src.models import model_factory
        print("✅ Successfully imported model_factory")
    except ImportError as e:
        print(f"⚠️ Could not import model_factory: {e}")
        print("❌ Cannot use cloud API fallback without model_factory")
        sys.exit(1)
else:
    cprint("⏭️  Skipping model_factory import (HPC-only mode)", "cyan")
    model_factory = None

# ... (rest of the file remains EXACTLY the same)
