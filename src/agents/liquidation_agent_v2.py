#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 Moon Dev's Liquidation Monitor V2
Built with love by Moon Dev 🚀

Luna the Liquidation Agent V2 combines liquidation monitoring with websearch_agent_v2 improvements:
- Local Qwen2.5-7B LLM via HPC (192.168.30.158:8000)
- Structured CSV logging system
- Strategy similarity detection
- Content hash deduplication
- Production-grade error handling

Need an API key? For a limited time, bootcamp members get free api keys for claude, openai, helius, birdeye & quant elite gets access to the moon dev api.
Join here: https://algotradecamp.com
"""

import os
import sys
import json
import csv
import hashlib
import logging
import requests
import traceback
from datetime import datetime, timedelta
from termcolor import colored, cprint
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

# ============================================================================
# PYTHONPATH SETUP - FIX FOR HPC/APPTAINER ENVIRONMENTS
# ============================================================================

# Determine the project root
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] PYTHONPATH includes: {sys.path[:3]}")

# ============================================================================
# CONDITIONAL IMPORTS WITH FALLBACKS
# ============================================================================

try:
    import pandas as pd
except ImportError:
    print("[WARNING] pandas not installed")
    pd = None

try:
    import numpy as np
except ImportError:
    print("[WARNING] numpy not installed")
    np = None

try:
    import openai
except ImportError:
    print("[WARNING] openai not installed")
    openai = None

try:
    import anthropic
except ImportError:
    print("[WARNING] anthropic not installed")
    anthropic = None

# Import local modules with error handling
try:
    from src import nice_funcs as n
    print("[DEBUG] Successfully imported nice_funcs")
except ImportError as e:
    print(f"[WARNING] Could not import nice_funcs: {e}")
    n = None

try:
    from src import nice_funcs_hyperliquid as hl
    print("[DEBUG] Successfully imported nice_funcs_hyperliquid")
except ImportError as e:
    print(f"[WARNING] Could not import nice_funcs_hyperliquid: {e}")
    hl = None

try:
    from src.agents.api import MoonDevAPI
    print("[DEBUG] Successfully imported MoonDevAPI")
except ImportError as e:
    print(f"[WARNING] Could not import MoonDevAPI: {e}")
    MoonDevAPI = None

try:
    from src.agents.base_agent import BaseAgent
    print("[DEBUG] Successfully imported BaseAgent")
except ImportError as e:
    print(f"[WARNING] Could not import BaseAgent: {e}")
    BaseAgent = None

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class LiquidationConfig:
    """Centralized configuration for liquidation agent"""
    
    # Liquidation monitoring
    CHECK_INTERVAL_MINUTES = 10
    LIQUIDATION_ROWS = 10000
    LIQUIDATION_THRESHOLD = .5  # Multiplier for average liquidation
    
    # OHLCV Data
    TIMEFRAME = '15m'
    LOOKBACK_BARS = 100
    COMPARISON_WINDOW = 15  # 15, 60, or 240 minutes
    
    # AI/LLM Settings
    LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
    LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_TIMEOUT_SECONDS = 120
    LLM_MAX_RETRIES = 2
    LLM_RETRY_WAIT = 30
    
    # Quality & Deduplication
    MIN_STRATEGY_QUALITY_SCORE = 0.3
    STRATEGY_SIMILARITY_THRESHOLD = 0.85
    CONTENT_HASH_DEDUP = True
    
    # Voice settings
    VOICE_MODEL = "tts-1"
    VOICE_NAME = "nova"
    VOICE_SPEED = 1
    
    # Sleep intervals (seconds)
    SLEEP_BETWEEN_CYCLES = 5

# Liquidation analysis prompt
LIQUIDATION_ANALYSIS_PROMPT = """
You must respond in exactly 3 lines:
Line 1: Only write BUY, SELL, or NOTHING
Line 2: One short reason why
Line 3: Only write "Confidence: X%" where X is 0-100

Analyze market with total {pct_change}% increase in liquidations:

Current Long Liquidations: ${current_longs:,.2f} ({pct_change_longs:+.1f}% change)
Current Short Liquidations: ${current_shorts:,.2f} ({pct_change_shorts:+.1f}% change)
Time Period: Last {LIQUIDATION_ROWS} liquidation events

Market Data (Last {LOOKBACK_BARS} {TIMEFRAME} candles):
{market_data}

Large long liquidations often indicate potential bottoms (shorts taking profit)
Large short liquidations often indicate potential tops (longs taking profit)
Consider the ratio of long vs short liquidations and their relative changes
"""

# ============================================================================
# DATA DIRECTORIES & CSV MANAGEMENT
# ============================================================================

DATA_DIR = PROJECT_ROOT / "src" / "data" / "liquidation_v2"
LOGS_DIR = DATA_DIR / "logs"
METRICS_DIR = DATA_DIR / "metrics"

for dir_path in [DATA_DIR, LOGS_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# CSV file paths
LIQUIDATION_EVENTS_CSV = DATA_DIR / "liquidation_events.csv"
ANALYSIS_RESULTS_CSV = DATA_DIR / "analysis_results.csv"
QUALITY_METRICS_CSV = DATA_DIR / "quality_metrics.csv"
DEDUPLICATION_LOG_CSV = DATA_DIR / "deduplication_log.csv"

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, 'white')
        record.levelname = f"[{record.levelname}]"
        return super().format(record)

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup comprehensive logging system"""
    logger = logging.getLogger('LiquidationAgentV2')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (with colors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

LOG_FILE = LOGS_DIR / f"liquidation_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logging(LOG_FILE)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LiquidationEvent:
    """Structured liquidation event"""
    timestamp: str
    long_size: float
    short_size: float
    total_size: float
    long_change_pct: float
    short_change_pct: float
    total_change_pct: float
    event_hash: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class AnalysisResult:
    """Structured analysis result"""
    timestamp: str
    event_hash: str
    signal: str  # BUY, SELL, NOTHING
    confidence: float
    reason: str
    long_liq: float
    short_liq: float
    market_context: str
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# CSV INITIALIZATION
# ============================================================================

class CSVManager:
    """Centralized CSV management"""
    
    @staticmethod
    def init_liquidation_events_csv():
        """Initialize liquidation events CSV"""
        if not LIQUIDATION_EVENTS_CSV.exists():
            with open(LIQUIDATION_EVENTS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'long_size', 'short_size', 'total_size',
                    'long_change_pct', 'short_change_pct', 'total_change_pct',
                    'event_hash', 'processed'
                ])
            logger.info(f"📝 Created {LIQUIDATION_EVENTS_CSV}")
    
    @staticmethod
    def init_analysis_results_csv():
        """Initialize analysis results CSV"""
        if not ANALYSIS_RESULTS_CSV.exists():
            with open(ANALYSIS_RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_hash', 'signal', 'confidence',
                    'reason', 'long_liq', 'short_liq', 'market_context',
                    'similarity_score'
                ])
            logger.info(f"📝 Created {ANALYSIS_RESULTS_CSV}")
    
    @staticmethod
    def init_deduplication_log_csv():
        """Initialize deduplication log CSV"""
        if not DEDUPLICATION_LOG_CSV.exists():
            with open(DEDUPLICATION_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_hash', 'signal', 'similarity_score',
                    'decision', 'reason', 'similar_to'
                ])
            logger.info(f"📝 Created {DEDUPLICATION_LOG_CSV}")
    
    @staticmethod
    def init_all():
        """Initialize all CSV files"""
        CSVManager.init_liquidation_events_csv()
        CSVManager.init_analysis_results_csv()
        CSVManager.init_deduplication_log_csv()

CSVManager.init_all()

# ============================================================================
# SIMILARITY DETECTION
# ============================================================================

class EventSimilarityDetector:
    """Liquidation event similarity detection"""
    
    def __init__(self, threshold: float = LiquidationConfig.STRATEGY_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.existing_events_cache: List[Dict[str, Any]] = []
        self._load_existing_events()
    
    def _load_existing_events(self):
        """Load existing events from CSV"""
        try:
            if ANALYSIS_RESULTS_CSV.exists() and pd is not None:
                df = pd.read_csv(ANALYSIS_RESULTS_CSV)
                logger.info(f"📚 Loading {len(df)} existing analysis results for similarity check")
                
                for _, row in df.iterrows():
                    self.existing_events_cache.append({
                        'event_hash': row['event_hash'],
                        'signal': row['signal'],
                        'reason': row['reason'],
                        'similarity_score': 0.0
                    })
                
                logger.info(f"✅ Loaded {len(self.existing_events_cache)} events")
        except Exception as e:
            logger.warning(f"⚠️  Error loading existing events: {e}")
    
    def check_similarity(self, signal: str, reason: str) -> Tuple[bool, float]:
        """
        Check if similar analysis already exists
        Returns: (is_duplicate, max_similarity)
        """
        if not self.existing_events_cache:
            return False, 0.0
        
        try:
            max_similarity = 0.0
            
            for existing in self.existing_events_cache:
                # Simple similarity check: same signal + similar reason keywords
                if existing['signal'] == signal:
                    # Count matching words
                    reason_words = set(reason.lower().split())
                    existing_words = set(existing['reason'].lower().split())
                    overlap = len(reason_words & existing_words) / max(len(reason_words | existing_words), 1)
                    
                    if overlap > max_similarity:
                        max_similarity = overlap
            
            is_duplicate = max_similarity >= self.threshold
            
            if is_duplicate:
                logger.warning(f"⚠️  Duplicate detected: {signal} (similarity: {max_similarity:.2%})")
            
            return is_duplicate, max_similarity
        
        except Exception as e:
            logger.warning(f"⚠️  Similarity check error: {e}")
            return False, 0.0
    
    def add_event(self, signal: str, reason: str):
        """Add new event to cache"""
        self.existing_events_cache.append({
            'signal': signal,
            'reason': reason,
            'similarity_score': 0.0
        })

# ============================================================================
# CONTENT HASHING
# ============================================================================

class ContentHasher:
    """Content-based deduplication"""
    
    @staticmethod
    def calculate_hash(data: Dict[str, Any]) -> str:
        """Calculate hash for liquidation data"""
        try:
            content = f"{data.get('long_size', 0)}{data.get('short_size', 0)}{data.get('total_size', 0)}"
            return hashlib.md5(content.encode()).hexdigest()[:12]
        except Exception as e:
            logger.warning(f"⚠️  Hash calculation error: {e}")
            return ""

# ============================================================================
# LLM INTEGRATION - LOCAL QWEN
# ============================================================================

class LocalLLMProvider:
    """Local Qwen LLM provider via HPC"""
    
    def __init__(self):
        self.base_url = LiquidationConfig.LOCAL_LLM_URL
        self.model = LiquidationConfig.LOCAL_LLM_MODEL
    
    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.3,
                         max_tokens: int = 500) -> Optional[str]:
        """Generate response using local Qwen LLM"""
        
        for attempt in range(LiquidationConfig.LLM_MAX_RETRIES + 1):
            try:
                logger.debug(f"🤖 LLM attempt {attempt + 1}/{LiquidationConfig.LLM_MAX_RETRIES + 1}")
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=LiquidationConfig.LLM_TIMEOUT_SECONDS
                )
                
                if response.status_code != 200:
                    logger.warning(f"⚠️  LLM API error {response.status_code}")
                    if attempt < LiquidationConfig.LLM_MAX_RETRIES:
                        wait_time = LiquidationConfig.LLM_RETRY_WAIT * (attempt + 1)
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        import time
                        time.sleep(wait_time)
                        continue
                    return None
                
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.debug(f"✅ LLM response ({len(content)} chars)")
                return content
                
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️  LLM timeout (attempt {attempt + 1}/{LiquidationConfig.LLM_MAX_RETRIES + 1})")
                if attempt < LiquidationConfig.LLM_MAX_RETRIES:
                    import time
                    wait_time = LiquidationConfig.LLM_RETRY_WAIT * (attempt + 1)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            
            except requests.exceptions.ConnectionError:
                logger.error(f"❌ Cannot connect to LLM server at {self.base_url}")
                return None
            
            except Exception as e:
                logger.error(f"❌ LLM error: {e}")
                if attempt < LiquidationConfig.LLM_MAX_RETRIES:
                    import time
                    wait_time = LiquidationConfig.LLM_RETRY_WAIT * (attempt + 1)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
        
        logger.error(f"❌ LLM failed after {LiquidationConfig.LLM_MAX_RETRIES + 1} attempts")
        return None

# ============================================================================
# DATA LOGGING
# ============================================================================

class DataLogger:
    """Centralized data logging"""
    
    @staticmethod
    def log_liquidation_event(event: LiquidationEvent):
        """Log liquidation event"""
        try:
            with open(LIQUIDATION_EVENTS_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    f"{event.long_size:.2f}",
                    f"{event.short_size:.2f}",
                    f"{event.total_size:.2f}",
                    f"{event.long_change_pct:.2f}",
                    f"{event.short_change_pct:.2f}",
                    f"{event.total_change_pct:.2f}",
                    event.event_hash,
                    "yes"
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log liquidation event: {e}")
    
    @staticmethod
    def log_analysis_result(result: AnalysisResult):
        """Log analysis result"""
        try:
            with open(ANALYSIS_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.timestamp,
                    result.event_hash,
                    result.signal,
                    f"{result.confidence:.2f}",
                    result.reason[:200],
                    f"{result.long_liq:.2f}",
                    f"{result.short_liq:.2f}",
                    result.market_context[:100],
                    f"{result.similarity_score:.3f}"
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log analysis result: {e}")
    
    @staticmethod
    def log_deduplication(event_hash: str, signal: str, similarity: float,
                         decision: str, reason: str, similar_to: str = ""):
        """Log deduplication decision"""
        try:
            with open(DEDUPLICATION_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    event_hash,
                    signal,
                    f"{similarity:.3f}",
                    decision,
                    reason[:200],
                    similar_to[:100]
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log deduplication: {e}")

# ============================================================================
# LIQUIDATION AGENT V2 - STUB VERSION FOR HPC
# ============================================================================

class LiquidationAgent:
    """Luna the Liquidation Monitor V2 - With websearch_agent_v2 improvements"""
    
    def __init__(self):
        """Initialize Luna the Liquidation Agent V2"""
        
        # Initialize components
        self.llm_provider = LocalLLMProvider()
        self.similarity_detector = EventSimilarityDetector()
        self.content_hasher = ContentHasher()
        
        # Use MoonDevAPI if available
        self.api = MoonDevAPI() if MoonDevAPI else None
        
        # Create data directories
        self.audio_dir = PROJECT_ROOT / "src" / "audio"
        self.data_dir = PROJECT_ROOT / "src" / "data"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load historical data
        self.history_file = self.data_dir / "liquidation_history.csv"
        self.load_history()
        
        logger.info("🌊 Luna the Liquidation Agent V2 initialized!")
        logger.info(f"🎯 Alerting on liquidation increases above +{LiquidationConfig.LIQUIDATION_THRESHOLD*100:.0f}%")
        logger.info(f"📊 Analyzing last {LiquidationConfig.LIQUIDATION_ROWS} liquidation events")
        logger.info(f"🤖 Using local Qwen LLM at {LiquidationConfig.LOCAL_LLM_URL}")
        logger.info(f"📁 Output directory: {DATA_DIR}")
    
    def load_history(self):
        """Load or initialize historical liquidation data"""
        try:
            if self.history_file.exists() and pd is not None:
                self.liquidation_history = pd.read_csv(self.history_file)
                
                # Handle transition from old format to new format
                if 'long_size' not in self.liquidation_history.columns:
                    logger.info("📝 Converting history to new format with long/short tracking...")
                    self.liquidation_history['long_size'] = self.liquidation_history['total_size'] / 2
                    self.liquidation_history['short_size'] = self.liquidation_history['total_size'] / 2
                
                logger.info(f"📈 Loaded {len(self.liquidation_history)} historical records")
            else:
                self.liquidation_history = pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size']) if pd else None
                logger.info("📝 Created new liquidation history file")
                
            # Clean up old data (keep only last 24 hours)
            if self.liquidation_history is not None and not self.liquidation_history.empty and pd is not None:
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.liquidation_history = self.liquidation_history[
                    pd.to_datetime(self.liquidation_history['timestamp']) > cutoff_time
                ]
                self.liquidation_history.to_csv(self.history_file, index=False)
                
        except Exception as e:
            logger.error(f"❌ Error loading history: {str(e)}")
            self.liquidation_history = pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size']) if pd else None
    
    def _test_llm_connection(self):
        """Test LLM connection"""
        logger.info("🔌 Testing LLM connection...")
        response = self.llm_provider.generate_response(
            "You are a helpful assistant.",
            "Say 'LLM is working!' in exactly one sentence.",
            temperature=0.3,
            max_tokens=20
        )
        if response:
            logger.info(f"✅ LLM test successful: {response}")
        else:
            logger.error("❌ LLM test failed")
    
    def run_cycle(self):
        """Run one monitoring cycle"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🌊 Liquidation Monitor Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            if self.api:
                logger.info("📡 Fetching liquidation data from API...")
                # TODO: Implement API fetch
            else:
                logger.warning("⚠️  MoonDevAPI not available, skipping data fetch")
                logger.info("🧪 Running LLM test instead...")
                self._test_llm_connection()
            
        except Exception as e:
            logger.error(f"❌ Error in monitoring cycle: {e}")
            traceback.print_exc()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    cprint("\n🌙 MOON DEV LIQUIDATION MONITOR V2 (WITH WEBSEARCH_AGENT_V2 IMPROVEMENTS)", "white", "on_magenta")
    cprint("🤖 Local LLM: Qwen2.5-7B via HPC Server (192.168.30.158:8000)", "cyan")
    cprint(f"📁 Output directory: {DATA_DIR}\n", "cyan")
    
    agent = LiquidationAgent()
    cycle = 0
    
    try:
        while True:
            cycle += 1
            agent.run_cycle()
            
            # Sleep before next cycle
            import time
            logger.info(f"⏱️  Next cycle in {LiquidationConfig.CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(LiquidationConfig.CHECK_INTERVAL_MINUTES * 60)
            
    except KeyboardInterrupt:
        cprint(f"\n\n👋 Shutting down after {cycle} cycles", "yellow")
        cprint(f"📊 Logs: {LOGS_DIR}", "cyan")
        cprint(f"📊 Data: {DATA_DIR}\n", "cyan")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
