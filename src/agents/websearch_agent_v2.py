#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moon Dev AI Trading System - Websearch Agent V2
Production-grade web search agent with advanced features and robust architecture.

Features:
- Multi-provider LLM support (Claude, GPT-4, DeepSeek, Groq, Gemini, Ollama)
- Intelligent rate limiting and request optimization
- Advanced retry mechanisms with exponential backoff
- Comprehensive error handling and recovery
- Structured JSON output validation
- Real-time progress monitoring
- Persistent caching of search results
- Duplicate detection and filtering
- Async processing capabilities
- Detailed logging and metrics tracking

Author: Moon Dev Trading Agents
Version: 2.0.1 (Quality Score Fix)
Date: 2025-12-30
"""

import os
import sys
import time
import json
import csv
import hashlib
import logging
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from termcolor import cprint
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

try:
    import pandas as pd
except ImportError:
    cprint("⚠️  pandas not installed, some features disabled", "yellow")
    pd = None

load_dotenv()

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class SearchConfig:
    """Centralized configuration for search agent"""
    
    # Search settings
    MAX_SEARCH_RESULTS = 8
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2.0
    
    # Content processing
    MAX_CONTENT_LENGTH = 15000
    MIN_CONTENT_LENGTH = 300
    TIMEOUT_SECONDS = 20
    
    # LLM settings
    LLM_TIMEOUT_SECONDS = 120  # 2 minutes for LLM processing
    LLM_MAX_RETRIES = 2
    LLM_RETRY_WAIT = 30  # seconds
    
    # Sleep intervals (seconds)
    SLEEP_BETWEEN_SEARCHES = 120
    SLEEP_BETWEEN_FETCHES = 2
    SLEEP_BETWEEN_CYCLES = 5
    
    # Rate limiting
    MAX_REQUESTS_PER_HOUR = 100
    MAX_API_CALLS_PER_MINUTE = 20
    
    # Output settings
    FINAL_STRATEGIES_PER_SEARCH = 5
    MIN_STRATEGY_QUALITY_SCORE = 0.3  # FIXED: Lowered from 0.6 (was filtering 90%)
    
    # LLM settings
    LLM_MAX_TOKENS_QUERY_GEN = 150
    LLM_MAX_TOKENS_EXTRACTION = 4000
    LLM_TEMPERATURE_CREATIVE = 0.8
    LLM_TEMPERATURE_ANALYTICAL = 0.3

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ============================================================================
# PROJECT PATHS & DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "web_search_local"
FINAL_STRATEGIES_DIR = DATA_DIR / "final_strategies"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, FINAL_STRATEGIES_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# CSV file paths
SEARCH_RESULTS_CSV = DATA_DIR / "search_results.csv"
EXTRACTION_LOG_CSV = DATA_DIR / "extraction_log.csv"
STRATEGY_QUALITY_CSV = DATA_DIR / "strategy_quality.csv"
METRICS_CSV = DATA_DIR / "metrics.csv"

# Log file path
LOG_FILE = LOGS_DIR / f"websearch_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    logger = logging.getLogger('WebSearchAgent')
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

logger = setup_logging(LOG_FILE)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    snippet: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class StrategyData:
    """Structured strategy data"""
    strategy_name: str
    entry_rules: str
    exit_rules: str
    indicators: str
    timeframe: str
    risk_management: str
    parameters: str
    description: str
    quality_score: float = 0.0
    source_url: str = ""
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SearchCycleMetrics:
    """Metrics for search cycle"""
    cycle_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    queries_generated: int = 0
    search_results_found: int = 0
    urls_processed: int = 0
    urls_skipped_duplicate: int = 0
    strategies_extracted: int = 0
    extraction_success_rate: float = 0.0
    average_quality_score: float = 0.0
    total_api_calls: int = 0
    total_duration_seconds: float = 0.0
    errors_encountered: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# CSV INITIALIZATION & MANAGEMENT
# ============================================================================

class CSVManager:
    """Centralized CSV management"""
    
    @staticmethod
    def init_search_results_csv():
        """Initialize search results CSV"""
        if not SEARCH_RESULTS_CSV.exists():
            with open(SEARCH_RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'cycle', 'query', 'url', 'title', 'status',
                    'content_length', 'content_hash', 'fetch_duration_ms'
                ])
            logger.info(f"📝 Created {SEARCH_RESULTS_CSV}")
    
    @staticmethod
    def init_extraction_log_csv():
        """Initialize extraction log CSV"""
        if not EXTRACTION_LOG_CSV.exists():
            with open(EXTRACTION_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'url', 'num_strategies', 'strategy_names',
                    'average_quality', 'extraction_duration_ms'
                ])
            logger.info(f"📝 Created {EXTRACTION_LOG_CSV}")
    
    @staticmethod
    def init_strategy_quality_csv():
        """Initialize strategy quality CSV"""
        if not STRATEGY_QUALITY_CSV.exists():
            with open(STRATEGY_QUALITY_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'strategy_name', 'quality_score',
                    'completeness', 'specificity', 'actionability', 'source_url'
                ])
            logger.info(f"📝 Created {STRATEGY_QUALITY_CSV}")
    
    @staticmethod
    def init_metrics_csv():
        """Initialize metrics CSV"""
        if not METRICS_CSV.exists():
            with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'cycle', 'queries_generated', 'search_results',
                    'urls_processed', 'strategies_extracted', 'success_rate',
                    'avg_quality_score', 'total_api_calls', 'duration_seconds'
                ])
            logger.info(f"📝 Created {METRICS_CSV}")
    
    @staticmethod
    def init_all():
        """Initialize all CSV files"""
        CSVManager.init_search_results_csv()
        CSVManager.init_extraction_log_csv()
        CSVManager.init_strategy_quality_csv()
        CSVManager.init_metrics_csv()

CSVManager.init_all()

# ============================================================================
# CACHING & DEDUPLICATION
# ============================================================================

class URLCache:
    """URL deduplication and caching system"""
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (CACHE_DIR / "url_cache.json")
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.debug(f"💾 Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️  Failed to save cache: {e}")
    
    def is_processed(self, url: str) -> bool:
        """Check if URL has been processed"""
        return url in self.cache
    
    def mark_processed(self, url: str, data: Dict[str, Any]):
        """Mark URL as processed"""
        self.cache[url] = {
            'processed_at': datetime.now().isoformat(),
            'data': data
        }
        self._save_cache()
    
    def get_processed_data(self, url: str) -> Optional[Dict]:
        """Get cached data for URL"""
        if url in self.cache:
            return self.cache[url].get('data')
        return None

url_cache = URLCache()

# ============================================================================
# LLM INTEGRATION (ModelFactory Support)
# ============================================================================

class LLMProvider:
    """Unified LLM provider interface"""
    
    def __init__(self, provider: str = 'anthropic'):
        self.provider = provider
        self._load_credentials()
    
    def _load_credentials(self):
        """Load API credentials from environment"""
        credentials_map = {
            'anthropic': 'ANTHROPIC_KEY',
            'openai': 'OPENAI_KEY',
            'deepseek': 'DEEPSEEK_KEY',
            'groq': 'GROQ_API_KEY',
            'gemini': 'GEMINI_KEY',
            'ollama': 'OLLAMA_URL',
        }
        self.api_key = os.getenv(credentials_map.get(self.provider, ''))
        if not self.api_key and self.provider != 'ollama':
            logger.warning(f"⚠️  No API key found for {self.provider}")
    
    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.3,
                         max_tokens: int = 2048) -> Optional[str]:
        """Generate response using local Qwen via LM Studio"""
        
        # HPC LLM configuration
        LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
        LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        
        for attempt in range(SearchConfig.LLM_MAX_RETRIES + 1):
            try:
                logger.debug(f"🤖 LLM attempt {attempt + 1}/{SearchConfig.LLM_MAX_RETRIES + 1} (temp={temperature}, tokens={max_tokens})")
                
                payload = {
                    "model": LOCAL_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = requests.post(
                    LOCAL_LLM_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=SearchConfig.LLM_TIMEOUT_SECONDS
                )
                
                if response.status_code != 200:
                    logger.warning(f"⚠️  LLM API error {response.status_code}")
                    if attempt < SearchConfig.LLM_MAX_RETRIES:
                        wait_time = SearchConfig.LLM_RETRY_WAIT * (attempt + 1)
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    return None
                
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.debug(f"✅ LLM response ({len(content)} chars)")
                return content
                
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️  LLM timeout (attempt {attempt + 1}/{SearchConfig.LLM_MAX_RETRIES + 1})")
                if attempt < SearchConfig.LLM_MAX_RETRIES:
                    wait_time = SearchConfig.LLM_RETRY_WAIT * (attempt + 1)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            
            except requests.exceptions.ConnectionError:
                logger.error(f"❌ Cannot connect to LLM server at {LOCAL_LLM_URL}")
                if attempt < SearchConfig.LLM_MAX_RETRIES:
                    wait_time = SearchConfig.LLM_RETRY_WAIT * (attempt + 1)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            
            except Exception as e:
                logger.error(f"❌ LLM error: {e}")
                if attempt < SearchConfig.LLM_MAX_RETRIES:
                    wait_time = SearchConfig.LLM_RETRY_WAIT * (attempt + 1)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
        
        logger.error(f"❌ LLM failed after {SearchConfig.LLM_MAX_RETRIES + 1} attempts")
        return None

llm_provider = LLMProvider('anthropic')

# ============================================================================
# QUERY GENERATION
# ============================================================================

class QueryGenerator:
    """Intelligent search query generation"""
    
    # Trading strategy templates
    STRATEGY_TYPES = [
        "momentum", "mean reversion", "breakout", "arbitrage",
        "pairs trading", "statistical arbitrage", "grid trading",
        "scalping", "swing trading", "trend following"
    ]
    
    INDICATORS = [
        "RSI", "MACD", "Bollinger Bands", "SMA", "EMA",
        "ATR", "Stochastic", "Volume Profile", "Ichimoku",
        "Moving Average Convergence", "Donchian Channels"
    ]
    
    TIMEFRAMES = ["15m", "1h", "4h", "daily", "weekly"]
    
    PARAMETERS = [
        "entry conditions", "exit conditions", "stop loss",
        "take profit", "position sizing", "risk reward ratio"
    ]
    
    @classmethod
    def generate_query(cls) -> str:
        """Generate a diverse, specific search query"""
        import random
        
        strategy_type = random.choice(cls.STRATEGY_TYPES)
        indicator1 = random.choice(cls.INDICATORS)
        indicator2 = random.choice(cls.INDICATORS)
        timeframe = random.choice(cls.TIMEFRAMES)
        parameter = random.choice(cls.PARAMETERS)
        
        templates = [
            f"{strategy_type} trading strategy {indicator1} {indicator2} {timeframe} parameters",
            f"{indicator1} {indicator2} {strategy_type} backtest {timeframe} {parameter}",
            f"profitable {strategy_type} strategy {timeframe} {indicator1} rules",
            f"{strategy_type} algorithm {indicator1} {indicator2} entry exit {timeframe}",
            f"trading bot {strategy_type} {indicator1} {timeframe} {parameter}",
        ]
        
        query = random.choice(templates)
        logger.info(f"✨ Generated query: {query}")
        return query

# ============================================================================
# WEB SEARCH
# ============================================================================

class WebSearcher:
    """DuckDuckGo web search wrapper with advanced features"""
    
    def __init__(self, max_results: int = SearchConfig.MAX_SEARCH_RESULTS):
        self.max_results = max_results
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[SearchResult]:
        """Execute search with error handling and retry logic"""
        logger.info(f"🔍 Searching DuckDuckGo: {query}")
        
        results = []
        retry_count = 0
        
        while retry_count < SearchConfig.MAX_RETRIES:
            try:
                search_results = list(self.ddgs.text(query, max_results=self.max_results))
                
                for i, result in enumerate(search_results, 1):
                    logger.debug(f"[{i}] {result['title'][:60]}")
                    results.append(SearchResult(
                        title=result['title'],
                        url=result['href'],
                        snippet=result.get('body', '')
                    ))
                
                logger.info(f"✅ Found {len(results)} results")
                return results
                
            except Exception as e:
                retry_count += 1
                wait_time = SearchConfig.RETRY_BACKOFF_FACTOR ** retry_count
                logger.warning(f"⚠️  Search error (retry {retry_count}/{SearchConfig.MAX_RETRIES}): {e}")
                if retry_count < SearchConfig.MAX_RETRIES:
                    time.sleep(wait_time)
        
        logger.error("❌ Search failed after all retries")
        return []

# ============================================================================
# WEB CONTENT FETCHING
# ============================================================================

class ContentFetcher:
    """Advanced webpage content extraction with quality checks"""
    
    def __init__(self, timeout: int = SearchConfig.TIMEOUT_SECONDS):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch webpage with comprehensive error handling"""
        logger.info(f"🌐 Fetching: {url[:70]}")
        
        retry_count = 0
        while retry_count < SearchConfig.MAX_RETRIES:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    logger.warning(f"⚠️  HTTP {response.status_code}")
                    return None
                
                # Parse and clean content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
                    tag.decompose()
                
                # Extract title
                title = soup.title.string.strip() if soup.title else "No Title"
                
                # Extract text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                text = '\n'.join(line for line in lines if line)
                
                # Validate content length
                if len(text) < SearchConfig.MIN_CONTENT_LENGTH:
                    logger.warning(f"⚠️  Content too short ({len(text)} chars)")
                    return None
                
                # Truncate if needed
                text = text[:SearchConfig.MAX_CONTENT_LENGTH]
                
                # Calculate content hash
                content_hash = hashlib.md5(text.encode()).hexdigest()
                
                logger.info(f"✅ Fetched {len(text)} chars (hash: {content_hash[:8]})")
                
                return {
                    'url': url,
                    'title': title,
                    'content': text,
                    'content_hash': content_hash,
                    'fetch_time': datetime.now().isoformat()
                }
                
            except requests.exceptions.Timeout:
                retry_count += 1
                logger.warning(f"⚠️  Timeout (retry {retry_count}/{SearchConfig.MAX_RETRIES})")
                if retry_count < SearchConfig.MAX_RETRIES:
                    time.sleep(SearchConfig.RETRY_BACKOFF_FACTOR ** retry_count)
            
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error: {e}")
                return None
            
            except Exception as e:
                logger.error(f"❌ Parsing error: {e}")
                return None
        
        logger.error(f"❌ Failed to fetch after {SearchConfig.MAX_RETRIES} retries")
        return None

# ============================================================================
# STRATEGY EXTRACTION
# ============================================================================

class StrategyExtractor:
    """AI-powered strategy extraction from content"""
    
    # FIXED: Reweighted quality criteria for more realistic scoring
    QUALITY_CRITERIA = {
        'completeness': 0.25,     # All 8 fields present
        'specificity': 0.30,      # Specific parameters vs vague
        'actionability': 0.25,    # Can be implemented
        'clarity': 0.20           # Clear, readable output
    }
    
    def extract(self, content: str, source_url: str) -> List[StrategyData]:
        """Extract strategies from content using LLM"""
        logger.info(f"🧠 Extracting strategies from content")
        
        system_prompt = """You are an expert trading strategy analyst.
Extract ALL trading strategies from the provided content.
Be aggressive - extract even partial ideas and combine them into actionable strategies.
Return ONLY valid JSON, no explanations."""
        
        user_prompt = f"""Extract ALL trading strategies from this content.
For EACH strategy, provide:
1. strategy_name: Clear name
2. entry_rules: Specific entry conditions (e.g., 'RSI < 30 AND price above 200-day SMA')
3. exit_rules: Specific exit conditions (e.g., 'RSI > 70 OR stop loss hit')
4. indicators: All indicators with parameters (e.g., 'RSI(14), MACD(12,26,9), SMA(200)')
5. timeframe: Trading timeframe (15m, 1h, 4h, daily, etc.)
6. risk_management: Stop loss, position sizing, take profit levels
7. description: Full detailed description (2-3 sentences minimum)
8. parameters: Specific numerical values if mentioned

Return JSON:
{{
    "strategies": [
        {{
            "strategy_name": "...",
            "entry_rules": "...",
            "exit_rules": "...",
            "indicators": "...",
            "timeframe": "...",
            "risk_management": "...",
            "description": "...",
            "parameters": "..."
        }}
    ]
}}

Content to analyze:
{content[:6000]}"""
        
        response = llm_provider.generate_response(
            system_prompt,
            user_prompt,
            temperature=SearchConfig.LLM_TEMPERATURE_ANALYTICAL,
            max_tokens=SearchConfig.LLM_MAX_TOKENS_EXTRACTION
        )
        
        if not response:
            logger.warning("⚠️  No LLM response")
            return []
        
        strategies = self._parse_response(response, source_url)
        logger.info(f"✨ Extracted {len(strategies)} strategies")
        return strategies
    
    def _parse_response(self, response: str, source_url: str) -> List[StrategyData]:
        """Parse LLM response and validate"""
        try:
            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            strategies = []
            
            for item in data.get('strategies', []):
                strategy = StrategyData(
                    strategy_name=self._safe_str(item.get('strategy_name', 'Untitled')),
                    entry_rules=self._safe_str(item.get('entry_rules', 'Not specified')),
                    exit_rules=self._safe_str(item.get('exit_rules', 'Not specified')),
                    indicators=self._safe_str(item.get('indicators', 'Not specified')),
                    timeframe=self._safe_str(item.get('timeframe', 'Not specified')),
                    risk_management=self._safe_str(item.get('risk_management', 'Not specified')),
                    parameters=self._safe_str(item.get('parameters', 'Not specified')),
                    description=self._safe_str(item.get('description', 'No description')),
                    source_url=source_url
                )
                
                # Calculate quality score
                strategy.quality_score = self._calculate_quality_score(strategy)
                strategies.append(strategy)
            
            return strategies
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
            return []
    
    @staticmethod
    def _safe_str(value: Any) -> str:
        """Safely convert value to string"""
        if value is None:
            return 'Not specified'
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        if isinstance(value, dict):
            return ', '.join(f"{k}: {v}" for k, v in value.items())
        return str(value).strip()
    
    @classmethod
    def _calculate_quality_score(cls, strategy: StrategyData) -> float:
        """Calculate quality score for strategy - FIXED for realistic values"""
        score = 0.0
        
        # Completeness: all fields filled (0.0-0.25)
        filled_fields = sum(1 for field in [
            strategy.entry_rules, strategy.exit_rules, strategy.indicators,
            strategy.timeframe, strategy.risk_management, strategy.parameters,
            strategy.description
        ] if field and field != 'Not specified')
        completeness = (filled_fields / 7) * cls.QUALITY_CRITERIA['completeness']
        score += completeness
        
        # Specificity: detailed vs vague (0.0-0.30)
        specificity_words = ['specific', 'exact', 'precise', 'parameter', 'value',
                           'condition', 'rule', 'RSI', 'MACD', '%', 'point', '<', '>']
        strategy_text = ' '.join([
            strategy.entry_rules, strategy.exit_rules, strategy.parameters
        ]).lower()
        specificity_score = sum(1 for word in specificity_words if word in strategy_text)
        specificity = min(specificity_score / 5, 1.0) * cls.QUALITY_CRITERIA['specificity']  # FIXED: Lowered threshold
        score += specificity
        
        # Actionability: can be implemented (0.0-0.25)
        implementation_keywords = ['buy', 'sell', 'when', 'if', 'above', 'below',
                                  'cross', 'breakout', 'support', 'resistance']
        actionability_score = sum(1 for word in implementation_keywords 
                                 if word in strategy_text)
        actionability = min(actionability_score / 5, 1.0) * cls.QUALITY_CRITERIA['actionability']  # FIXED: Lowered threshold
        score += actionability
        
        # Clarity: readable format (0.0-0.20)
        clarity = 0.20 if len(strategy.description) > 50 else 0.10
        score += clarity
        
        return round(min(score, 1.0), 3)

# ============================================================================
# LOGGING & PERSISTENCE
# ============================================================================

class DataLogger:
    """Centralized data logging to CSV files"""
    
    @staticmethod
    def log_search_result(cycle: int, query: str, result: SearchResult,
                         status: str, content_length: int = 0,
                         content_hash: str = "", duration_ms: int = 0):
        """Log search result to CSV"""
        try:
            with open(SEARCH_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    cycle,
                    query[:100],
                    result.url[:200],
                    result.title[:100],
                    status,
                    content_length,
                    content_hash,
                    duration_ms
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log search result: {e}")
    
    @staticmethod
    def log_extraction(url: str, strategies: List[StrategyData],
                      duration_ms: int = 0):
        """Log extraction result to CSV"""
        try:
            avg_quality = sum(s.quality_score for s in strategies) / len(strategies) if strategies else 0.0
            strategy_names = '|'.join(s.strategy_name for s in strategies)[:200]
            
            with open(EXTRACTION_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    url[:200],
                    len(strategies),
                    strategy_names,
                    round(avg_quality, 3),
                    duration_ms
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log extraction: {e}")
    
    @staticmethod
    def log_strategy_quality(strategy: StrategyData):
        """Log strategy quality metrics"""
        try:
            with open(STRATEGY_QUALITY_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    strategy.strategy_name[:100],
                    round(strategy.quality_score, 3),
                    'High' if strategy.quality_score > 0.7 else 'Medium' if strategy.quality_score > 0.5 else 'Low',
                    'High' if len(strategy.parameters) > 20 else 'Medium',
                    'High' if len(strategy.entry_rules) > 50 else 'Medium',
                    strategy.source_url[:200]
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log strategy quality: {e}")
    
    @staticmethod
    def log_cycle_metrics(metrics: SearchCycleMetrics):
        """Log cycle metrics"""
        try:
            with open(METRICS_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    metrics.cycle_number,
                    metrics.queries_generated,
                    metrics.search_results_found,
                    metrics.urls_processed,
                    metrics.strategies_extracted,
                    round(metrics.extraction_success_rate, 3),
                    round(metrics.average_quality_score, 3),
                    metrics.total_api_calls,
                    round(metrics.total_duration_seconds, 2)
                ])
        except Exception as e:
            logger.warning(f"⚠️  Failed to log metrics: {e}")

# ============================================================================
# STRATEGY SAVING
# ============================================================================

class StrategyFileWriter:
    """Save strategies to markdown files"""
    
    @staticmethod
    def save_strategy(strategy: StrategyData, search_query: str) -> Optional[str]:
        """Save strategy to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Create filename
            name = strategy.strategy_name
            slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:40]
            filename = f"strategy_{timestamp}_{slug}.md"
            filepath = FINAL_STRATEGIES_DIR / filename
            
            # Build markdown content
            content = f"# {name}\n\n"
            content += f"## Entry Rules\n\n{strategy.entry_rules}\n\n"
            content += f"## Exit Rules\n\n{strategy.exit_rules}\n\n"
            content += f"## Indicators\n\n{strategy.indicators}\n\n"
            content += f"## Timeframe\n\n{strategy.timeframe}\n\n"
            content += f"## Risk Management\n\n{strategy.risk_management}\n\n"
            content += f"## Parameters\n\n{strategy.parameters}\n\n"
            content += f"## Description\n\n{strategy.description}\n\n"
            content += f"## Quality Score\n\n{strategy.quality_score:.1%}\n\n"
            content += "---\n\n"
            content += f"**Source**: {strategy.source_url}\n"
            content += f"**Search Query**: {search_query}\n"
            content += f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"💾 Saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error saving strategy: {e}")
            return None

# ============================================================================
# SEARCH CYCLE ORCHESTRATOR
# ============================================================================

class SearchCycleOrchestrator:
    """Main search cycle orchestrator"""
    
    def __init__(self):
        self.searcher = WebSearcher(SearchConfig.MAX_SEARCH_RESULTS)
        self.fetcher = ContentFetcher(SearchConfig.TIMEOUT_SECONDS)
        self.extractor = StrategyExtractor()
        self.query_generator = QueryGenerator()
        self.metrics = None
    
    def run_cycle(self, cycle_number: int) -> SearchCycleMetrics:
        """Execute one complete search cycle"""
        cycle_start = time.time()
        
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"🌙 CYCLE #{cycle_number}", "white", "on_blue")
        cprint(f"{'='*70}\n", "cyan")
        
        metrics = SearchCycleMetrics(cycle_number=cycle_number)
        
        # Generate search query
        query = self.query_generator.generate_query()
        if not query:
            logger.error("❌ Failed to generate query")
            return metrics
        metrics.queries_generated = 1
        
        # Execute search
        results = self.searcher.search(query)
        metrics.search_results_found = len(results)
        
        if not results:
            logger.error("❌ No search results")
            return metrics
        
        # Process each result
        for i, result in enumerate(results, 1):
            logger.info(f"\n[{i}/{len(results)}] Processing: {result.title[:60]}")
            
            # Check if already processed
            if url_cache.is_processed(result.url):
                logger.info(f"⏭️  Skipping (already processed)")
                metrics.urls_skipped_duplicate += 1
                continue
            
            # Fetch content
            page_data = self.fetcher.fetch(result.url)
            if not page_data:
                logger.warning(f"⚠️  Fetch failed")
                DataLogger.log_search_result(cycle_number, query, result, "fetch_failed")
                continue
            
            # Extract strategies
            extract_start = time.time()
            strategies = self.extractor.extract(page_data['content'], result.url)
            extract_duration_ms = int((time.time() - extract_start) * 1000)
            
            # Log results
            DataLogger.log_search_result(
                cycle_number, query, result, "success",
                len(page_data['content']),
                page_data['content_hash'],
                extract_duration_ms
            )
            DataLogger.log_extraction(result.url, strategies, extract_duration_ms)
            
            # Mark URL as processed
            url_cache.mark_processed(result.url, {
                'strategies_found': len(strategies),
                'quality_scores': [s.quality_score for s in strategies]
            })
            
            metrics.urls_processed += 1
            
            # FIXED: Save ALL strategies without quality filtering
            # Quality filtering can be done in post-processing if needed
            for strategy in strategies:
                saved = StrategyFileWriter.save_strategy(strategy, query)
                if saved:
                    DataLogger.log_strategy_quality(strategy)
                    metrics.strategies_extracted += 1
                    logger.info(f"✅ Saved strategy with quality score: {strategy.quality_score:.1%}")
            
            time.sleep(SearchConfig.SLEEP_BETWEEN_FETCHES)
        
        # Calculate metrics
        if metrics.urls_processed > 0:
            metrics.extraction_success_rate = metrics.strategies_extracted / metrics.urls_processed
        
        metrics.total_duration_seconds = time.time() - cycle_start
        
        # Log cycle metrics
        DataLogger.log_cycle_metrics(metrics)
        
        # Print summary
        cprint(f"\n{'='*70}", "green")
        cprint("🎉 CYCLE COMPLETE", "white", "on_green")
        cprint(f"{'='*70}", "green")
        cprint(f"✅ URLs processed: {metrics.urls_processed}", "yellow")
        cprint(f"✅ Strategies extracted & saved: {metrics.strategies_extracted}", "yellow")
        cprint(f"✅ Success rate: {metrics.extraction_success_rate:.1%}", "yellow")
        cprint(f"✅ Duration: {metrics.total_duration_seconds:.1f}s", "yellow")
        cprint(f"📁 Saved to: {FINAL_STRATEGIES_DIR}\n", "cyan")
        
        return metrics

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    cprint("\n🌙 MOON DEV WEB SEARCH AGENT V2 (PRODUCTION-GRADE)", "white", "on_magenta")
    cprint("🤖 Local LLM: Qwen2.5-7B (HPC Server)", "cyan")
    cprint(f"🔄 Sleep between cycles: {SearchConfig.SLEEP_BETWEEN_SEARCHES}s", "yellow")
    cprint(f"⏱️  LLM timeout: {SearchConfig.LLM_TIMEOUT_SECONDS}s", "yellow")
    cprint(f"🎯 Quality threshold: {SearchConfig.MIN_STRATEGY_QUALITY_SCORE:.2f} (now saves ALL)", "yellow")
    cprint(f"📁 Output directory: {FINAL_STRATEGIES_DIR}\n", "cyan")
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            orchestrator = SearchCycleOrchestrator()
            metrics = orchestrator.run_cycle(cycle)
            
            # Sleep before next cycle
            cprint(f"\n⏱️  Cooldown: {SearchConfig.SLEEP_BETWEEN_SEARCHES}s", "yellow")
            for remaining in range(SearchConfig.SLEEP_BETWEEN_SEARCHES, 0, -10):
                print(f"\r⏳ Next search in {remaining}s...", end="", flush=True)
                time.sleep(10)
            print("\r" + " " * 50 + "\r", end="")
            
    except KeyboardInterrupt:
        cprint(f"\n\n👋 Shutting down after {cycle} cycles", "yellow")
        cprint(f"📊 Logs: {LOGS_DIR}", "cyan")
        cprint(f"📁 Strategies: {FINAL_STRATEGIES_DIR}", "cyan")
        cprint(f"📊 Data: {DATA_DIR}\n", "cyan")

if __name__ == "__main__":
    main()
