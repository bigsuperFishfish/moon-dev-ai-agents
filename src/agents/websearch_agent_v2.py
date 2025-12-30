import os
import time
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from termcolor import cprint
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# ⚠️ HPC LLM URL - CRITICAL FOR YOUR SETUP
LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SLEEP_BETWEEN_SEARCHES = 120
MAX_SEARCH_RESULTS = 8
MAX_CONTENT_LENGTH = 12000
MIN_CONTENT_LENGTH = 500

# ✅ SIMILARITY THRESHOLDS - Configurable deduplication
STRATEGY_SIMILARITY_THRESHOLD = 0.85  # 0-1, higher = stricter dedup
CONTENT_HASH_DEDUP = True  # Enable content-level deduplication

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "web_search_local"
FINAL_STRATEGIES_DIR = DATA_DIR / "final_strategies"
SEARCH_RESULTS_CSV = DATA_DIR / "search_results.csv"
EXTRACTION_LOG_CSV = DATA_DIR / "extraction_log.csv"
DEDUPLICATION_LOG_CSV = DATA_DIR / "deduplication_log.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

def init_csv_files():
    """Initialize CSV files if they don't exist"""
    if not SEARCH_RESULTS_CSV.exists():
        with open(SEARCH_RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'query', 'url', 'title', 'status', 'content_length', 'content_hash'])
    
    if not EXTRACTION_LOG_CSV.exists():
        with open(EXTRACTION_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'url', 'num_strategies', 'strategy_names', 'extraction_quality'])

    if not DEDUPLICATION_LOG_CSV.exists():
        with open(DEDUPLICATION_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'strategy_name', 'content_hash', 'similarity_score', 'decision', 'reason'])

init_csv_files()

def get_content_hash(content):
    """Calculate SHA256 hash of content for deduplication"""
    return hashlib.sha256(content.encode()).hexdigest()[:12]

def url_processed(url):
    """Check if URL has already been processed"""
    if not SEARCH_RESULTS_CSV.exists():
        return False
    try:
        with open(SEARCH_RESULTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('url') == url:
                    return True
    except:
        pass
    return False

def content_hash_exists(content_hash):
    """Check if content hash has been processed (content-level dedup)"""
    if not SEARCH_RESULTS_CSV.exists():
        return False
    try:
        with open(SEARCH_RESULTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('content_hash') == content_hash:
                    return True
    except:
        pass
    return False

def cosine_similarity_score(text1, text2):
    """Calculate cosine similarity between two text strings (0-1)"""
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors)[0, 1]
        return float(similarity)
    except:
        return 0.0

def check_strategy_similarity(strategy_name, strategy_description):
    """
    Check if similar strategy already exists in final_strategies folder
    Returns: (is_duplicate, best_match_name, similarity_score)
    """
    existing_files = list(FINAL_STRATEGIES_DIR.glob("*.md"))
    
    if not existing_files:
        return False, None, 0.0
    
    max_similarity = 0.0
    best_match = None
    
    try:
        strategy_text = f"{strategy_name} {strategy_description}".lower()
        
        for existing_file in existing_files:
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read().lower()
                
                # Calculate similarity of strategy description
                similarity = cosine_similarity_score(strategy_text, existing_content)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = existing_file.name
            except:
                continue
        
        is_duplicate = max_similarity >= STRATEGY_SIMILARITY_THRESHOLD
        return is_duplicate, best_match, max_similarity
        
    except Exception as e:
        cprint(f"⚠️ Similarity check error: {str(e)}", "yellow")
        return False, None, 0.0

def log_search_result(query, url, title, status, content_length, content_hash=""):
    """Log search result to CSV with content hash"""
    try:
        with open(SEARCH_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                query[:100],
                url[:200],
                title[:100],
                status,
                content_length,
                content_hash
            ])
    except Exception as e:
        cprint("❌ Error logging search result: " + str(e), "red")

def log_extraction(url, num_strategies, strategy_names, quality):
    """Log extraction result to CSV"""
    try:
        with open(EXTRACTION_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                url[:200],
                num_strategies,
                '|'.join(strategy_names)[:200],
                quality
            ])
    except Exception as e:
        cprint("❌ Error logging extraction: " + str(e), "red")

def log_deduplication(strategy_name, content_hash, similarity_score, decision, reason):
    """Log deduplication decision to CSV"""
    try:
        with open(DEDUPLICATION_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                strategy_name[:100],
                content_hash,
                f"{similarity_score:.3f}",
                decision,
                reason[:200]
            ])
    except Exception as e:
        cprint("❌ Error logging deduplication: " + str(e), "red")

def call_local_llm(messages, max_tokens=2048, temperature=0.7):
    """Call local Qwen LLM on HPC"""
    try:
        cprint("\n🤖 Calling Qwen (HPC)...", "cyan")
        
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            LOCAL_LLM_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # Increased timeout to 120s for HPC
        )
        
        if response.status_code != 200:
            cprint("❌ LLM API Error: " + str(response.status_code), "red")
            return None
            
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        cprint("✅ LLM (" + str(len(content)) + " chars)", "green")
        return content
        
    except Exception as e:
        cprint("❌ Error calling LLM: " + str(e), "red")
        return None

def generate_search_query():
    """Generate a specific, high-quality search query"""
    cprint("\n" + "="*70, "cyan")
    cprint("🧠 GENERATING DIVERSE SEARCH QUERY", "white", "on_blue")
    cprint("="*70, "cyan")
    
    system_msg = "You are a professional trading strategy researcher. Generate ONE very specific search query."
    
    user_msg = "Generate ONE creative search query to find high-quality trading strategies with specific parameters.\n"
    user_msg += "Be VERY SPECIFIC. Include:\n"
    user_msg += "- Strategy type (momentum, mean reversion, breakout, arbitrage, pairs trading, statistical arbitrage)\n"
    user_msg += "- Indicators (RSI, MACD, Bollinger Bands, moving averages, volume profile)\n"
    user_msg += "- Timeframes (15m, 1h, 4h, daily)\n"
    user_msg += "- Parameters and entry/exit rules\n"
    user_msg += "- Sometimes filter by site (reddit.com/r/algotrading, tradingview.com, github.com)\n"
    user_msg += "- Sometimes filter by file type (backtest results, PDF, academic papers)\n"
    user_msg += "- Mix in keywords: 'parameters', 'rules', 'entry', 'exit', 'stop loss', 'take profit'\n"
    user_msg += "\nONLY output the raw search query. NO explanations, NO quotes. Just the query text."
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    query = call_local_llm(messages, max_tokens=100, temperature=0.9)
    
    if query:
        query = query.strip().replace('"', '').replace('\n', ' ')
        cprint("\n✨ Generated Query: " + query, "yellow", "on_blue")
        return query
    
    return None

def search_with_duckduckgo(query, max_results=8):
    """Search DuckDuckGo for results"""
    cprint("\n" + "="*70, "cyan")
    cprint("🦆 SEARCHING DUCKDUCKGO", "white", "on_magenta")
    cprint("="*70, "cyan")
    
    cprint("\n🔍 Query: " + query, "cyan")
    
    try:
        results = []
        
        ddgs = DDGS()
        search_results = list(ddgs.text(query, max_results=max_results))
        
        for i, result in enumerate(search_results, 1):
            cprint("\n[" + str(i) + "] " + result['title'][:60], "green")
            cprint("    " + result['href'][:70], "cyan")
            
            results.append({
                'title': result['title'],
                'url': result['href'],
                'snippet': result.get('body', '')
            })
        
        cprint("\n✅ Found " + str(len(results)) + " results", "green")
        return results
        
    except Exception as e:
        cprint("❌ DuckDuckGo error: " + str(e), "red")
        return []

def fetch_webpage_content(url):
    """Fetch and clean webpage content"""
    try:
        cprint("\n🌐 Fetching: " + url[:60] + "...", "cyan")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            cprint("❌ HTTP " + str(response.status_code), "red")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        title = soup.title.string.strip() if soup.title else "No Title"
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        text = text[:MAX_CONTENT_LENGTH]
        
        # Validate content length
        if len(text) < MIN_CONTENT_LENGTH:
            cprint("❌ Content too short (" + str(len(text)) + " chars, min: " + str(MIN_CONTENT_LENGTH) + ")", "red")
            return None
        
        cprint("✅ Fetched " + str(len(text)) + " chars", "green")
        
        return {
            'url': url,
            'title': title,
            'content': text
        }
        
    except Exception as e:
        cprint("❌ Fetch error: " + str(e), "red")
        return None

def safe_str(value):
    """Safely convert any value to string, handling lists and dicts"""
    if value is None:
        return 'Not specified'
    if isinstance(value, list):
        return ', '.join(str(item) for item in value)
    if isinstance(value, dict):
        return ', '.join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip()

def extract_strategies_from_content(content, source_url):
    """Extract trading strategies from content using LLM"""
    cprint("\n" + "="*70, "cyan")
    cprint("🧠 EXTRACTING STRATEGIES", "white", "on_blue")
    cprint("="*70, "cyan")
    
    system_msg = "You are an expert trading strategy analyst. Extract ACTIONABLE, SPECIFIC strategies. Return ONLY valid JSON."
    
    user_msg = "Extract ALL trading strategies from this content. Be AGGRESSIVE - extract even partial ideas.\n\n"
    user_msg += "For EACH strategy provide:\n"
    user_msg += "1. strategy_name: Clear, concise name\n"
    user_msg += "2. entry_rules: SPECIFIC entry conditions (e.g., 'RSI < 30 AND price above 200-day SMA')\n"
    user_msg += "3. exit_rules: SPECIFIC exit conditions (e.g., 'RSI > 70 OR stop loss hit')\n"
    user_msg += "4. indicators: All indicators used (RSI, MACD, SMA, etc. with parameters)\n"
    user_msg += "5. timeframe: Trading timeframe (15m, 1h, 4h, daily, etc.)\n"
    user_msg += "6. risk_management: Stop loss, position sizing, take profit\n"
    user_msg += "7. description: Full detailed description (2-3 sentences minimum)\n"
    user_msg += "8. parameters: Specific parameter values if mentioned\n\n"
    user_msg += "Return JSON ONLY in this format:\n"
    user_msg += '{"strategies": [{"strategy_name": "...", "entry_rules": "...", "exit_rules": "...", "indicators": "...", "timeframe": "...", "risk_management": "...", "description": "...", "parameters": "..."}]}\n\n'
    user_msg += "Content:\n" + content[:6000]
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    response = call_local_llm(messages, max_tokens=3000, temperature=0.3)
    
    if not response:
        return []
    
    try:
        # Extract JSON from response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response)
        strategies = data.get('strategies', [])
        
        cprint("\n✨ Extracted " + str(len(strategies)) + " strategies!", "green")
        for i, s in enumerate(strategies, 1):
            cprint("  [" + str(i) + "] " + safe_str(s.get('strategy_name', 'Untitled')), "yellow")
        
        return strategies
        
    except json.JSONDecodeError as e:
        cprint("❌ JSON parse error: " + str(e), "red")
        cprint("Response was: " + response[:200], "red")
        return []

def save_strategy(strategy, source_url, search_query):
    """Save strategy to markdown file with all 8 fields"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        name = safe_str(strategy.get('strategy_name', 'Untitled Strategy'))
        slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:40]
        
        filename = "strategy_" + timestamp + "_" + slug + ".md"
        filepath = FINAL_STRATEGIES_DIR / filename
        
        # Build content with safe string conversion for all fields
        content = "# " + name + "\n\n"
        content += "## Entry Rules\n\n"
        content += safe_str(strategy.get('entry_rules', 'Not specified')) + "\n\n"
        content += "## Exit Rules\n\n"
        content += safe_str(strategy.get('exit_rules', 'Not specified')) + "\n\n"
        content += "## Indicators\n\n"
        content += safe_str(strategy.get('indicators', 'Not specified')) + "\n\n"
        content += "## Timeframe\n\n"
        content += safe_str(strategy.get('timeframe', 'Not specified')) + "\n\n"
        content += "## Risk Management\n\n"
        content += safe_str(strategy.get('risk_management', 'Not specified')) + "\n\n"
        content += "## Parameters\n\n"
        content += safe_str(strategy.get('parameters', 'Not specified')) + "\n\n"
        content += "## Description\n\n"
        content += safe_str(strategy.get('description', 'No description')) + "\n\n"
        content += "---\n\n"
        content += "**Source**: " + source_url + "\n"
        content += "**Search Query**: " + search_query + "\n"
        content += "**Extracted**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        cprint("💾 Saved: " + filename, "green")
        return filename
        
    except Exception as e:
        cprint("❌ Error saving strategy: " + str(e), "red")
        return None

def run_search_cycle():
    """Run one complete search cycle with deduplication"""
    cprint("\n" + "="*70, "magenta")
    cprint("🚀 STARTING SEARCH CYCLE", "white", "on_magenta")
    cprint("="*70, "magenta")
    
    query = generate_search_query()
    if not query:
        cprint("❌ Failed to generate query", "red")
        return False
    
    results = search_with_duckduckgo(query, max_results=MAX_SEARCH_RESULTS)
    if not results:
        cprint("❌ No search results", "red")
        return False
    
    total_strategies = 0
    total_new_urls = 0
    total_duplicates = 0
    
    for i, result in enumerate(results, 1):
        # Check for URL duplicates
        if url_processed(result['url']):
            cprint("\n⏭️  Skipping (URL already processed): " + result['url'][:50], "yellow")
            continue
        
        total_new_urls += 1
        
        cprint("\n" + "="*70, "cyan")
        cprint("📄 Processing [" + str(i) + "/" + str(len(results)) + "] " + result['title'][:50], "white", "on_blue")
        cprint("="*70, "cyan")
        
        page_data = fetch_webpage_content(result['url'])
        if not page_data:
            log_search_result(query, result['url'], result['title'], "fetch_failed", 0, "")
            continue
        
        # Calculate content hash for content-level dedup
        content_hash = get_content_hash(page_data['content'])
        cprint(f"\n🔐 Content Hash: {content_hash}", "cyan")
        
        # Check content hash deduplication
        if CONTENT_HASH_DEDUP and content_hash_exists(content_hash):
            cprint(f"⏭️  Skipping (content hash already processed: {content_hash})", "yellow")
            log_search_result(query, result['url'], result['title'], "content_duplicate", 
                            len(page_data['content']), content_hash)
            continue
        
        strategies = extract_strategies_from_content(
            page_data['content'],
            result['url']
        )
        
        # Log search result
        log_search_result(
            query,
            result['url'],
            result['title'],
            "success",
            len(page_data['content']),
            content_hash
        )
        
        # Log extraction quality
        quality = "high" if len(strategies) > 0 else "low"
        strategy_names = [safe_str(s.get('strategy_name', 'Unknown')) for s in strategies]
        log_extraction(
            result['url'],
            len(strategies),
            strategy_names,
            quality
        )
        
        # Save each strategy with similarity check
        for strategy in strategies:
            strategy_name = safe_str(strategy.get('strategy_name', 'Untitled'))
            strategy_desc = safe_str(strategy.get('description', ''))
            
            # Check strategy similarity
            is_duplicate, best_match, similarity = check_strategy_similarity(strategy_name, strategy_desc)
            
            if is_duplicate:
                cprint(f"⚠️ Duplicate detected: {strategy_name} (similarity: {similarity:.2%})", "yellow")
                cprint(f"   Similar to: {best_match}", "yellow")
                log_deduplication(strategy_name, content_hash, similarity, "SKIPPED", 
                                f"Similar to {best_match}")
                total_duplicates += 1
            else:
                saved = save_strategy(strategy, result['url'], query)
                if saved:
                    cprint(f"✅ Saved: {strategy_name} (similarity: {similarity:.2%})", "green")
                    log_deduplication(strategy_name, content_hash, similarity, "SAVED", 
                                    f"New strategy")
                    total_strategies += 1
        
        time.sleep(2)
    
    cprint("\n" + "="*70, "green")
    cprint("🎉 CYCLE COMPLETE", "white", "on_green")
    cprint("="*70, "green")
    cprint("\n✅ New URLs processed: " + str(total_new_urls), "yellow")
    cprint("✅ Strategies extracted: " + str(total_strategies), "yellow")
    cprint("⏭️  Strategies skipped (duplicates): " + str(total_duplicates), "yellow")
    cprint("✅ Files saved to: " + str(FINAL_STRATEGIES_DIR), "yellow")
    cprint("✅ Dedup logs: " + str(DEDUPLICATION_LOG_CSV), "yellow")
    
    return True

def main():
    """Main entry point"""
    cprint("\n🌙 MOON DEV WEBSEARCH AGENT V2 WITH DEDUPLICATION 🌙", "white", "on_magenta")
    cprint("🤖 Model: " + LOCAL_LLM_MODEL, "cyan")
    cprint("🔄 LLM URL: " + LOCAL_LLM_URL, "cyan")
    cprint("🔄 Search interval: " + str(SLEEP_BETWEEN_SEARCHES) + "s", "yellow")
    cprint("📊 Similarity threshold: " + str(STRATEGY_SIMILARITY_THRESHOLD), "cyan")
    cprint("📁 Strategies folder: " + str(FINAL_STRATEGIES_DIR), "cyan")
    cprint("📊 CSV logs: " + str(DATA_DIR), "cyan\n")
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            cprint("\n" + "="*70, "blue")
            cprint("🔄 CYCLE #" + str(cycle), "white", "on_blue")
            cprint("="*70, "blue")
            
            run_search_cycle()
            
            cprint("\n⏱️ Cooldown: " + str(SLEEP_BETWEEN_SEARCHES) + "s", "yellow")
            for remaining in range(SLEEP_BETWEEN_SEARCHES, 0, -10):
                print("\r⏳ Next search in " + str(remaining) + "s...", end="", flush=True)
                time.sleep(10)
            print("\r" + " "*50 + "\r", end="")
            
    except KeyboardInterrupt:
        cprint("\n\n👋 Shutting down after " + str(cycle) + " cycles", "yellow")
        cprint("📊 Check CSVs in: " + str(DATA_DIR), "cyan")
        cprint("📁 Strategies in: " + str(FINAL_STRATEGIES_DIR), "cyan")
        cprint("🌙 Thanks for using Moon Dev Search Agent!", "magenta")

if __name__ == "__main__":
    main()
