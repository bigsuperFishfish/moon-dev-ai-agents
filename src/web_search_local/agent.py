import os
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from termcolor import cprint
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SLEEP_BETWEEN_SEARCHES = 120
MAX_SEARCH_RESULTS = 8
MAX_CONTENT_LENGTH = 12000
MIN_CONTENT_LENGTH = 500

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "web_search_local"
FINAL_STRATEGIES_DIR = DATA_DIR / "final_strategies"
SEARCH_RESULTS_CSV = DATA_DIR / "search_results.csv"
EXTRACTION_LOG_CSV = DATA_DIR / "extraction_log.csv"

# 🔥 DEBUG: Print paths on startup
cprint(f"\n📁 PROJECT_ROOT: {PROJECT_ROOT}", "cyan")
cprint(f"📁 DATA_DIR: {DATA_DIR}", "cyan")
cprint(f"📁 FINAL_STRATEGIES_DIR: {FINAL_STRATEGIES_DIR}", "cyan")
cprint(f"✅ FINAL_STRATEGIES_DIR exists: {FINAL_STRATEGIES_DIR.exists()}", "green" if FINAL_STRATEGIES_DIR.exists() else "red")

DATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# 🔥 Verify after creation
cprint(f"✅ FINAL_STRATEGIES_DIR now exists: {FINAL_STRATEGIES_DIR.exists()}\n", "green")

def init_csv_files():
    """Initialize CSV files if they don't exist"""
    if not SEARCH_RESULTS_CSV.exists():
        with open(SEARCH_RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'query', 'url', 'title', 'status', 'content_length'])
    
    if not EXTRACTION_LOG_CSV.exists():
        with open(EXTRACTION_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'url', 'num_strategies', 'strategy_names', 'extraction_quality'])

init_csv_files()

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

def log_search_result(query, url, title, status, content_length):
    """Log search result to CSV"""
    try:
        with open(SEARCH_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                query[:100],
                url[:200],
                title[:100],
                status,
                content_length
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

def call_local_llm(messages, max_tokens=2048, temperature=0.7):
    """Call local Qwen LLM with retry logic"""
    max_retries = 2
    retry_wait = 30
    
    for attempt in range(max_retries + 1):
        try:
            cprint(f"\n🤖 Calling Qwen (attempt {attempt + 1}/{max_retries + 1})...", "cyan")
            
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
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code != 200:
                cprint("❌ LLM API Error: " + str(response.status_code), "red")
                if attempt < max_retries:
                    wait_time = retry_wait * (attempt + 1)
                    cprint(f"⏳ Retrying in {wait_time}s...", "yellow")
                    time.sleep(wait_time)
                    continue
                return None
                
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            cprint("✅ LLM (" + str(len(content)) + " chars)", "green")
            return content
            
        except requests.exceptions.Timeout:
            cprint(f"⚠️  LLM timeout (attempt {attempt + 1}/{max_retries + 1})", "yellow")
            if attempt < max_retries:
                wait_time = retry_wait * (attempt + 1)
                cprint(f"⏳ Retrying in {wait_time}s...", "yellow")
                time.sleep(wait_time)
                continue
            return None
        
        except Exception as e:
            cprint("❌ Error calling LLM: " + str(e), "red")
            if attempt < max_retries:
                wait_time = retry_wait * (attempt + 1)
                cprint(f"⏳ Retrying in {wait_time}s...", "yellow")
                time.sleep(wait_time)
                continue
            return None
    
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
        # Join list items with commas
        return ', '.join(str(item) for item in value)
    if isinstance(value, dict):
        # Convert dict to readable format
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
        cprint("⚠️  No LLM response", "yellow")
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
        
        # 🔥 DEBUG: Print path info
        cprint(f"\n🔍 Saving strategy to:", "cyan")
        cprint(f"   Directory: {FINAL_STRATEGIES_DIR}", "cyan")
        cprint(f"   File: {filename}", "cyan")
        cprint(f"   Full path: {filepath}", "cyan")
        cprint(f"   Directory exists: {FINAL_STRATEGIES_DIR.exists()}", "green" if FINAL_STRATEGIES_DIR.exists() else "red")
        cprint(f"   Is directory: {FINAL_STRATEGIES_DIR.is_dir()}", "green" if FINAL_STRATEGIES_DIR.is_dir() else "red")
        
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
        
        # 🔥 Verify file was written
        if filepath.exists():
            cprint(f"✅ SAVED: {filename} ({len(content)} bytes)", "green")
            return filename
        else:
            cprint(f"❌ SAVE FAILED: File not found after write: {filepath}", "red")
            return None
        
    except Exception as e:
        cprint(f"❌ Error saving strategy: {str(e)}", "red")
        import traceback
        cprint(traceback.format_exc(), "red")
        return None

def run_search_cycle():
    """Run one complete search cycle"""
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
    
    for i, result in enumerate(results, 1):
        # Check for duplicates
        if url_processed(result['url']):
            cprint("\n⏭️  Skipping (already processed): " + result['url'][:50], "yellow")
            continue
        
        total_new_urls += 1
        
        cprint("\n" + "="*70, "cyan")
        cprint("📄 Processing [" + str(i) + "/" + str(len(results)) + "] " + result['title'][:50], "white", "on_blue")
        cprint("="*70, "cyan")
        
        page_data = fetch_webpage_content(result['url'])
        if not page_data:
            log_search_result(query, result['url'], result['title'], "fetch_failed", 0)
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
            len(page_data['content'])
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
        
        # 🔥 DEBUG: Print extraction results
        cprint(f"\n📊 Extraction results: {len(strategies)} strategies found", "cyan")
        
        # Save each strategy
        for strategy in strategies:
            result = save_strategy(strategy, result['url'], query)
            if result:
                total_strategies += 1
        
        time.sleep(2)
    
    cprint("\n" + "="*70, "green")
    cprint("🎉 CYCLE COMPLETE", "white", "on_green")
    cprint("="*70, "green")
    cprint("\n✅ New URLs processed: " + str(total_new_urls), "yellow")
    cprint("✅ Strategies extracted: " + str(total_strategies), "yellow")
    cprint(f"✅ Files saved to: {FINAL_STRATEGIES_DIR}", "yellow")
    cprint(f"✅ Directory contents: {list(FINAL_STRATEGIES_DIR.glob('*.md'))}", "yellow")
    
    return True

def main():
    """Main entry point"""
    cprint("\n🌙 MOON DEV IMPROVED LOCAL WEB SEARCH AGENT 🌙", "white", "on_magenta")
    cprint("🤖 Model: " + LOCAL_LLM_MODEL, "cyan")
    cprint("🔄 LLM URL: " + LOCAL_LLM_URL, "cyan")
    cprint("🔄 Search interval: " + str(SLEEP_BETWEEN_SEARCHES) + "s", "yellow")
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
