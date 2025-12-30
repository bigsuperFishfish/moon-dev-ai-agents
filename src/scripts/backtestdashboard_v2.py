"""
🌙 Moon Dev's AI Agent Backtests Dashboard v2 🚀
FastAPI web interface for viewing backtest results from rbi_agent_pp_multi_v2.py
Built with love by Moon Dev

NEW IN V2:
- 🌙 HPC LLM Integration (local Qwen 2.5 7B)
- 🔄 Multi-source strategy reading (websearch_local + websearch_research + ideas.txt)
- 📊 Enhanced dashboard with HPC LLM status
- 💾 Thread-safe CSV logging

================================================================================
📝 HOW TO USE THIS DASHBOARD:
================================================================================

1. RUN THE RBI AGENT v2 to generate backtest results:
   ```bash
   python src/agents/rbi_agent_pp_multi_v2.py
   ```
   This will create a CSV file with all your backtest stats at:
   src/data/rbi_pp_multi/backtest_stats.csv

2. CONFIGURE THE CSV PATH below (line 60) to point to your stats CSV

3. RUN THIS DASHBOARD:
   ```bash
   python src/scripts/backtestdashboard_v2.py
   ```

4. OPEN YOUR BROWSER to: http://localhost:8002

================================================================================
⚡ CONFIGURATION:
================================================================================
"""

from fastapi import FastAPI, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn
import shutil
import subprocess
import threading
from datetime import datetime
import sys
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import traceback
import logging
import requests
from termcolor import cprint

# Import MoonDevAPI from this project
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.agents.api import MoonDevAPI
import websockets
import json

# ============================================
# 🌙 HPC LLM CONFIGURATION - V2 NEW!
# ============================================
USE_LOCAL_HPC_LLM = True
LOCAL_LLM_URL = "http://192.168.30.158:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ============================================
# 🔏 CONFIGURATION - CHANGE THESE PATHS TO MATCH YOUR SETUP!
# ============================================

# 📊 Path to your backtest stats CSV file
# This CSV is created by rbi_agent_pp_multi_v2.py after running backtests
# Default: src/data/rbi_pp_multi/backtest_stats.csv (PROJECT_ROOT already defined above)
STATS_CSV = PROJECT_ROOT / "src" / "data" / "rbi_pp_multi" / "backtest_stats.csv"

# 📁 Directory for static files (CSS, JS) and templates (HTML)
# These files are located in: src/data/rbi_pp_multi/static and src/data/rbi_pp_multi/templates
TEMPLATE_BASE_DIR = PROJECT_ROOT / "src" / "data" / "rbi_pp_multi"

# 🗌️ Directory to store user-created folders
# Folders allow you to organize and group your backtest results
USER_FOLDERS_DIR = TEMPLATE_BASE_DIR / "user_folders"

# 🎯 Target return percentage (must match rbi_agent_pp_multi_v2.py TARGET_RETURN)
TARGET_RETURN = 50  # % - Optimization goal
SAVE_IF_OVER_RETURN = 1.0  # % - Minimum return to save to CSV

# 📊 Data Portal Configuration - Moon Dev
DATA_DIR = TEMPLATE_BASE_DIR / "downloads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 📊 Test Data Sets Directory - Historical datasets for backtesting
TEST_DATA_DIR = PROJECT_ROOT / "src" / "data" / "private_data"

# TEST MODE for data portal - Set to True for fast testing with sample data
TEST_MODE = True

# 🎯 Polymarket CSV Paths (relative to PROJECT_ROOT parent - sibling repo)
POLYMARKET_SWEEPS_CSV = PROJECT_ROOT.parent / "Polymarket-Trading-Bots" / "data" / "sweeps_database.csv"
POLYMARKET_EXPIRING_CSV = PROJECT_ROOT.parent / "Polymarket-Trading-Bots" / "data" / "expiring_markets.csv"

# 🎯 Liquidation CSV Paths (relative to PROJECT_ROOT parent - sibling repo)
LIQUIDATIONS_MINI_CSV = PROJECT_ROOT.parent / "Untitled" / "binance_trades_mini.csv"
LIQUIDATIONS_BIG_CSV = PROJECT_ROOT.parent / "Untitled" / "binance_trades.csv"
LIQUIDATIONS_GRAND_CSV = PROJECT_ROOT.parent / "Untitled" / "binance.csv"

# ============================================
# 🚀 FASTAPI APP INITIALIZATION
# ============================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moon Dev's AI Agent Backtests - v2")

# Create user_folders directory if it doesn't exist
USER_FOLDERS_DIR.mkdir(exist_ok=True)

# Track running backtests
running_backtests = {}

# 🌙 Moon Dev Data API Integration
moon_api = MoonDevAPI()

# Track data update status
data_status = {
    "liquidations": {"status": "pending", "last_updated": None, "file_size": None},
    "oi": {"status": "pending", "last_updated": None, "file_size": None}
}

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(TEMPLATE_BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_BASE_DIR / "templates"))

# 🌙 Moon Dev: Request models for folder operations
class AddToFolderRequest(BaseModel):
    folder_name: str
    backtests: List[Dict[str, Any]]

class DeleteFolderRequest(BaseModel):
    folder_name: str

class BacktestRunRequest(BaseModel):
    ideas: str
    run_name: str

# ============================================
# 🌙 HPC LLM FUNCTIONS - V2 NEW!
# ============================================

def check_hpc_llm_status():
    """🌙 Check if HPC LLM is available"""
    try:
        response = requests.post(
            LOCAL_LLM_URL,
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.3
            },
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

# ============================================
# 🌙 MOON DEV DATA API FUNCTIONS
# ============================================

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

# ============================================
# 📊 BACKTEST STATS ENDPOINTS - V2
# ============================================

@app.get("/api/stats")
async def get_stats():
    """📊 Get all backtest stats from CSV"""
    try:
        if not STATS_CSV.exists():
            return {"data": [], "message": "No backtest stats yet"}
        
        df = pd.read_csv(STATS_CSV)
        
        # Convert to JSON-serializable format
        data = df.to_dict('records')
        
        return {
            "data": data,
            "total": len(df),
            "successful": len(df[df['Return %'] > SAVE_IF_OVER_RETURN]) if 'Return %' in df.columns else 0,
            "targets_hit": len(df[df['Return %'] >= TARGET_RETURN]) if 'Return %' in df.columns else 0
        }
    except Exception as e:
        logger.error(f"❌ Error reading stats: {str(e)}")
        return {"data": [], "error": str(e)}

@app.get("/api/stats/summary")
async def get_stats_summary():
    """📊 Get summary statistics"""
    try:
        if not STATS_CSV.exists():
            return {
                "total": 0,
                "successful": 0,
                "targets_hit": 0,
                "avg_return": 0,
                "best_return": 0,
                "best_strategy": "N/A"
            }
        
        df = pd.read_csv(STATS_CSV)
        
        if 'Return %' not in df.columns:
            return {
                "total": len(df),
                "successful": 0,
                "targets_hit": 0,
                "avg_return": 0,
                "best_return": 0,
                "best_strategy": "N/A"
            }
        
        successful = df[df['Return %'] > SAVE_IF_OVER_RETURN]
        targets_hit = df[df['Return %'] >= TARGET_RETURN]
        
        best_idx = df['Return %'].idxmax()
        best_strategy = df.loc[best_idx, 'Strategy Name'] if 'Strategy Name' in df.columns else "N/A"
        best_return = float(df.loc[best_idx, 'Return %']) if pd.notna(df.loc[best_idx, 'Return %']) else 0
        
        return {
            "total": len(df),
            "successful": len(successful),
            "targets_hit": len(targets_hit),
            "avg_return": float(df['Return %'].mean()) if pd.notna(df['Return %'].mean()) else 0,
            "best_return": best_return,
            "best_strategy": str(best_strategy)
        }
    except Exception as e:
        logger.error(f"❌ Error calculating summary: {str(e)}")
        return {
            "total": 0,
            "successful": 0,
            "targets_hit": 0,
            "avg_return": 0,
            "best_return": 0,
            "best_strategy": "Error"
        }

@app.get("/api/hpc-status")
async def get_hpc_status():
    """🌙 Get HPC LLM status"""
    if not USE_LOCAL_HPC_LLM:
        return {"status": "disabled", "mode": "deepseek_fallback"}
    
    is_available = check_hpc_llm_status()
    return {
        "status": "connected" if is_available else "disconnected",
        "url": LOCAL_LLM_URL,
        "model": LOCAL_LLM_MODEL,
        "mode": "hpc_local"
    }

# ============================================
# 📁 FOLDER MANAGEMENT - V2
# ============================================

@app.post("/api/folders")
async def create_folder(request: dict):
    """📁 Create a new folder"""
    try:
        folder_name = request.get("name")
        folder_path = USER_FOLDERS_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        
        return {"success": True, "message": f"Folder '{folder_name}' created"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/folders")
async def get_folders():
    """📁 Get all user folders"""
    try:
        folders = [f.name for f in USER_FOLDERS_DIR.iterdir() if f.is_dir()]
        return {"folders": folders}
    except Exception as e:
        return {"folders": [], "error": str(e)}

@app.post("/api/folders/{folder_name}/add")
async def add_to_folder(folder_name: str, request: AddToFolderRequest):
    """📁 Add backtests to a folder"""
    try:
        folder_path = USER_FOLDERS_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        
        # Save backtest data to folder
        data_file = folder_path / "backtests.json"
        with open(data_file, 'w') as f:
            json.dump(request.backtests, f, indent=2)
        
        return {"success": True, "message": f"Added {len(request.backtests)} backtests to folder"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/folders/{folder_name}")
async def delete_folder(folder_name: str):
    """📁 Delete a folder"""
    try:
        folder_path = USER_FOLDERS_DIR / folder_name
        shutil.rmtree(folder_path)
        return {"success": True, "message": f"Folder '{folder_name}' deleted"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# 📄 HTML ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """📊 Main dashboard page"""
    try:
        hpc_status = await get_hpc_status()
        summary = await get_stats_summary()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "target_return": TARGET_RETURN,
            "hpc_status": hpc_status,
            "summary": summary
        })
    except Exception as e:
        return f"<h1>❌ Error loading dashboard</h1><p>{str(e)}</p>"

# ============================================
# 🌍 MAIN
# ============================================

def main():
    """🚀 Start the dashboard server"""
    cprint(f"\n{'='*70}", "cyan", attrs=['bold'])
    cprint(f"🌙 Moon Dev's Backtests Dashboard v2 🚀", "cyan", attrs=['bold'])
    cprint(f"{'='*70}", "cyan", attrs=['bold'])
    
    cprint(f"\n📊 Stats CSV: {STATS_CSV}", "magenta")
    cprint(f"🎯 Target Return: {TARGET_RETURN}%", "green", attrs=['bold'])
    cprint(f"🌙 LLM Mode: {'LOCAL HPC' if USE_LOCAL_HPC_LLM else 'DEEPSEEK'}", "magenta", attrs=['bold'])
    
    # Check HPC status
    hpc_ok = check_hpc_llm_status() if USE_LOCAL_HPC_LLM else False
    status_color = "green" if hpc_ok else "red"
    status_text = "Connected" if hpc_ok else "Disconnected"
    cprint(f"🌙 HPC LLM Status: {status_text}", status_color, attrs=['bold'])
    
    cprint(f"\n🚀 Starting server on http://localhost:8002", "yellow", attrs=['bold'])
    cprint(f"{'='*70}\n", "cyan", attrs=['bold'])
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

if __name__ == "__main__":
    main()
