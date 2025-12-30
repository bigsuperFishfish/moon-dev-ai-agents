# 🔧 Complete Fix for Worker Threads

## Error: `stats_lock` is not defined

這個錯誤是因為缺少必要的變數定義。以下是完整修復。

---

## Step 1: 添加 worker 函數

在 `main()` 函數**之前**，添加這個函數：

```python
def worker(worker_id, idea_queue, stats, stats_lock, queued_ideas, queued_lock, stop_flag):
    """🔥 Worker thread - processes strategies from queue"""
    thread_print(f"🚀 Worker started", worker_id, "green")
    
    while not stop_flag.get('stop', False):
        try:
            # 從 queue 取策略 (5 秒 timeout)
            idea = idea_queue.get(timeout=5)
            
            # 更新 active 計數
            with stats_lock:
                stats['active'] += 1
            
            thread_print(f"📥 Processing strategy...", worker_id, "cyan")
            
            # 🔥 這裡才會調用 AI!
            result = process_trading_idea_parallel(idea, worker_id)
            
            # 更新統計
            with stats_lock:
                stats['completed'] += 1
                stats['active'] -= 1
                
                if result.get('success'):
                    stats['successful'] += 1
                    return_pct = result.get('return', 0)
                    if return_pct >= TARGET_RETURN:
                        stats['targets_hit'] += 1
                        thread_print(f"🎯 TARGET HIT! {result.get('strategy_name', 'Unknown')}: {return_pct}%", 
                                   worker_id, "green", attrs=['bold'])
                else:
                    stats['failed'] += 1
                    thread_print(f"❌ Failed: {result.get('error', 'Unknown')}", 
                               worker_id, "red")
            
            # 從 queued 集合移除
            idea_hash = get_idea_hash(idea)
            with queued_lock:
                queued_ideas.discard(idea_hash)
            
            # 標記任務完成
            idea_queue.task_done()
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            if "Empty" not in str(type(e).__name__):
                thread_print(f"💥 Worker error: {str(e)}", worker_id, "red")
                with stats_lock:
                    if stats['active'] > 0:
                        stats['active'] -= 1
            time.sleep(1)
    
    thread_print(f"👋 Worker stopped", worker_id, "yellow")
```

---

## Step 2: 修改 main() 函數

找到 `main()` 函數，在 **idea_queue 定義之後**，添加所有必需的變數：

```python
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

    # 🔥 定義所有必需的變數
    idea_queue = Queue()
    queued_ideas = set()
    queued_lock = Lock()
    stats_lock = Lock()  # 🔥 這個很重要！
    stats = {
        'completed': 0,
        'successful': 0,
        'failed': 0,
        'targets_hit': 0,
        'active': 0
    }
    stop_flag = {'stop': False}

    # 啟動監控執行緒
    monitor = Thread(target=idea_monitor_thread, 
                    args=(idea_queue, queued_ideas, queued_lock, stop_flag), 
                    daemon=True)
    monitor.start()
    cprint("✅ Idea monitor thread started", "green")

    # 🔥 啟動 18 個工作執行緒!
    workers = []
    cprint(f"\n🚀 Starting {MAX_PARALLEL_THREADS} worker threads...", "cyan", attrs=['bold'])
    
    for i in range(MAX_PARALLEL_THREADS):
        t = Thread(target=worker, 
                  args=(i, idea_queue, stats, stats_lock, queued_ideas, queued_lock, stop_flag),
                  daemon=True)
        t.start()
        workers.append(t)
        time.sleep(0.05)  # Stagger starts
    
    cprint(f"✅ {len(workers)} workers ready and processing!\n", "green", attrs=['bold'])

    # 主循環 - 監控狀態
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
        
        # 等待所有 workers 完成
        cprint(f"⏳ Waiting for workers to finish...", "yellow")
        for i, worker_thread in enumerate(workers):
            worker_thread.join(timeout=5)
            if worker_thread.is_alive():
                cprint(f"⚠️  Worker {i} still running (forced stop)", "yellow")

        # 顯示最終統計
        cprint(f"\n{'='*60}", "cyan", attrs=['bold'])
        cprint(f"📊 FINAL STATS", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan", attrs=['bold'])
        cprint(f"✅ Successful: {stats['successful']}", "green")
        cprint(f"🎯 Targets hit: {stats['targets_hit']}", "green", attrs=['bold'])
        cprint(f"❌ Failed: {stats['failed']}", "red")
        cprint(f"📊 Total completed: {stats['completed']}", "cyan")
        cprint(f"{'='*60}\n", "cyan", attrs=['bold'])
```

---

## Step 3: 添加必要的 import

在文件頂部確保有這些 imports：

```python
from queue import Queue, Empty
from threading import Lock, Semaphore, Thread
```

---

## 驗證修復成功

運行後應該看到：

```bash
✅ Idea monitor thread started

🚀 Starting 18 worker threads...
[T00] 🚀 Worker started
[T01] 🚀 Worker started
[T02] 🚀 Worker started
...
[T17] 🚀 Worker started
✅ 18 workers ready and processing!

[T00] 📥 Processing strategy...
[T00] 🔍 RESEARCH: Starting analysis...     # 🔥 AI 開始調用！
[T01] 📥 Processing strategy...
[T01] 🔍 RESEARCH: Starting analysis...     # 🔥 多執行緒並發！
[T02] 📥 Processing strategy...
```

## HPC 上檢查 API 調用

在 HPC 終端運行：

```bash
# 監控連接到 LLM 伺服器的連接數
watch -n 1 'netstat -an | grep 192.168.30.158:8000 | grep ESTABLISHED | wc -l'

# 應該看到 1-18 個並發連接
```

---

## 完整的關鍵變數清單

這些都必須在 `main()` 開始時定義：

| 變數 | 類型 | 用途 |
|------|------|------|
| `idea_queue` | Queue | 存儲待處理策略 |
| `queued_ideas` | set | 防止重複加入 |
| `queued_lock` | Lock | 保護 queued_ideas |
| `stats_lock` | Lock | 🔥 保護 stats 字典 |
| `stats` | dict | 統計計數器 |
| `stop_flag` | dict | 控制執行緒停止 |
| `workers` | list | 存儲 worker 執行緒 |

---

🚀 **現在應該可以正常工作了！**
