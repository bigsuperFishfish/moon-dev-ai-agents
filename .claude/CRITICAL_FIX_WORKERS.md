# 🚨 CRITICAL BUG: Worker Threads Not Started!

## 問題診斷

您的代碼顯示：
```
✅ Found 213 strategies
🧵 18 worker threads ready
```

**但實際上工作執行緒沒有啟動！** 這就是為什麼沒有 API 調用。

## 當前代碼問題

```python
# 當前 main() 函數 (BROKEN!):
def main():
    # ... setup code ...
    
    # ✅ 監控執行緒已啟動
    monitor = Thread(target=idea_monitor_thread, ...)
    monitor.start()
    
    # ❌ 缺少: 工作執行緒從未啟動!
    # 結果: 策略在 queue 中，但沒人處理!
    
    while True:
        time.sleep(5)  # 只是等待，什麼都不做
        # 沒有執行緒從 queue 取策略並處理!
```

## 修復方案

在 `main()` 函數中，**在 monitor.start() 之後**，添加:

```python
def worker(worker_id, idea_queue, stats, stats_lock, queued_ideas, queued_lock):
    """🔥 Worker thread - processes strategies from queue"""
    thread_print(f"🚀 Worker started", worker_id, "green")
    
    while not stop_flag.get('stop', False):
        try:
            # 從 queue 取一個策略
            idea = idea_queue.get(timeout=5)
            
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
                    if result.get('return', 0) >= TARGET_RETURN:
                        stats['targets_hit'] += 1
                        thread_print(f"🎯 TARGET HIT! {result['strategy_name']}: {result['return']}%", 
                                   worker_id, "green", attrs=['bold'])
                else:
                    stats['failed'] += 1
                    thread_print(f"❌ Failed: {result.get('error', 'Unknown')}", 
                               worker_id, "red")
            
            # 從 queued 集合移除
            idea_hash = get_idea_hash(idea)
            with queued_lock:
                queued_ideas.discard(idea_hash)
            
            idea_queue.task_done()
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            thread_print(f"💥 Worker error: {str(e)}", worker_id, "red")
            with stats_lock:
                if stats['active'] > 0:
                    stats['active'] -= 1
            time.sleep(5)


def main(run_name=None):
    """Main orchestrator - CONTINUOUS QUEUE MODE WITH V2 FEATURES"""
    
    # ... (existing setup code) ...
    
    idea_queue = Queue()
    queued_ideas = set()
    queued_lock = Lock()
    stats_lock = Lock()
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

    # 🔥 NEW: 啟動 18 個工作執行緒!
    workers = []
    cprint(f"\n🚀 Starting {MAX_PARALLEL_THREADS} worker threads...", "cyan", attrs=['bold'])
    
    for i in range(MAX_PARALLEL_THREADS):
        t = Thread(target=worker, 
                  args=(i, idea_queue, stats, stats_lock, queued_ideas, queued_lock),
                  daemon=True)
        t.start()
        workers.append(t)
        time.sleep(0.1)  # Stagger starts slightly
    
    cprint(f"✅ {len(workers)} workers ready and processing!\n", "green", attrs=['bold'])

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
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=5)

        cprint(f"\n{'='*60}", "cyan", attrs=['bold'])
        cprint(f"📊 FINAL STATS", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan", attrs=['bold'])
        cprint(f"✅ Successful: {stats['successful']}", "green")
        cprint(f"🎯 Targets hit: {stats['targets_hit']}", "green", attrs=['bold'])
        cprint(f"❌ Failed: {stats['failed']}", "red")
        cprint(f"📊 Total completed: {stats['completed']}", "cyan")
        cprint(f"{'='*60}\n", "cyan", attrs=['bold'])
```

## 驗證修復

修復後，您應該看到:

```
✅ Found 213 strategies

🚀 Starting 18 worker threads...
[T00] 🚀 Worker started
[T01] 🚀 Worker started
[T02] 🚀 Worker started
...
[T17] 🚀 Worker started
✅ 18 workers ready and processing!

[T00] 📥 Processing strategy...
[T00] 🔍 RESEARCH: Starting analysis...          # 🔥 AI 調用開始!
[T01] 📥 Processing strategy...
[T01] 🔍 RESEARCH: Starting analysis...          # 🔥 多個 AI 同時調用!
```

## 現在您會看到 API 調用

在 HPC 終端運行:
```bash
# 監控 LLM 伺服器日誌
tail -f /path/to/llm/server.log

# 或檢查請求
watch -n 1 'netstat -an | grep 8000 | grep ESTABLISHED | wc -l'
```

您應該看到 **多個並發連接** 到 `:8000` 端口！

---

**這是核心問題** - 沒有工作執行緒，就沒有 AI 調用！🔥
