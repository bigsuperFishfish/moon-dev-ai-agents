# 🌙 Factor Validation + Orchestrator 實現指南

## 📋 目錄
1. [系統架構概述](#系統架構概述)
2. [因子驗證實現](#因子驗證實現)
3. [Orchestrator 實現](#orchestrator-實現)
4. [集成步驟](#集成步驟)
5. [實戰示例](#實戰示例)
6. [常見陷阱](#常見陷阱)

---

## 系統架構概述

```
┌─────────────────────────────────────────────────────────┐
│              Agent Orchestrator (主調度器)              │
│  統一管理所有 Agent 的生命週期、信號、監控              │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [Signal Layer]  [Decision Layer] [Risk Layer]
   ✅ Agent 只     ✅ 因子組合     ✅ 最後防線
   生成信號        ✅ Sharpe        ✅ Position
                   加權             ✅ Daily P&L
                                     ✅ Limits
   ┌─────────────────────────────────────────┐
   │  Factor Validation Registry             │
   │  所有信號必須來自已驗證的因子           │
   │                                         │
   │  momentum (IC=0.045, Status=VALID)      │
   │  order_imbalance (IC=0.058, VALID)      │
   │  mean_reversion (IC=0.032, WEAK)        │
   │  liquidation_pressure (IC=0.021, WEAK)  │
   └─────────────────────────────────────────┘
```

**核心原則：**
- ❌ Agent 不直接下單
- ✅ Agent 生成結構化信號
- ✅ DecisionEngine 按因子權重組合信號
- ✅ RiskManager 最後驗證

---

## 因子驗證實現

### 第 1 步：安裝依賴

```bash
# 如果還沒有安裝統計庫
pip install scipy numpy pandas scikit-learn

# 更新 requirements.txt
pip freeze > requirements.txt
```

### 第 2 步：使用 FactorValidator

#### 基本用法

```python
from src.factor_research.factor_validator import FactorValidator
import numpy as np

# 初始化驗證器
validator = FactorValidator(
    economic_threshold=0.015,  # IC 必須 > 0.015 才算經濟顯著
    fdr_level=0.05,            # 多重檢驗修正的 FDR 閾值
    bootstrap_samples=1000     # 1000 次重抽樣
)

# 準備數據（例如：過去 252 個交易日）
n_samples = 252
true_factor = np.random.randn(n_samples)  # 因子信號
target_returns = 0.05 * true_factor + np.random.randn(n_samples) * 0.8  # 目標收益

# 驗證因子
result = validator.validate_factor(
    factor_values=true_factor,
    target_returns=target_returns,
    factor_name='momentum',
    holding_periods=[1, 5, 10, 20]  # 不同持倉期的衰減分析
)

# 查看結果
print(f"Status: {result.status}")  # 'VALID', 'WEAK', or 'INVALID'
print(f"IC: {result.ic:.4f}")
print(f"IC 95% CI: [{result.ic_ci_lower:.4f}, {result.ic_ci_upper:.4f}]")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Decay Slope: {result.decay_slope:.6f}")
print(f"Recommendation: {result.recommendation}")
```

#### 理解驗證結果

```
結果詳解：

1. Status 判定邏輯
   ├─ VALID: IC 顯著 + 經濟顯著 + 衰減不過快
   ├─ WEAK: IC 顯著但經濟效應有限，或衰減過快
   └─ INVALID: IC 不顯著

2. IC 與 CI
   ├─ IC = 信息係數（-1 到 1）
   ├─ 95% CI 不包含 0 → 統計顯著
   └─ 例：IC=0.045 [0.020, 0.070] → 可信的正信號

3. Sharpe Ratio
   ├─ 粗略估計的 Sharpe (未考慮交易成本)
   ├─ Sharpe > 1.0 → 不錯
   ├─ Sharpe > 1.5 → 很好
   └─ Sharpe < 0.5 → 較弱

4. Decay Slope
   ├─ 持倉期越長，IC 應該衰減
   ├─ Decay Slope > -0.002/期 → 衰減過快 (⚠️)
   ├─ Decay Slope ≈ -0.0005/期 → 正常
   └─ 快速衰減 = 短期噪聲，不是真實信號

5. Holding Period Decay
   例：
   {
     1: 0.045,   # 1期持倉時 IC = 0.045
     5: 0.032,   # 5期持倉時 IC = 0.032 (衰減 29%)
    10: 0.020,   # 10期持倉時 IC = 0.020 (衰減 56%)
    20: 0.008    # 20期持倉時 IC = 0.008 (衰減 82%)
   }
```

### 第 3 步：批量驗證多個因子

```python
from src.factor_research.factor_validator import FactorValidator
import pandas as pd

validator = FactorValidator()

# 因子候選池
factors_to_test = {
    'momentum': compute_momentum(price_data),
    'mean_reversion': compute_mean_reversion(price_data),
    'order_imbalance': compute_order_imbalance(order_data),
    'volatility_zscore': compute_volatility_zscore(price_data),
    'liquidation_pressure': compute_liquidation_pressure(liq_data)
}

validation_results = {}

for factor_name, factor_values in factors_to_test.items():
    result = validator.validate_factor(
        factor_values=factor_values,
        target_returns=target_returns,
        factor_name=factor_name
    )
    validation_results[factor_name] = result

# 獲取摘要
summary = validator.get_summary()
print(summary)

# 保存結果
validator.save_results('src/data/factor_validation_results.json')
```

### 第 4 步：篩選已驗證的因子

```python
# 只保留 VALID 的因子
valid_factors = {
    name: result for name, result in validation_results.items()
    if result.status == 'VALID'
}

print(f"✅ 已驗證因子: {len(valid_factors)}")
for name, result in valid_factors.items():
    print(f"  • {name}: IC={result.ic:.4f}, Sharpe={result.sharpe_ratio:.2f}")

# WEAK 因子可用於輔助（權重 0.5 倍）
weak_factors = {
    name: result for name, result in validation_results.items()
    if result.status == 'WEAK'
}

if weak_factors:
    print(f"⚠️  辅助因子 (WEAK): {len(weak_factors)}")
    for name, result in weak_factors.items():
        print(f"  • {name}: IC={result.ic:.4f} (權重 50%)")
```

---

## Orchestrator 實現

### 第 1 步：初始化 Orchestrator

```python
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.factor_research.factor_validator import FactorValidator

# 創建編制器
orchestrator = AgentOrchestrator(
    poll_interval_seconds=900,  # 15 分鐘檢查一次
    max_agents_parallel=4       # 最多並行 4 個 Agent
)

# 從驗證結果中註冊因子
for name, result in valid_factors.items():
    orchestrator.register_factor_from_validation(result)
```

### 第 2 步：定義信號生成 Agent

**關鍵轉變：Agent 不再生成 "BUY" 或 "SELL"，而是生成結構化信號**

```python
# src/agents/momentum_signal_agent.py
from src.orchestrator.agent_orchestrator import AgentSignal, SignalDirection, ConfidenceLevel
from datetime import datetime

class MomentumSignalAgent:
    """
    改造後的 Momentum Agent - 生成信號而非決策交易
    """
    def __init__(self):
        self.name = "momentum_signal"
    
    def generate_signal(self) -> AgentSignal:
        """
        返回結構化信號，而非直接下單命令
        """
        # 計算 momentum 因子
        momentum_strength = self._calculate_momentum()  # 返回 -1 到 +1
        
        # 評估信心
        if abs(momentum_strength) > 0.7:
            confidence = ConfidenceLevel.HIGH
        elif abs(momentum_strength) > 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        # 決定方向
        if momentum_strength > 0.3:
            direction = SignalDirection.LONG
        elif momentum_strength < -0.3:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT
        
        # 生成信號
        signal = AgentSignal(
            agent_name=self.name,
            timestamp=datetime.now().isoformat(),
            signal_strength=momentum_strength,
            signal_direction=direction,
            confidence=confidence,
            reasoning=f"Momentum factor: {momentum_strength:.3f}. 過去 20 日收益趨勢",
            metadata={
                'lookback_period': 20,
                'current_momentum': momentum_strength,
                'threshold_used': 0.3
            }
        )
        
        return signal
    
    def _calculate_momentum(self) -> float:
        # 實現 momentum 計算
        pass


# src/agents/order_imbalance_agent.py
class OrderImbalanceAgent:
    """
    Order Imbalance 信號 Agent
    """
    def __init__(self):
        self.name = "order_imbalance_signal"
    
    def generate_signal(self) -> AgentSignal:
        # 計算 order imbalance
        imbalance = self._calculate_imbalance()  # 0 到 1
        
        # imbalance > 0.6 = 買盤強勢
        signal_strength = (imbalance - 0.5) * 2  # 轉換為 -1 到 +1
        
        confidence = ConfidenceLevel.MEDIUM
        direction = SignalDirection.LONG if imbalance > 0.6 else (
            SignalDirection.SHORT if imbalance < 0.4 else SignalDirection.FLAT
        )
        
        signal = AgentSignal(
            agent_name=self.name,
            timestamp=datetime.now().isoformat(),
            signal_strength=signal_strength,
            signal_direction=direction,
            confidence=confidence,
            reasoning=f"Order imbalance: {imbalance:.2%}",
            metadata={'imbalance_ratio': imbalance}
        )
        
        return signal
    
    def _calculate_imbalance(self) -> float:
        pass
```

### 第 3 步：註冊 Agent 到 Orchestrator

```python
from src.agents.momentum_signal_agent import MomentumSignalAgent
from src.agents.order_imbalance_agent import OrderImbalanceAgent

# 創建 Agent 實例
momentum_agent = MomentumSignalAgent()
order_agent = OrderImbalanceAgent()

# 註冊到編制器
orchestrator.register_agent(
    agent_name="momentum_signal",
    agent_instance=momentum_agent,
    enabled=True  # 啟用
)

orchestrator.register_agent(
    agent_name="order_imbalance_signal",
    agent_instance=order_agent,
    enabled=True
)

# 也可以註冊已有的 Agent（如 sentiment_agent, whale_agent）
# 但需要修改它們的 generate_signal() 方法
```

### 第 4 步：執行編制循環

```python
# 啟動編制器
orchestrator.run()

# 或者手動執行單個循環
for i in range(10):  # 執行 10 個循環
    cycle_result = orchestrator.orchestrate_single_cycle()
    print(f"\n循環 {i+1} 結果:")
    print(f"  最終決策: {cycle_result['decision']['direction']}")
    print(f"  信心強度: {cycle_result['decision']['strength']:.2%}")
    print(f"  風險檢查: {'通過' if cycle_result['risk_checks_passed'] else '未通過'}")
```

---

## 集成步驟

### Step 1: 創建因子驗證管道

```bash
mkdir -p src/factor_research
touch src/factor_research/__init__.py
touch src/factor_research/validated_factors.py
```

### Step 2: 創建 Orchestrator 目錄

```bash
mkdir -p src/orchestrator
touch src/orchestrator/__init__.py
```

### Step 3: 修改現有 Agent

**从:**
```python
# 舊模式
class TradingAgent:
    def run(self):
        # 分析市場
        decision = "BUY"  # 直接決策
        # 執行交易
        self.place_order(decision)
```

**到:**
```python
# 新模式
class TradingAgent:
    def generate_signal(self) -> AgentSignal:
        # 分析市場
        strength = 0.65  # 返回信號強度
        # 返回結構化信號，讓 Orchestrator 決定
        return AgentSignal(
            agent_name="trading",
            timestamp=datetime.now().isoformat(),
            signal_strength=strength,
            signal_direction=SignalDirection.LONG if strength > 0.3 else SignalDirection.FLAT,
            confidence=ConfidenceLevel.HIGH if strength > 0.7 else ConfidenceLevel.MEDIUM,
            reasoning="Market conditions favorable for long positions"
        )
```

### Step 4: 更新 main.py

```python
# src/main.py
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.agents.trading_agent import TradingAgent
from src.agents.risk_agent import RiskAgent
# ... 其他 Agent

def main():
    # 初始化編制器
    orchestrator = AgentOrchestrator()
    
    # 註冊已驗證的因子
    # (從 src/data/factor_validation_results.json 讀取)
    # ...
    
    # 註冊 Agent
    orchestrator.register_agent(
        "trading",
        TradingAgent(),
        enabled=True
    )
    orchestrator.register_agent(
        "risk",
        RiskAgent(),
        enabled=True
    )
    # ... 其他 Agent
    
    # 啟動
    orchestrator.run()

if __name__ == "__main__":
    main()
```

---

## 實戰示例

### 完整的 3 因子系統

```python
#!/usr/bin/env python3
"""
完整示例：驗證 3 個因子並用 Orchestrator 整合
"""

from src.factor_research.factor_validator import FactorValidator
from src.orchestrator.agent_orchestrator import AgentOrchestrator
import numpy as np
from datetime import datetime

def main():
    print("🌙 Moon Dev - 因子驗證 + Orchestrator 完整示例\n")
    
    # =============
    # 第 1 階段：因子驗證
    # =============
    print("[1/3] 驗證因子...")
    
    validator = FactorValidator(economic_threshold=0.015)
    
    # 生成模擬數據
    np.random.seed(42)
    n_samples = 252
    
    # 因子 1: Momentum
    momentum = np.random.randn(n_samples)
    returns_1 = 0.05 * momentum + np.random.randn(n_samples) * 0.8
    
    result_momentum = validator.validate_factor(
        factor_values=momentum,
        target_returns=returns_1,
        factor_name='momentum',
        holding_periods=[1, 5, 10]
    )
    
    # 因子 2: Order Imbalance
    order_imbalance = np.random.randn(n_samples)
    returns_2 = 0.06 * order_imbalance + np.random.randn(n_samples) * 0.75
    
    result_oi = validator.validate_factor(
        factor_values=order_imbalance,
        target_returns=returns_2,
        factor_name='order_imbalance'
    )
    
    # 因子 3: Mean Reversion
    mean_rev = np.random.randn(n_samples)
    returns_3 = 0.03 * mean_rev + np.random.randn(n_samples) * 0.85
    
    result_mr = validator.validate_factor(
        factor_values=mean_rev,
        target_returns=returns_3,
        factor_name='mean_reversion'
    )
    
    # 打印驗證摘要
    print("\n" + "="*60)
    print(validator.get_summary())
    print("="*60)
    
    # =============
    # 第 2 階段：Orchestrator 設置
    # =============
    print("\n[2/3] 初始化 Orchestrator...")
    
    orchestrator = AgentOrchestrator(poll_interval_seconds=60)
    
    # 註冊已驗證的因子
    orchestrator.register_factor_from_validation(result_momentum)
    orchestrator.register_factor_from_validation(result_oi)
    orchestrator.register_factor_from_validation(result_mr)
    
    print("\n已驗證因子註冊表:")
    print(orchestrator.factor_registry.get_summary())
    
    # =============
    # 第 3 階段：模擬信號聚合
    # =============
    print("\n[3/3] 模擬信號聚合...")
    
    from src.orchestrator.agent_orchestrator import AgentSignal, SignalDirection, ConfidenceLevel
    
    # 模擬 3 個 Agent 的信號
    test_signals = {
        'momentum': AgentSignal(
            agent_name='momentum',
            timestamp=datetime.now().isoformat(),
            signal_strength=0.7,
            signal_direction=SignalDirection.LONG,
            confidence=ConfidenceLevel.HIGH,
            reasoning='Strong momentum signal'
        ),
        'order_imbalance': AgentSignal(
            agent_name='order_imbalance',
            timestamp=datetime.now().isoformat(),
            signal_strength=0.5,
            signal_direction=SignalDirection.LONG,
            confidence=ConfidenceLevel.MEDIUM,
            reasoning='Moderate buy pressure'
        ),
        'mean_reversion': AgentSignal(
            agent_name='mean_reversion',
            timestamp=datetime.now().isoformat(),
            signal_strength=-0.3,
            signal_direction=SignalDirection.SHORT,
            confidence=ConfidenceLevel.MEDIUM,
            reasoning='Slight reversal signal'
        )
    }
    
    # 組合信號
    decision = orchestrator.decision_engine.combine_signals(test_signals)
    
    print(f"\n最終決策:")
    print(f"  方向: {decision.final_direction.name}")
    print(f"  信心強度: {decision.final_strength:.2%}")
    print(f"  貢獻 Agent: {', '.join(decision.contributing_agents)}")
    print(f"  理由: {decision.rationale}")
    
    # 風險檢查
    is_valid, warnings = orchestrator.risk_manager.validate_trade(
        decision=decision,
        current_balance=10000,
        current_positions={},
        proposed_size=1000
    )
    
    print(f"\n風險檢查: {'✅ 通過' if is_valid else '❌ 未通過'}")
    if warnings:
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    print("\n✅ 完整示例執行完畢")

if __name__ == "__main__":
    main()
```

執行:
```bash
python examples/factor_validation_orchestrator_demo.py
```

---

## 常見陷阱

### ❌ 陷阱 1: 因子驗證中的過度擬合

**問題:**
```python
# 不好：使用全部數據驗證
result = validator.validate_factor(
    factor_values=all_historical_data,  # 包含未來信息
    target_returns=future_returns
)
```

**解決:**
```python
# 好：使用 Walk-Forward 分割
n_samples = len(data)
train_test_split = int(n_samples * 0.8)

train_factor = data[:train_test_split]
train_returns = returns[:train_test_split]

test_factor = data[train_test_split:]
test_returns = returns[train_test_split:]

# 在訓練集上驗證
result_train = validator.validate_factor(train_factor, train_returns, 'test_factor')

# 在測試集上驗證（Out-of-Sample 驗證）
result_oos = validator.validate_factor(test_factor, test_returns, 'test_factor_oos')
```

### ❌ 陷阱 2: 忽略因子衰減

**問題:**
```python
# IC 在 1 期時 0.045，但在 20 期時只有 0.008
# 這可能是噪聲而非真實信號
holding_period_decay = {
    1: 0.045,
    5: 0.032,
    10: 0.020,
    20: 0.008  # ⚠️ 衰減 82%！
}
```

**檢查衰減是否過快:**
```python
# 計算衰減速率
decay_slope = (holding_period_decay[20] - holding_period_decay[1]) / (20 - 1)
print(f"衰減速率: {decay_slope:.6f}/期")

# 如果 < -0.002/期，認為衰減過快
if decay_slope < -0.002:
    print("⚠️  該因子衰減過快，可能不是真實信號")
```

### ❌ 陷阱 3: 信號強度沒有合理的範圍

**問題:**
```python
# Agent 返回的信號強度亂七八糟
signal_strength = np.random.uniform(-100, 100)  # 不合理！
```

**正確做法:**
```python
# 信號強度應該始終在 -1 到 +1 之間
signal_strength = np.clip(normalized_factor, -1, 1)

# 例如：
# raw_momentum = -5.2  → 正規化 → -1.0
# raw_momentum = 3.8   → 正規化 →  1.0
# raw_momentum = 0.5   → 正規化 →  0.2
```

### ❌ 陷阱 4: 未驗證的因子進入決策層

**問題:**
```python
# 新加入一個因子但沒有驗證就直接用
decision_engine.combine_signals({
    'momentum': valid_signal,
    'new_untested_factor': untested_signal  # ❌ 沒驗證過！
})
```

**正確做法:**
```python
# 必須先驗證
result = validator.validate_factor(...)

if result.status == 'VALID':
    orchestrator.register_factor_from_validation(result)
    # 現在才能用
else:
    print(f"❌ 因子驗證失敗: {result.recommendation}")
```

### ❌ 陷阱 5: 忽視多重檢驗修正

**問題:**
```python
# 測試 100 個因子，其中 5 個在 5% 水平「顯著」
# 實際上可能全是假陽性（隨機出現）
```

**Benjamini-Hochberg FDR 修正已內置:**
```python
# FactorValidator 已自動進行 FDR 修正
result = validator.validate_factor(...)
print(result.ic_fdr_adjusted)  # True = 通過 FDR 檢驗
```

---

## 檢查清單

部署前確認:

- [ ] 所有因子已通過 Factor Validator
- [ ] 至少 3 個因子標記為 'VALID'
- [ ] 所有 Agent 都有 `generate_signal()` 方法
- [ ] Orchestrator 已註冊所有因子和 Agent
- [ ] RiskManager 的限制設置合理
- [ ] 已進行 Walk-Forward 測試（非必須但推薦）
- [ ] 有監控和告警機制
- [ ] 可以手動干預或緊急停止

---

## 下一步

1. 實現真實的數據 Pipeline（連接 BirdEye API、Moon Dev API）
2. 將現有 Agent 改造為信號生成模式
3. 添加實時監控儀表板
4. 集成實際交易執行層
5. 進行至少 3 個月的 Paper Trading

---

**相關文檔:**
- 💾 [factor_validator.py](src/factor_research/factor_validator.py)
- 🎛️ [agent_orchestrator.py](src/orchestrator/agent_orchestrator.py)
- 📊 [CLAUDE.md](CLAUDE.md) - 開發指南

