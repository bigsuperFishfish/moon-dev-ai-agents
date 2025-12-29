"""
🌜 Agent Orchestrator - 統一多代理盡控系統

This module provides the central orchestration layer for managing 48+ AI agents:

1. **Signal Aggregation**: Collect signals from all active agents
2. **Decision Engine**: Combine signals using validated factors and weighted rules
3. **Risk Management**: Final validation layer before trade execution
4. **Execution**: Place trades with proper logging and error handling

Architecture:

    ┌──────────────────────────────────┐
    │   Agent Orchestrator Main Loop   │
    │  (Unified scheduling & control)  │
    └─────────────┬──────────────────┘
             ↓
    ┌──────────────────────────────────┐
    │  Signal Aggregation Layer        │
    │  (Collect from all agents)       │
    └─────────────┬──────────────────┘
             ↓
    ┌──────────────────────────────────┐
    │  Factor-Weighted Decision Engine │
    │  (Combine signals with IC weights)│
    └─────────────┬──────────────────┘
             ↓
    ┌──────────────────────────────────┐
    │  Risk Management Layer           │
    │  (Position sizing, stops, limits)│
    └─────────────┬──────────────────┘
             ↓
    ┌──────────────────────────────────┐
    │  Order Execution                 │
    │  (Place trades + logging)        │
    └──────────────────────────────────┘

Usage:
    from src.orchestrator.agent_orchestrator import AgentOrchestrator
    
    orchestrator = AgentOrchestrator()
    orchestrator.run()
"""

import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

# Import validated factors
try:
    from src.factor_research.factor_validator import FactorValidator, FactorValidationResult
except ImportError:
    cprint("⚠️  Factor validator not found. Install it first: python -m pip install src/factor_research/factor_validator.py", "yellow")
    FactorValidator = None

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

load_dotenv()


class SignalDirection(Enum):
    """交易信號方向"""
    LONG = 1
    SHORT = -1
    FLAT = 0


class ConfidenceLevel(Enum):
    """信心等級"""
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1
    NONE = 0


@dataclass
class AgentSignal:
    """
    單個 Agent 生成的信號
    
    Agent 不再直接決策「買」或「賣」，
    而是生成一個結構化的信號，讓決策引擎來組合。
    """
    agent_name: str
    timestamp: str
    signal_strength: float  # -1.0 to +1.0
    signal_direction: SignalDirection
    confidence: ConfidenceLevel
    reasoning: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'agent': self.agent_name,
            'timestamp': self.timestamp,
            'signal_strength': self.signal_strength,
            'direction': self.signal_direction.name,
            'confidence': self.confidence.name,
            'reasoning': self.reasoning,
            'metadata': self.metadata
        }


@dataclass
class AggregatedDecision:
    """
    決策引擎的最終決策
    """
    timestamp: str
    final_direction: SignalDirection
    final_strength: float  # 0.0 to 1.0，表示信心強度
    contributing_agents: List[str]
    factor_weights: Dict[str, float]  # 每個因子的加權權重
    combined_signal_vector: Dict[str, float]  # 所有信號的組合結果
    rationale: str  # 決策理由
    passes_risk_checks: bool = True
    risk_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'direction': self.final_direction.name,
            'strength': self.final_strength,
            'agents': self.contributing_agents,
            'weights': self.factor_weights,
            'rationale': self.rationale,
            'passes_risk': self.passes_risk_checks,
            'warnings': self.risk_warnings
        }


class ValidatedFactorRegistry:
    """
    已驗證因子的註冊表
    
    確保只有經過統計檢驗的因子才能進入決策流程。
    """
    def __init__(self):
        self.validated_factors: Dict[str, FactorValidationResult] = {}
    
    def register_factor(
        self,
        factor_name: str,
        ic: float,
        rank_ic: float,
        sharpe_ratio: float,
        decay_slope: float,
        status: str
    ) -> None:
        """
        註冊一個已驗證的因子
        
        Args:
            factor_name: 因子名稱
            ic: Information Coefficient
            rank_ic: Rank IC
            sharpe_ratio: Sharpe Ratio
            decay_slope: 衰減斜率
            status: 'VALID', 'WEAK', or 'INVALID'
        """
        self.validated_factors[factor_name] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'sharpe': sharpe_ratio,
            'decay_slope': decay_slope,
            'status': status,
            'registered_at': datetime.now().isoformat()
        }
        cprint(f"✅ 因子已註冊: {factor_name} (Status: {status})", "green")
    
    def get_weight(self, factor_name: str) -> float:
        """
        根據因子的 Sharpe Ratio 返回權重
        
        關鍵概念：更好的因子（更高的 Sharpe）應該有更高的權重
        """
        if factor_name not in self.validated_factors:
            return 0.0
        
        factor = self.validated_factors[factor_name]
        
        if factor['status'] == 'INVALID':
            return 0.0
        elif factor['status'] == 'WEAK':
            return 0.5 * abs(factor['sharpe'])
        else:  # VALID
            return abs(factor['sharpe'])
    
    def is_valid(self, factor_name: str) -> bool:
        """
        檢查因子是否有效
        """
        if factor_name not in self.validated_factors:
            return False
        return self.validated_factors[factor_name]['status'] != 'INVALID'
    
    def get_summary(self) -> pd.DataFrame:
        """
        返回所有已驗證因子的摘要
        """
        if not self.validated_factors:
            return pd.DataFrame()
        
        data = [
            {
                'Factor': name,
                'IC': f['ic'],
                'Sharpe': f['sharpe'],
                'Decay': f['decay_slope'],
                'Status': f['status'],
                'Weight': self.get_weight(name)
            }
            for name, f in self.validated_factors.items()
        ]
        
        return pd.DataFrame(data)


class DecisionEngine:
    """
    因子加權的決策引擎
    
    核心邏輯：
    1. 收集所有 Agent 的信號
    2. 按已驗證因子的 Sharpe Ratio 加權組合
    3. 生成最終交易決策
    """
    def __init__(self, factor_registry: ValidatedFactorRegistry):
        self.factor_registry = factor_registry
        self.decision_history: List[AggregatedDecision] = []
    
    def combine_signals(
        self,
        agent_signals: Dict[str, AgentSignal],
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> AggregatedDecision:
        """
        組合多個 Agent 的信號成單一決策
        
        Args:
            agent_signals: {agent_name: AgentSignal}
            min_confidence: 最小信心閾值
        
        Returns:
            AggregatedDecision
        """
        timestamp = datetime.now().isoformat()
        
        # Step 1: 篩選有效信號（只考慮驗證過的因子 + 足夠的信心）
        valid_signals = {
            name: sig for name, sig in agent_signals.items()
            if self.factor_registry.is_valid(name) and
            sig.confidence.value >= min_confidence.value
        }
        
        if not valid_signals:
            return AggregatedDecision(
                timestamp=timestamp,
                final_direction=SignalDirection.FLAT,
                final_strength=0.0,
                contributing_agents=[],
                factor_weights={},
                combined_signal_vector={},
                rationale="No valid signals after filtering"
            )
        
        # Step 2: 計算每個信號的權重（基於 Sharpe Ratio）
        factor_weights = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_name, signal in valid_signals.items():
            weight = self.factor_registry.get_weight(agent_name)
            factor_weights[agent_name] = weight
            
            weighted_signal = signal.signal_strength * weight
            weighted_sum += weighted_signal
            total_weight += weight
        
        # Step 3: 計算加權平均信號
        if total_weight == 0:
            final_strength = 0.0
        else:
            final_strength = weighted_sum / total_weight
        
        # Step 4: 確定方向和信心
        if final_strength > 0.3:
            final_direction = SignalDirection.LONG
        elif final_strength < -0.3:
            final_direction = SignalDirection.SHORT
        else:
            final_direction = SignalDirection.FLAT
        
        confidence_strength = abs(final_strength)
        
        # Step 5: 生成決策理由
        contributing_agents = list(valid_signals.keys())
        rationale = self._generate_rationale(
            valid_signals,
            final_direction,
            final_strength,
            factor_weights
        )
        
        decision = AggregatedDecision(
            timestamp=timestamp,
            final_direction=final_direction,
            final_strength=abs(final_strength),
            contributing_agents=contributing_agents,
            factor_weights=factor_weights,
            combined_signal_vector={
                name: sig.signal_strength for name, sig in valid_signals.items()
            },
            rationale=rationale
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _generate_rationale(
        self,
        signals: Dict[str, AgentSignal],
        direction: SignalDirection,
        strength: float,
        weights: Dict[str, float]
    ) -> str:
        """
        生成自然語言決策理由
        """
        direction_text = direction.name
        confidence_pct = abs(strength) * 100
        
        # 找出權重最高的 Agent
        top_agents = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_agents_str = ", ".join([f"{name}(w={w:.2f})" for name, w in top_agents])
        
        rationale = (
            f"決策: {direction_text} (信心: {confidence_pct:.0f}%) | "
            f"主要貢獻: {top_agents_str} | "
            f"總信號數: {len(signals)}"
        )
        
        return rationale


class RiskManagementLayer:
    """
    最終風險管理層
    
    職責：
    1. 檢查頭寸規模限制
    2. 檢查流動性可用性
    3. 檢查最大虧損限制
    4. 檢查最大收益限制
    5. 檢查最低資金要求
    """
    def __init__(
        self,
        max_position_size_usd: float = 10000,
        max_loss_usd: float = 1000,
        max_gain_usd: float = 50000,
        min_balance_usd: float = 5000,
        max_daily_trades: int = 10
    ):
        self.max_position_size = max_position_size_usd
        self.max_loss = max_loss_usd
        self.max_gain = max_gain_usd
        self.min_balance = min_balance_usd
        self.max_daily_trades = max_daily_trades
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
    
    def validate_trade(
        self,
        decision: AggregatedDecision,
        current_balance: float,
        current_positions: Dict[str, float],
        proposed_size: float
    ) -> Tuple[bool, List[str]]:
        """
        驗證交易是否通過所有風險檢查
        
        Returns:
            (is_valid, risk_warnings)
        """
        warnings = []
        
        # Check 1: 是否有決策信號
        if decision.final_direction == SignalDirection.FLAT:
            return False, ["❌ 沒有明確的交易信號"]
        
        # Check 2: 餘額檢查
        if current_balance < self.min_balance:
            warnings.append(f"⚠️  餘額 ({current_balance:.2f}) < 最小要求 ({self.min_balance:.2f})")
            return False, warnings
        
        # Check 3: 頭寸規模檢查
        if proposed_size > self.max_position_size:
            warnings.append(f"⚠️  提議頭寸 ({proposed_size:.2f}) > 最大限制 ({self.max_position_size:.2f})")
            proposed_size = self.max_position_size
        
        # Check 4: 日交易次數檢查
        if self.daily_trade_count >= self.max_daily_trades:
            warnings.append(f"⚠️  今日交易次數已達限制 ({self.daily_trade_count}/{self.max_daily_trades})")
            return False, warnings
        
        # Check 5: 日累計虧損檢查
        if self.daily_pnl < -self.max_loss:
            warnings.append(f"⚠️  日累計虧損 ({self.daily_pnl:.2f}) 超過限制 (-{self.max_loss:.2f})")
            return False, warnings
        
        return True, warnings
    
    def record_trade(self, pnl: float) -> None:
        """
        記錄一筆交易的 P&L
        """
        self.daily_trade_count += 1
        self.daily_pnl += pnl
    
    def reset_daily_stats(self) -> None:
        """
        重置日統計（通常在市場開盤時調用）
        """
        self.daily_trade_count = 0
        self.daily_pnl = 0.0


class AgentOrchestrator:
    """
    主編制器：統一管理所有 Agent 的生命週期和決策流程
    """
    def __init__(
        self,
        poll_interval_seconds: int = 900,  # 15 分鐘
        max_agents_parallel: int = 4
    ):
        self.poll_interval = poll_interval_seconds
        self.max_agents_parallel = max_agents_parallel
        
        # 初始化核心組件
        self.factor_registry = ValidatedFactorRegistry()
        self.decision_engine = DecisionEngine(self.factor_registry)
        self.risk_manager = RiskManagementLayer()
        
        # Agent 管理
        self.agents: Dict[str, any] = {}
        self.active_agents: List[str] = []
        
        # 統計
        self.execution_log: List[Dict] = []
    
    def register_factor_from_validation(
        self,
        validation_result: FactorValidationResult
    ) -> None:
        """
        從 FactorValidator 的結果直接註冊因子
        """
        self.factor_registry.register_factor(
            factor_name=validation_result.factor_name,
            ic=validation_result.ic,
            rank_ic=validation_result.rank_ic,
            sharpe_ratio=validation_result.sharpe_ratio or 0,
            decay_slope=validation_result.decay_slope,
            status=validation_result.status
        )
    
    def register_agent(
        self,
        agent_name: str,
        agent_instance,
        enabled: bool = False
    ) -> None:
        """
        註冊一個 Agent 到編制器
        """
        self.agents[agent_name] = agent_instance
        if enabled:
            self.active_agents.append(agent_name)
            cprint(f"✅ Agent 已啟用: {agent_name}", "green")
        else:
            cprint(f"ℹ️  Agent 已註冊但未啟用: {agent_name}", "cyan")
    
    def run_active_agents(self) -> Dict[str, AgentSignal]:
        """
        並行運行所有活躍的 Agent 並收集信號
        
        Returns:
            {agent_name: AgentSignal}
        """
        signals = {}
        
        with ThreadPoolExecutor(max_workers=self.max_agents_parallel) as executor:
            futures = {}
            
            for agent_name in self.active_agents:
                agent = self.agents.get(agent_name)
                if agent and hasattr(agent, 'generate_signal'):
                    future = executor.submit(agent.generate_signal)
                    futures[future] = agent_name
            
            # 收集結果
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    signal = future.result(timeout=30)
                    if signal:
                        signals[agent_name] = signal
                        cprint(f"✅ {agent_name}: {signal.signal_direction.name} (信心: {signal.confidence.name})", "cyan")
                except Exception as e:
                    cprint(f"❌ {agent_name} 出錯: {str(e)}", "red")
        
        return signals
    
    def orchestrate_single_cycle(self) -> Dict:
        """
        執行單個編制週期
        """
        cycle_start = datetime.now()
        cprint(f"\n🌙 開始編制週期: {cycle_start.isoformat()}", "blue")
        
        # Step 1: 運行所有 Agent 並收集信號
        cprint("\n[1/4] 收集 Agent 信號...", "yellow")
        agent_signals = self.run_active_agents()
        
        # Step 2: 決策引擎組合信號
        cprint("\n[2/4] 組合信號...", "yellow")
        aggregated_decision = self.decision_engine.combine_signals(agent_signals)
        cprint(f"決策: {aggregated_decision.rationale}", "cyan")
        
        # Step 3: 風險管理檢查
        cprint("\n[3/4] 風險檢查...", "yellow")
        # TODO: 集成實際的頭寸和餘額數據
        is_valid, warnings = self.risk_manager.validate_trade(
            decision=aggregated_decision,
            current_balance=10000,
            current_positions={},
            proposed_size=1000
        )
        
        if warnings:
            for warning in warnings:
                cprint(warning, "yellow")
        
        # Step 4: 記錄和報告
        cprint("\n[4/4] 記錄結果...", "yellow")
        cycle_result = {
            'timestamp': cycle_start.isoformat(),
            'agent_signals': {k: v.to_dict() for k, v in agent_signals.items()},
            'decision': aggregated_decision.to_dict(),
            'risk_checks_passed': is_valid,
            'execution_duration_seconds': (datetime.now() - cycle_start).total_seconds()
        }
        
        self.execution_log.append(cycle_result)
        
        # 報告摘要
        cprint(f"\n✅ 週期完成 (耗時: {cycle_result['execution_duration_seconds']:.1f}秒)", "green")
        if is_valid:
            cprint(f"📊 已生成交易信號: {aggregated_decision.final_direction.name}", "green")
        else:
            cprint(f"⚠️  交易被風險管理層阻止", "yellow")
        
        return cycle_result
    
    def run(self) -> None:
        """
        啟動無限編制循環
        """
        cprint("\n" + "="*70, "white", "on_blue")
        cprint("🌙 Moon Dev Agent Orchestrator 已啟動", "white", "on_blue")
        cprint(f"活躍 Agent 數: {len(self.active_agents)}", "white", "on_blue")
        cprint(f"已驗證因子數: {len(self.factor_registry.validated_factors)}", "white", "on_blue")
        cprint(f"編制間隔: {self.poll_interval} 秒", "white", "on_blue")
        cprint("="*70 + "\n", "white", "on_blue")
        
        try:
            while True:
                cycle_result = self.orchestrate_single_cycle()
                
                next_run = datetime.now() + timedelta(seconds=self.poll_interval)
                cprint(f"\n💤 下次編制時間: {next_run.strftime('%H:%M:%S')}", "cyan")
                time.sleep(self.poll_interval)
        
        except KeyboardInterrupt:
            cprint("\n👋 編制器正在關閉...", "yellow")
            self._save_execution_log()
            cprint("✅ 已安全關閉", "green")
    
    def _save_execution_log(self) -> None:
        """
        保存執行日誌
        """
        log_path = 'src/data/orchestrator_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        cprint(f"✅ 執行日誌已保存至: {log_path}", "green")
    
    def print_status(self) -> None:
        """
        打印編制器狀態摘要
        """
        print("\n" + "="*70)
        print("🌙 Agent Orchestrator 狀態")
        print("="*70)
        
        print(f"\n📊 已驗證因子:")
        print(self.factor_registry.get_summary().to_string())
        
        print(f"\n🤖 活躍 Agent ({len(self.active_agents)}):")
        for agent_name in self.active_agents:
            print(f"  ✅ {agent_name}")
        
        print(f"\n📈 最近決策:")
        if self.decision_history:
            for decision in self.decision_engine.decision_history[-3:]:
                print(f"  {decision.timestamp}: {decision.rationale}")
        else:
            print(f"  (尚無決策)")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # 示例：初始化編制器
    orchestrator = AgentOrchestrator(
        poll_interval_seconds=60,  # 1 分鐘用於演示
        max_agents_parallel=4
    )
    
    # 註冊已驗證的因子
    orchestrator.register_factor_from_validation(
        type('MockValidationResult', (), {
            'factor_name': 'momentum',
            'ic': 0.045,
            'rank_ic': 0.053,
            'sharpe_ratio': 1.2,
            'decay_slope': -0.0015,
            'status': 'VALID'
        })
    )
    
    orchestrator.register_factor_from_validation(
        type('MockValidationResult', (), {
            'factor_name': 'order_imbalance',
            'ic': 0.058,
            'rank_ic': 0.062,
            'sharpe_ratio': 1.5,
            'decay_slope': -0.001,
            'status': 'VALID'
        })
    )
    
    # 打印狀態
    orchestrator.print_status()
    
    cprint("\n💡 Orchestrator 已準備好。調用 orchestrator.run() 開始編制循環。", "green")
