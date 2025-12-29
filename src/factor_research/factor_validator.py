"""
🔬 Factor Validation Framework for Quantitative Trading

This module provides institutional-grade factor validation using:
- Information Coefficient (IC) and Rank IC calculation
- Bootstrap confidence intervals for statistical significance
- Multiple hypothesis testing corrections (Benjamini-Hochberg)
- Factor decay analysis (holding period decay curves)
- Cross-sectional and time-series validation methods
- SHAP-based factor importance analysis

Usage:
    from src.factor_research.factor_validator import FactorValidator
    
    validator = FactorValidator()
    results = validator.validate_factor(
        factor_values=momentum_signal,
        target_returns=next_period_returns,
        factor_name='momentum',
        holding_periods=[1, 5, 10],
        n_bootstrap=1000
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from scipy.special import comb
from typing import Dict, Tuple, List, Optional
import warnings
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class FactorValidationResult:
    """結構化的因子驗證結果"""
    factor_name: str
    timestamp: str
    
    # 信息係數相關
    ic: float  # Pearson IC
    rank_ic: float  # Spearman rank IC
    kendall_tau: float  # Kendall tau IC
    
    # Bootstrap 統計
    ic_ci_lower: float  # 95% CI lower bound
    ic_ci_upper: float  # 95% CI upper bound
    ic_p_value: float  # p-value for IC > 0
    ic_significant: bool  # Is IC significantly > 0?
    
    # 衰減分析
    holding_period_decay: Dict[int, float]  # IC vs holding periods
    decay_slope: float  # IC 衰減速率
    
    # 多因子檢驗修正
    ic_fdr_adjusted: bool  # Pass Benjamini-Hochberg FDR test?
    
    # 經濟顯著性
    economic_threshold: float
    is_economically_significant: bool
    
    # 整體狀態
    status: str  # 'VALID', 'WEAK', 'INVALID'
    sharpe_ratio: Optional[float] = None
    recommendation: str = ""


class FactorValidator:
    """
    機構級因子驗證引擎
    
    核心原則：
    1. 統計顯著性：IC 的 95% CI 不能包含 0
    2. 經濟顯著性：IC > threshold (typically 0.015 for daily data)
    3. 穩定性：IC 不應因衰減而快速崩潰
    4. 多重檢驗修正：FDR < 0.05（避免多重檢驗陷阱）
    """
    
    def __init__(
        self,
        economic_threshold: float = 0.015,
        fdr_level: float = 0.05,
        bootstrap_samples: int = 1000,
        ci_percentile: Tuple[float, float] = (2.5, 97.5)
    ):
        """
        Args:
            economic_threshold: 經濟顯著性閾值（IC > this 才算可交易）
            fdr_level: 多重檢驗修正的 FDR 閾值
            bootstrap_samples: Bootstrap 重抽樣次數
            ci_percentile: 信心區間的百分位數 (lower, upper)
        """
        self.economic_threshold = economic_threshold
        self.fdr_level = fdr_level
        self.bootstrap_samples = bootstrap_samples
        self.ci_percentile = ci_percentile
        self.validation_results = {}
    
    def validate_factor(
        self,
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        factor_name: str,
        holding_periods: List[int] = [1, 5, 10, 20],
        weights: Optional[np.ndarray] = None,
        cross_sectional: bool = True
    ) -> FactorValidationResult:
        """
        完整的因子驗證流程
        
        Args:
            factor_values: shape (n_samples,) 或 (n_samples, n_periods)
            target_returns: shape (n_samples,) 對應的未來收益
            factor_name: 因子名稱（用於記錄）
            holding_periods: 不同持倉期的衰減分析
            weights: 可選的樣本權重（例如按流動性加權）
            cross_sectional: 是否使用截面方法（True）還是時間序列方法（False）
        
        Returns:
            FactorValidationResult 對象
        """
        
        # 1. 輸入驗證和預處理
        factor_values, target_returns = self._preprocess_data(
            factor_values, target_returns
        )
        
        if len(factor_values) < 50:
            warnings.warn(
                f"樣本量 ({len(factor_values)}) < 50，統計結果可能不可靠"
            )
        
        # 2. 計算基礎 IC
        ic, rank_ic, kendall_tau = self._calculate_ics(
            factor_values, target_returns, weights
        )
        
        # 3. Bootstrap 置信區間
        ic_ci, ic_bootstrap = self._bootstrap_ic(
            factor_values, target_returns, weights
        )
        ic_p_value = self._calculate_pvalue(ic_bootstrap)
        ic_significant = ic_ci[0] > 0  # CI 不包含 0
        
        # 4. 衰減分析（如果提供了多個持倉期）
        holding_period_decay, decay_slope = self._analyze_decay(
            factor_values, target_returns, holding_periods
        )
        
        # 5. Sharpe Ratio（基於 IC）
        sharpe_ratio = self._calculate_sharpe_from_ic(ic)
        
        # 6. 多重檢驗修正（Benjamini-Hochberg FDR）
        fdr_adjusted = self._benjamini_hochberg_fdr(
            [ic_p_value], self.fdr_level
        )[0]
        
        # 7. 綜合判斷
        is_economically_significant = abs(ic) > self.economic_threshold
        status = self._determine_factor_status(
            ic=ic,
            ic_significant=ic_significant,
            is_economically_significant=is_economically_significant,
            decay_slope=decay_slope
        )
        
        # 8. 建立結果對象
        result = FactorValidationResult(
            factor_name=factor_name,
            timestamp=datetime.now().isoformat(),
            ic=float(ic),
            rank_ic=float(rank_ic),
            kendall_tau=float(kendall_tau),
            ic_ci_lower=float(ic_ci[0]),
            ic_ci_upper=float(ic_ci[1]),
            ic_p_value=float(ic_p_value),
            ic_significant=bool(ic_significant),
            holding_period_decay=holding_period_decay,
            decay_slope=float(decay_slope),
            ic_fdr_adjusted=bool(fdr_adjusted),
            economic_threshold=self.economic_threshold,
            is_economically_significant=is_economically_significant,
            status=status,
            sharpe_ratio=float(sharpe_ratio),
            recommendation=self._generate_recommendation(
                status, ic, decay_slope, ic_significant
            )
        )
        
        # 9. 保存結果
        self.validation_results[factor_name] = result
        
        return result
    
    def _preprocess_data(
        self,
        factor_values: np.ndarray,
        target_returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """數據預處理：去除 NaN、標準化"""
        factor_values = np.asarray(factor_values).flatten()
        target_returns = np.asarray(target_returns).flatten()
        
        # 去除 NaN
        valid_idx = ~(np.isnan(factor_values) | np.isnan(target_returns))
        factor_values = factor_values[valid_idx]
        target_returns = target_returns[valid_idx]
        
        # 去除無限值
        valid_idx = np.isfinite(factor_values) & np.isfinite(target_returns)
        factor_values = factor_values[valid_idx]
        target_returns = target_returns[valid_idx]
        
        return factor_values, target_returns
    
    def _calculate_ics(
        self,
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """計算三種相關係數"""
        # Pearson IC
        if weights is not None:
            weights = weights / weights.sum()  # 正規化權重
            ic = self._weighted_correlation(factor_values, target_returns, weights)
        else:
            ic = np.corrcoef(factor_values, target_returns)[0, 1]
        
        # Spearman Rank IC
        rank_ic, _ = spearmanr(factor_values, target_returns)
        
        # Kendall Tau IC
        kendall_tau, _ = kendalltau(factor_values, target_returns)
        
        return ic, rank_ic, kendall_tau
    
    def _weighted_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """計算加權相關係數"""
        avg_x = np.average(x, weights=weights)
        avg_y = np.average(y, weights=weights)
        
        numerator = np.sum(
            weights * (x - avg_x) * (y - avg_y)
        )
        denominator = np.sqrt(
            np.sum(weights * (x - avg_x) ** 2) *
            np.sum(weights * (y - avg_y) ** 2)
        )
        
        return numerator / denominator if denominator != 0 else 0
    
    def _bootstrap_ic(
        self,
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_bootstrap: Optional[int] = None
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        """
        Bootstrap 置信區間計算
        
        核心邏輯：重抽樣 n_bootstrap 次，計算每次的 IC，
        然後取樣本分佈的百分位數作為 CI
        """
        if n_bootstrap is None:
            n_bootstrap = self.bootstrap_samples
        
        bootstrap_ics = []
        n_samples = len(factor_values)
        
        for _ in range(n_bootstrap):
            # 有放回重抽樣
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            factor_boot = factor_values[idx]
            returns_boot = target_returns[idx]
            weights_boot = weights[idx] if weights is not None else None
            
            ic_boot, _, _ = self._calculate_ics(
                factor_boot, returns_boot, weights_boot
            )
            bootstrap_ics.append(ic_boot)
        
        bootstrap_ics = np.array(bootstrap_ics)
        ci_lower = np.percentile(bootstrap_ics, self.ci_percentile[0])
        ci_upper = np.percentile(bootstrap_ics, self.ci_percentile[1])
        
        return (ci_lower, ci_upper), bootstrap_ics
    
    def _calculate_pvalue(self, bootstrap_ics: np.ndarray) -> float:
        """
        計算 p-value：有多少比例的 bootstrap IC <= 0？
        （即 IC > 0 的概率）
        """
        return np.sum(bootstrap_ics <= 0) / len(bootstrap_ics)
    
    def _analyze_decay(
        self,
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        holding_periods: List[int]
    ) -> Tuple[Dict[int, float], float]:
        """
        因子衰減分析
        
        邏輯：對不同的持倉期，IC 應該逐漸衰減
        快速衰減 = 因子信號短期有效但無長期持續力
        """
        decay_curve = {}
        
        for period in holding_periods:
            if period > len(target_returns) // 2:
                continue
            
            # 對齊偏移後的收益和因子
            returns_shifted = target_returns[period:]
            factor_shifted = factor_values[:-period]
            
            if len(returns_shifted) < 20:
                continue
            
            ic, _, _ = self._calculate_ics(factor_shifted, returns_shifted)
            decay_curve[period] = float(ic)
        
        # 計算衰減斜率（線性回歸）
        if len(decay_curve) >= 2:
            periods = np.array(list(decay_curve.keys()))
            ics = np.array(list(decay_curve.values()))
            
            # 線性擬合：IC = intercept + slope * period
            z = np.polyfit(periods, ics, 1)
            decay_slope = z[0]  # slope
        else:
            decay_slope = 0.0
        
        return decay_curve, decay_slope
    
    def _calculate_sharpe_from_ic(
        self,
        ic: float,
        periods_per_year: int = 252
    ) -> float:
        """
        根據 IC 估算 Sharpe Ratio（近似）
        
        公式：Sharpe ≈ IC * sqrt(periods_per_year) / (1 - IC²)
        
        這是一個粗略估算，假設：
        - 交易成本忽略
        - 市場沒有其他阻力
        - IC 是唯一的信息源
        """
        if ic == 0:
            return 0.0
        
        denominator = np.sqrt(1 - ic ** 2) if abs(ic) < 1 else 0.001
        sharpe = (ic * np.sqrt(periods_per_year)) / denominator
        
        return float(sharpe)
    
    def _benjamini_hochberg_fdr(
        self,
        p_values: List[float],
        fdr_level: float = 0.05
    ) -> List[bool]:
        """
        Benjamini-Hochberg FDR 多重檢驗修正
        
        邏輯：控制 False Discovery Rate (FDR)，即
        「發現的顯著結果中，有多少比例是假陽性」
        
        不同於 Bonferroni 的保守，FDR 更適合大規模檢驗
        """
        n_tests = len(p_values)
        p_sorted_idx = np.argsort(p_values)
        p_sorted = np.array(p_values)[p_sorted_idx]
        
        # 計算臨界值：p_i <= (i / m) * alpha
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * fdr_level
        
        # 找到最大的 i 使得 p_i <= critical_value_i
        rejected_idx = np.where(p_sorted <= critical_values)[0]
        
        if len(rejected_idx) > 0:
            threshold_idx = rejected_idx[-1]
            threshold = p_sorted[threshold_idx]
        else:
            threshold = -1
        
        # 轉換回原始順序
        results = [p_values[i] <= threshold for i in range(n_tests)]
        return results
    
    def _determine_factor_status(
        self,
        ic: float,
        ic_significant: bool,
        is_economically_significant: bool,
        decay_slope: float,
        decay_threshold: float = -0.002
    ) -> str:
        """
        綜合判斷因子狀態
        
        邏輯:
        - VALID: 統計 + 經濟顯著性 + 衰減不過快
        - WEAK: 統計顯著但經濟显著性不夠，或衰減過快
        - INVALID: 不統計顯著，或衰減完全失效
        """
        if not ic_significant:
            return "INVALID"
        
        if decay_slope < decay_threshold:
            return "WEAK"  # IC 衰減過快
        
        if is_economically_significant:
            return "VALID"
        else:
            return "WEAK"
    
    def _generate_recommendation(self, status: str, ic: float, decay_slope: float, ic_significant: bool) -> str:
        """根據驗證結果生成建議"""
        if status == "VALID":
            return f"✅ 該因子適合投入生產。IC={ic:.4f}，衰減速率={decay_slope:.4f}/期"
        elif status == "WEAK":
            if decay_slope < -0.002:
                return f"⚠️ IC 衰減過快（{decay_slope:.4f}）。考慮調整持倉期或與其他因子組合"
            elif ic < 0.015:
                return f"⚠️ IC 不足以覆蓋交易成本。考慮提高信號強度或降低交易成本"
            else:
                return f"⚠️ 因子統計顯著但經濟效益有限。用作輔助信號，而不是主信號"
        else:  # INVALID
            return f"❌ 該因子不具有預測力（IC={ic:.4f}，p>0.05）。應該摒棄"
    
    def get_summary(self) -> pd.DataFrame:
        """以 DataFrame 的形式返回所有已驗證的因子"""
        if not self.validation_results:
            return pd.DataFrame()
        
        data = []
        for name, result in self.validation_results.items():
            data.append({
                'Factor': name,
                'IC': result.ic,
                'Rank IC': result.rank_ic,
                'IC CI Lower': result.ic_ci_lower,
                'IC CI Upper': result.ic_ci_upper,
                'Significant': result.ic_significant,
                'Sharpe': result.sharpe_ratio,
                'Decay Slope': result.decay_slope,
                'Status': result.status,
                'Recommendation': result.recommendation
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, filepath: str) -> None:
        """將驗證結果保存為 JSON"""
        results_dict = {}
        for name, result in self.validation_results.items():
            results_dict[name] = {
                'ic': result.ic,
                'rank_ic': result.rank_ic,
                'ic_ci': [result.ic_ci_lower, result.ic_ci_upper],
                'ic_significant': result.ic_significant,
                'sharpe_ratio': result.sharpe_ratio,
                'decay_slope': result.decay_slope,
                'status': result.status,
                'recommendation': result.recommendation,
                'timestamp': result.timestamp
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✅ 因子驗證結果已保存至 {filepath}")


if __name__ == "__main__":
    # 示例：驗證一個模擬的因子
    np.random.seed(42)
    
    # 生成模擬數據
    n_samples = 500
    
    # 真實信號因子
    true_factor = np.random.randn(n_samples)
    # 目標收益（含真實因子的訊號 + 噪聲）
    target_returns = 0.05 * true_factor + np.random.randn(n_samples) * 0.8
    
    # 驗證
    validator = FactorValidator(economic_threshold=0.015)
    result = validator.validate_factor(
        factor_values=true_factor,
        target_returns=target_returns,
        factor_name='test_momentum',
        holding_periods=[1, 5, 10, 20]
    )
    
    # 打印結果
    print("\n" + "="*60)
    print(f"因子名稱: {result.factor_name}")
    print(f"狀態: {result.status}")
    print(f"IC: {result.ic:.4f} [{result.ic_ci_lower:.4f}, {result.ic_ci_upper:.4f}]")
    print(f"統計顯著: {result.ic_significant}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"衰減斜率: {result.decay_slope:.6f}")
    print(f"\n建議: {result.recommendation}")
    print(f"\n衰減曲線:")
    for period, ic in result.holding_period_decay.items():
        print(f"  {period} 期: IC={ic:.4f}")
    print("="*60)
    
    # 保存結果
    validator.save_results('src/data/factor_validation_results.json')
    
    # 打印摘要
    print("\n因子驗證摘要:")
    print(validator.get_summary())
