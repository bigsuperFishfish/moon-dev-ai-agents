"""StrategySpec v3: Type-safe contract for complete trading strategy units.

Core principle: A complete strategy unit = entry rules + exit rules + position sizing.
No mixing of factors, no psychological stops, only objective market-based rules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class VolMethod(str, Enum):
    """Volatility estimation method."""
    GARMAN_KLASS = "gk"
    PARKINSON = "parkinson"


@dataclass
class StrategySpecV3:
    """Complete v3 strategy specification.
    
    This contract enforces:
    1. Entry is a SINGLE factor (score: continuous -1 to 1)
    2. Exit uses ONLY objective criteria (ATR multiples, time, prices from entry)
    3. Position sizing uses vol targeting (no account equity references)
    4. All rules reference market data only, never account state
    """
    
    # === ENTRY SPEC ===
    entry_factor_name: str
    """Name of entry factor (e.g., 'rsi_oversold', 'ma_crossover', 'zscore_reversion')."""
    
    entry_parameters: Dict[str, Any] = field(default_factory=dict)
    """Entry factor parameters (e.g., {'rsi_period': 14, 'threshold': 30})."""
    
    # === EXIT SPEC (objective market-based only) ===
    stop_loss_atr_mult: float = 2.0
    """ATR multiplier for stop loss. E.g., SL = entry_price - 2.0*atr_at_entry (long)."""
    
    take_profit_atr_mult: float = 3.0
    """ATR multiplier for take profit. E.g., TP = entry_price + 3.0*atr_at_entry (long)."""
    
    time_stop_bars: int = 20
    """Maximum holding period (bars). Exit if not hit TP/SL by this time."""
    
    atr_period: int = 14
    """ATR lookback period for exit calculation."""
    
    # === POSITION SIZING SPEC (vol targeting) ===
    target_vol: float = 0.10
    """Target volatility for position sizing (annualized). E.g., 0.10 = 10%."""
    
    vol_method: VolMethod = VolMethod.GARMAN_KLASS
    """Volatility estimator method: GK (default) or Parkinson."""
    
    vol_window: int = 20
    """Rolling window for realized volatility calculation (bars)."""
    
    vol_floor: float = 0.001
    """Minimum volatility floor (annualized). E.g., 0.001 = 0.1% minimum."""
    
    # === POSITION ADJUSTMENT CONSTRAINTS ===
    delta_pos_max: float = 0.1
    """Maximum position change per bar. Prevents excessive rebalancing due to vol estimates.
    E.g., if delta_pos_max=0.1, position can change by max ±0.1 per bar.
    """
    
    # === METADATA ===
    strategy_name: str = "V3Strategy"
    """Human-readable strategy name."""
    
    description: str = ""
    """Strategy description (for documentation)."""
    
    single_factor_only: bool = True
    """Enforce single-factor rule? If True, entry_factor_name must be ONE factor."""
    
    use_volume_adjustment: bool = False
    """Use volume-weighted volatility estimate (experimental)."""
    
    def __post_init__(self):
        """Validate spec on creation."""
        if self.stop_loss_atr_mult <= 0:
            raise ValueError("stop_loss_atr_mult must be > 0")
        if self.take_profit_atr_mult <= 0:
            raise ValueError("take_profit_atr_mult must be > 0")
        if self.time_stop_bars <= 0:
            raise ValueError("time_stop_bars must be > 0")
        if self.target_vol <= 0:
            raise ValueError("target_vol must be > 0")
        if self.vol_window <= 0:
            raise ValueError("vol_window must be > 0")
        if self.vol_floor < 0:
            raise ValueError("vol_floor must be >= 0")
        if not (0 < self.delta_pos_max <= 1.0):
            raise ValueError("delta_pos_max must be in (0, 1.0]")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be > 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/config."""
        return {
            "entry_factor_name": self.entry_factor_name,
            "entry_parameters": self.entry_parameters,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
            "take_profit_atr_mult": self.take_profit_atr_mult,
            "time_stop_bars": self.time_stop_bars,
            "atr_period": self.atr_period,
            "target_vol": self.target_vol,
            "vol_method": self.vol_method.value,
            "vol_window": self.vol_window,
            "vol_floor": self.vol_floor,
            "delta_pos_max": self.delta_pos_max,
            "strategy_name": self.strategy_name,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategySpecV3":
        """Deserialize from dict."""
        # Convert vol_method string to enum
        if "vol_method" in d and isinstance(d["vol_method"], str):
            d["vol_method"] = VolMethod(d["vol_method"])
        return cls(**d)


# === EXAMPLE SPECS ===

RSI_MEAN_REVERSION_V3 = StrategySpecV3(
    strategy_name="RSI Mean Reversion v3",
    entry_factor_name="rsi_oversold",
    entry_parameters={"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=3.0,
    time_stop_bars=20,
    atr_period=14,
    target_vol=0.10,
    vol_window=20,
    vol_floor=0.001,
    delta_pos_max=0.1,
    description="Buy when RSI < 30, exit on TP/SL/time. Vol-targeted sizing.",
)

MA_CROSSOVER_V3 = StrategySpecV3(
    strategy_name="MA Crossover v3",
    entry_factor_name="ma_crossover",
    entry_parameters={"fast_ma": 20, "slow_ma": 50},
    stop_loss_atr_mult=1.5,
    take_profit_atr_mult=2.5,
    time_stop_bars=50,
    atr_period=14,
    target_vol=0.12,
    vol_window=20,
    vol_floor=0.001,
    delta_pos_max=0.15,
    description="Golden cross / death cross with ATR exits.",
)

MOMENTUM_ZSCORE_V3 = StrategySpecV3(
    strategy_name="Momentum Z-score v3",
    entry_factor_name="momentum_zscore",
    entry_parameters={"momentum_period": 10, "zscore_window": 20, "zscore_threshold": 1.5},
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=2.0,
    time_stop_bars=15,
    atr_period=14,
    target_vol=0.15,
    vol_window=30,  # Longer window for momentum stability
    vol_floor=0.002,
    delta_pos_max=0.12,
    description="Momentum-based Z-score entry with symmetric ATR exits.",
)
