"""
Market simulation module - liquidity, position sizing, and TRUE 1:1 simulation.

Components:
- Liquidity: Slippage, market impact, execution price
- Kelly: Position sizing using Kelly criterion
- RealisticSimulator: TRUE 1:1 simulation using BLOCKCHAIN DATA
"""
from engine.market.liquidity import (
    calculate_slippage,
    calculate_market_impact,
    get_max_position_size,
    calculate_execution_price,
    calculate_realistic_pnl,
)
from engine.market.kelly_scaler import (
    calculate_kelly_fraction,
    get_capital_tier_kelly_mult,
    calculate_position_size,
    get_max_position_by_tier,
    calculate_trades_to_target,
    estimate_time_to_target,
)
from engine.market.realistic_simulator import (
    RealisticSimulator,
    SimulationConfig,
    FillResult,
    SimulationStats,
    BlockchainOrderBook,
    RejectionReason,
)
from engine.market.calibration import (
    CalibrationSystem,
    CalibrationConfig,
    CalibrationMetrics,
    TradeComparison,
)

__all__ = [
    # Liquidity
    'calculate_slippage',
    'calculate_market_impact',
    'get_max_position_size',
    'calculate_execution_price',
    'calculate_realistic_pnl',
    # Kelly
    'calculate_kelly_fraction',
    'get_capital_tier_kelly_mult',
    'calculate_position_size',
    'get_max_position_by_tier',
    'calculate_trades_to_target',
    'estimate_time_to_target',
    # TRUE 1:1 Realistic Simulator (uses BLOCKCHAIN DATA)
    'RealisticSimulator',
    'SimulationConfig',
    'FillResult',
    'SimulationStats',
    'BlockchainOrderBook',
    'RejectionReason',
    # Calibration System (Paper vs Live)
    'CalibrationSystem',
    'CalibrationConfig',
    'CalibrationMetrics',
    'TradeComparison',
]
