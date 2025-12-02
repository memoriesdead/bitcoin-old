"""Market simulation module - liquidity and position sizing."""
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
]
