"""
Real Exchange Fee Structures for True 1:1 Simulation.

Every trade deducts actual exchange fees.
Target: 50.75% win rate AFTER fees.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExchangeFees:
    """Exchange fee structure."""
    name: str
    maker_fee: float  # Limit order fee
    taker_fee: float  # Market order fee
    withdrawal_fee_btc: float = 0.0
    min_order_usd: float = 1.0

    @property
    def round_trip_maker(self) -> float:
        """Entry + Exit with maker orders."""
        return self.maker_fee * 2

    @property
    def round_trip_taker(self) -> float:
        """Entry + Exit with taker orders."""
        return self.taker_fee * 2

    @property
    def round_trip_mixed(self) -> float:
        """Entry taker, Exit maker (common pattern)."""
        return self.taker_fee + self.maker_fee


# Real exchange fees as of 2024
EXCHANGE_FEES: Dict[str, ExchangeFees] = {
    'coinbase': ExchangeFees(
        name='Coinbase Pro',
        maker_fee=0.004,      # 0.40%
        taker_fee=0.006,      # 0.60%
        withdrawal_fee_btc=0.0,
        min_order_usd=1.0,
    ),
    'coinbase_adv': ExchangeFees(
        name='Coinbase Advanced',
        maker_fee=0.004,      # 0.40%
        taker_fee=0.006,      # 0.60%
        withdrawal_fee_btc=0.0,
        min_order_usd=1.0,
    ),
    'kraken': ExchangeFees(
        name='Kraken',
        maker_fee=0.0016,     # 0.16%
        taker_fee=0.0026,     # 0.26%
        withdrawal_fee_btc=0.00002,
        min_order_usd=10.0,
    ),
    'binance_us': ExchangeFees(
        name='Binance US',
        maker_fee=0.001,      # 0.10%
        taker_fee=0.001,      # 0.10%
        withdrawal_fee_btc=0.0,
        min_order_usd=1.0,
    ),
    'bitstamp': ExchangeFees(
        name='Bitstamp',
        maker_fee=0.003,      # 0.30%
        taker_fee=0.003,      # 0.30%
        withdrawal_fee_btc=0.0,
        min_order_usd=20.0,
    ),
    'gemini': ExchangeFees(
        name='Gemini',
        maker_fee=0.002,      # 0.20%
        taker_fee=0.004,      # 0.40%
        withdrawal_fee_btc=0.0,
        min_order_usd=1.0,
    ),
    'ftx_us': ExchangeFees(
        name='FTX US (historical)',
        maker_fee=0.001,      # 0.10%
        taker_fee=0.004,      # 0.40%
        withdrawal_fee_btc=0.0,
        min_order_usd=1.0,
    ),
}


# Slippage estimates by order size
SLIPPAGE_ESTIMATES: Dict[str, float] = {
    'micro': 0.0001,      # < $100: 0.01%
    'small': 0.0002,      # $100-1000: 0.02%
    'medium': 0.0005,     # $1000-10000: 0.05%
    'large': 0.001,       # $10000-100000: 0.10%
    'whale': 0.002,       # > $100000: 0.20%
}


def get_slippage_estimate(order_size_usd: float) -> float:
    """Get estimated slippage based on order size."""
    if order_size_usd < 100:
        return SLIPPAGE_ESTIMATES['micro']
    elif order_size_usd < 1000:
        return SLIPPAGE_ESTIMATES['small']
    elif order_size_usd < 10000:
        return SLIPPAGE_ESTIMATES['medium']
    elif order_size_usd < 100000:
        return SLIPPAGE_ESTIMATES['large']
    else:
        return SLIPPAGE_ESTIMATES['whale']


def calculate_breakeven_winrate(
    fee_pct: float,
    stop_loss_pct: float = 0.10,
    take_profit_pct: float = 0.15,
    include_slippage: bool = True,
    avg_slippage_pct: float = 0.0002
) -> float:
    """
    Calculate breakeven win rate given fees and risk/reward.

    With stop loss = 10%, take profit = 15%:
    - Loss = -10% - fees - slippage
    - Win = +15% - fees - slippage

    Breakeven: WR * Win + (1-WR) * Loss = 0
    """
    total_cost = fee_pct * 2  # Entry + exit
    if include_slippage:
        total_cost += avg_slippage_pct * 2

    actual_loss = stop_loss_pct + total_cost
    actual_win = take_profit_pct - total_cost

    # WR * actual_win - (1-WR) * actual_loss = 0
    # WR * actual_win - actual_loss + WR * actual_loss = 0
    # WR * (actual_win + actual_loss) = actual_loss
    # WR = actual_loss / (actual_win + actual_loss)

    breakeven = actual_loss / (actual_win + actual_loss)
    return breakeven


def print_fee_analysis():
    """Print fee analysis for all exchanges."""
    print("=" * 70)
    print("EXCHANGE FEE ANALYSIS - TRUE 1:1 SIMULATION")
    print("=" * 70)
    print(f"Stop Loss: 10% | Take Profit: 15%")
    print("=" * 70)

    for name, fees in EXCHANGE_FEES.items():
        breakeven = calculate_breakeven_winrate(
            fee_pct=fees.taker_fee,
            stop_loss_pct=0.10,
            take_profit_pct=0.15,
            include_slippage=True,
        )
        print(f"\n{fees.name}:")
        print(f"  Taker Fee:      {fees.taker_fee*100:.2f}%")
        print(f"  Round-trip:     {fees.round_trip_taker*100:.2f}%")
        print(f"  Breakeven WR:   {breakeven*100:.2f}%")
        print(f"  Our Edge:       55.5% - {breakeven*100:.2f}% = {(0.555 - breakeven)*100:.2f}%")

    print("\n" + "=" * 70)
    print("TARGET: 50.75% win rate after all costs")
    print("OUR BACKTEST: 55.5% win rate")
    print("=" * 70)


if __name__ == '__main__':
    print_fee_analysis()
