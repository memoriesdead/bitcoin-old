"""
Dry Run Executor
================

Simulates order execution with realistic fills.
Ported from Freqtrade's dry-run mode.

Features:
- Realistic slippage
- Fee calculation
- Partial fills
- Latency simulation
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class DryRunFill:
    """Simulated fill result."""
    amount: float
    price: float
    fee: float
    fee_currency: str
    slippage: float
    latency_ms: float
    timestamp: float


class DryRunExecutor:
    """
    Dry run executor for realistic simulation.

    Freqtrade pattern: Simulate execution with configurable realism.
    """

    # Exchange-specific configs
    EXCHANGE_CONFIGS = {
        'binance': {
            'taker_fee': 0.001,
            'maker_fee': 0.001,
            'min_latency_ms': 20,
            'max_latency_ms': 100,
            'slippage_base': 0.0001,
            'slippage_per_btc': 0.00005,
        },
        'kraken': {
            'taker_fee': 0.0026,
            'maker_fee': 0.0016,
            'min_latency_ms': 50,
            'max_latency_ms': 200,
            'slippage_base': 0.0002,
            'slippage_per_btc': 0.0001,
        },
        'coinbase': {
            'taker_fee': 0.006,
            'maker_fee': 0.004,
            'min_latency_ms': 80,
            'max_latency_ms': 300,
            'slippage_base': 0.0003,
            'slippage_per_btc': 0.00015,
        },
        'default': {
            'taker_fee': 0.001,
            'maker_fee': 0.0005,
            'min_latency_ms': 30,
            'max_latency_ms': 150,
            'slippage_base': 0.0001,
            'slippage_per_btc': 0.00005,
        },
    }

    def __init__(self, exchange: str = 'default',
                 custom_config: Optional[Dict] = None):
        """
        Initialize dry run executor.

        Args:
            exchange: Exchange name for config
            custom_config: Override default config
        """
        self.exchange = exchange
        self.config = self.EXCHANGE_CONFIGS.get(
            exchange,
            self.EXCHANGE_CONFIGS['default']
        )

        if custom_config:
            self.config.update(custom_config)

        # Stats
        self.stats = {
            'fills': 0,
            'total_slippage': 0.0,
            'total_fees': 0.0,
        }

    def simulate_fill(self, symbol: str, side: str, amount: float,
                      price: float, order_type: str = "market",
                      volatility: float = 0.01) -> DryRunFill:
        """
        Simulate order fill.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            amount: Order amount
            price: Current market price
            order_type: "market" or "limit"
            volatility: Current volatility (affects slippage)

        Returns:
            DryRunFill with execution details
        """
        self.stats['fills'] += 1

        # Calculate slippage
        if order_type == "market":
            slippage = self._calculate_slippage(amount, volatility)

            if side == "buy":
                fill_price = price * (1 + slippage)
            else:
                fill_price = price * (1 - slippage)
        else:
            # Limit orders: no slippage (or might not fill)
            slippage = 0.0
            fill_price = price

        # Calculate fee
        fee_rate = self.config['taker_fee'] if order_type == "market" else self.config['maker_fee']
        fee = amount * fill_price * fee_rate

        # Simulate latency
        latency = np.random.uniform(
            self.config['min_latency_ms'],
            self.config['max_latency_ms']
        )

        # Update stats
        self.stats['total_slippage'] += abs(slippage)
        self.stats['total_fees'] += fee

        return DryRunFill(
            amount=amount,
            price=fill_price,
            fee=fee,
            fee_currency="USDT",  # Assume USDT for simplicity
            slippage=slippage,
            latency_ms=latency,
            timestamp=time.time(),
        )

    def _calculate_slippage(self, amount: float, volatility: float) -> float:
        """
        Calculate realistic slippage.

        Slippage depends on:
        1. Base slippage (spread)
        2. Order size impact
        3. Volatility
        """
        base = self.config['slippage_base']
        size_impact = self.config['slippage_per_btc'] * amount
        vol_impact = volatility * 0.1  # Volatility multiplier

        # Add randomness
        noise = np.random.uniform(0.8, 1.2)

        return (base + size_impact + vol_impact) * noise

    def simulate_partial_fill(self, symbol: str, side: str, amount: float,
                              price: float, fill_probability: float = 0.8) -> DryRunFill:
        """
        Simulate partial fill for limit orders.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            amount: Order amount
            price: Limit price
            fill_probability: Probability of fill

        Returns:
            DryRunFill (may be partial)
        """
        # Determine fill amount
        if np.random.random() > fill_probability:
            # No fill
            return DryRunFill(
                amount=0,
                price=price,
                fee=0,
                fee_currency="USDT",
                slippage=0,
                latency_ms=0,
                timestamp=time.time(),
            )

        # Partial fill
        fill_ratio = np.random.uniform(0.3, 1.0)
        filled_amount = amount * fill_ratio

        fee = filled_amount * price * self.config['maker_fee']

        latency = np.random.uniform(
            self.config['min_latency_ms'],
            self.config['max_latency_ms']
        )

        self.stats['fills'] += 1
        self.stats['total_fees'] += fee

        return DryRunFill(
            amount=filled_amount,
            price=price,
            fee=fee,
            fee_currency="USDT",
            slippage=0,
            latency_ms=latency,
            timestamp=time.time(),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        avg_slippage = (
            self.stats['total_slippage'] / self.stats['fills']
            if self.stats['fills'] > 0 else 0
        )

        return {
            **self.stats,
            'avg_slippage': avg_slippage,
            'exchange': self.exchange,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Dry Run Executor Demo")
    print("=" * 50)

    # Test different exchanges
    exchanges = ['binance', 'kraken', 'coinbase']

    for exchange in exchanges:
        executor = DryRunExecutor(exchange=exchange)

        print(f"\n{exchange.upper()}:")

        # Simulate market buy
        fill = executor.simulate_fill(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            price=42000.0,
            order_type="market",
        )

        print(f"  Market Buy 0.5 BTC:")
        print(f"    Price: {fill.price:.2f} (slippage: {fill.slippage*100:.4f}%)")
        print(f"    Fee: ${fill.fee:.2f}")
        print(f"    Latency: {fill.latency_ms:.1f}ms")

        # Simulate limit sell
        fill = executor.simulate_fill(
            symbol="BTC/USDT",
            side="sell",
            amount=0.5,
            price=42500.0,
            order_type="limit",
        )

        print(f"  Limit Sell 0.5 BTC:")
        print(f"    Price: {fill.price:.2f}")
        print(f"    Fee: ${fill.fee:.2f}")
