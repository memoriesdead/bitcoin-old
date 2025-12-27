#!/usr/bin/env python3
"""
SELECTIVE PAPER TRADER - Only trades patterns with 100% historical accuracy.

Based on correlation data analysis:
- coinbase OUTFLOW 10-20 BTC: 100% (LONG when BTC leaves)
- gemini INFLOW 20-50 BTC: 100% (SHORT on deposits)
- huobi/gate.io INFLOW >20 BTC: 100%
- kraken INFLOW 10-20 BTC: 88.9%
- gemini INFLOW 10-20 BTC: 82.4%

ONLY TRADES THESE HIGH-ACCURACY PATTERNS.
"""

import os
import sys
import time
import sqlite3
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# High-accuracy patterns from correlation analysis
HIGH_ACCURACY_PATTERNS = {
    # (exchange, direction, min_btc, max_btc): historical_accuracy
    ('coinbase', 'OUTFLOW', 10, 20): 1.00,    # 5/5 = 100%
    ('gemini', 'INFLOW', 20, 50): 1.00,       # 2/2 = 100%
    ('gemini', 'INFLOW', 10, 20): 0.824,      # 14/17 = 82.4%
    ('kraken', 'INFLOW', 10, 20): 0.889,      # 16/18 = 88.9%
    ('huobi', 'INFLOW', 20, 1000): 1.00,      # 3/3 = 100%
    ('gate.io', 'INFLOW', 50, 1000): 1.00,    # 2/2 = 100%
}

# Only trade patterns with >= 85% historical accuracy
MIN_ACCURACY = 0.85


@dataclass
class Position:
    id: int
    exchange: str
    direction: str
    entry_price: float
    entry_time: float
    size_usd: float
    size_btc: float
    stop_loss: float
    take_profit: float
    pattern: str
    expected_accuracy: float


class SelectiveTrader:
    def __init__(self, capital: float = 400.0):
        self.capital = capital
        self.initial_capital = capital
        self.positions: Dict[int, Position] = {}
        self.next_id = 1
        self.trades: List[Dict] = []
        self.lock = threading.Lock()

        # Fees
        self.fees = {
            'coinbase': 0.006,
            'gemini': 0.004,
            'kraken': 0.0026,
            'huobi': 0.002,
            'gate.io': 0.002,
            'default': 0.001
        }

        print('=' * 70)
        print('SELECTIVE PAPER TRADER - High-Accuracy Patterns Only')
        print('=' * 70)
        print(f'Capital: ${capital:.2f}')
        print(f'Min Accuracy: {MIN_ACCURACY*100:.0f}%')
        print('Patterns:')
        for (ex, dir, min_btc, max_btc), acc in HIGH_ACCURACY_PATTERNS.items():
            if acc >= MIN_ACCURACY:
                print(f'  {ex} {dir} {min_btc}-{max_btc} BTC: {acc*100:.1f}%')
        print('=' * 70)
        print()

    def check_pattern(self, exchange: str, direction: str, amount_btc: float) -> Optional[float]:
        """Check if flow matches a high-accuracy pattern. Returns accuracy or None."""
        for (ex, dir, min_btc, max_btc), accuracy in HIGH_ACCURACY_PATTERNS.items():
            if ex == exchange.lower() and dir == direction:
                if min_btc <= amount_btc <= max_btc:
                    if accuracy >= MIN_ACCURACY:
                        return accuracy
        return None

    def on_signal(self, exchange: str, direction: str, amount_btc: float,
                  price: float, flow_type: str):
        """
        Handle signal - only trade if matches high-accuracy pattern.

        flow_type: 'INFLOW' or 'OUTFLOW'
        direction: 'LONG' or 'SHORT'
        """
        with self.lock:
            # Check if this matches a high-accuracy pattern
            accuracy = self.check_pattern(exchange, flow_type, amount_btc)

            if accuracy is None:
                return  # Skip - not a high-accuracy pattern

            # Skip if already have position on this exchange
            for pos in self.positions.values():
                if pos.exchange == exchange:
                    return

            # Calculate position size (25% of capital)
            size_usd = self.capital * 0.25
            size_btc = size_usd / price

            # Entry with slippage (0.03%)
            slippage = 0.0003
            if direction == 'LONG':
                entry_price = price * (1 + slippage)
                stop_loss = entry_price * 0.98  # 2% SL
                take_profit = entry_price * 1.04  # 4% TP
            else:
                entry_price = price * (1 - slippage)
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96

            # Entry fee
            fee = size_usd * self.fees.get(exchange, self.fees['default'])
            self.capital -= fee

            # Create position
            pos = Position(
                id=self.next_id,
                exchange=exchange,
                direction=direction,
                entry_price=entry_price,
                entry_time=time.time(),
                size_usd=size_usd,
                size_btc=size_btc,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pattern=f'{flow_type} {amount_btc:.1f} BTC',
                expected_accuracy=accuracy
            )

            self.positions[pos.id] = pos
            self.next_id += 1

            print(f'\n\033[92m[OPEN]\033[0m {direction} {exchange.upper()}')
            print(f'       Pattern: {flow_type} {amount_btc:.1f} BTC (historical: {accuracy*100:.0f}%)')
            print(f'       Entry: ${entry_price:,.2f} | Size: ${size_usd:.2f}')
            print(f'       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}')

    def update_prices(self, prices: Dict[str, float]):
        """Update positions with current prices."""
        with self.lock:
            now = time.time()

            for pos_id, pos in list(self.positions.items()):
                price = prices.get(pos.exchange)
                if not price:
                    continue

                # Check SL/TP
                reason = None
                if pos.direction == 'LONG':
                    if price <= pos.stop_loss:
                        reason = 'SL'
                    elif price >= pos.take_profit:
                        reason = 'TP'
                else:
                    if price >= pos.stop_loss:
                        reason = 'SL'
                    elif price <= pos.take_profit:
                        reason = 'TP'

                # Timeout after 5 min
                if now - pos.entry_time >= 300:
                    reason = 'TIMEOUT'

                if reason:
                    self._close_position(pos, price, reason)

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close position."""
        # Apply exit slippage
        slippage = 0.0003
        if pos.direction == 'LONG':
            exit_price = exit_price * (1 - slippage)
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            exit_price = exit_price * (1 + slippage)
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pnl_pct * pos.size_usd

        # Exit fee
        fee = pos.size_usd * self.fees.get(pos.exchange, self.fees['default'])
        net_pnl = pnl_usd - fee

        self.capital += net_pnl

        # Record trade
        self.trades.append({
            'exchange': pos.exchange,
            'direction': pos.direction,
            'entry': pos.entry_price,
            'exit': exit_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'reason': reason,
            'pattern': pos.pattern,
            'expected_acc': pos.expected_accuracy
        })

        # Remove position
        del self.positions[pos.id]

        # Print
        color = '\033[92m' if net_pnl > 0 else '\033[91m'
        print(f'\n{color}[CLOSE]\033[0m {pos.direction} {pos.exchange.upper()} - {reason}')
        print(f'        Entry: ${pos.entry_price:,.2f} -> Exit: ${exit_price:,.2f}')
        print(f'        P&L: {color}${net_pnl:+.2f}\033[0m ({pnl_pct*100:+.2f}%)')
        print(f'        Capital: ${self.capital:,.2f}')

    def print_stats(self):
        """Print trading stats."""
        total_pnl = self.capital - self.initial_capital
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        losses = len(self.trades) - wins
        win_rate = wins / len(self.trades) * 100 if self.trades else 0

        print('\n' + '=' * 70)
        print('SELECTIVE TRADER STATS')
        print('=' * 70)
        print(f'Capital: ${self.capital:.2f} (started: ${self.initial_capital:.2f})')
        print(f'P&L: ${total_pnl:+.2f} ({total_pnl/self.initial_capital*100:+.2f}%)')
        print(f'Trades: {len(self.trades)} | Wins: {wins} | Losses: {losses}')
        print(f'Win Rate: {win_rate:.1f}%')
        print('=' * 70)


# Integration with master pipeline
if __name__ == '__main__':
    from master_exchange_pipeline import MasterExchangePipeline, PipelineConfig
    from multi_price_feed import MultiExchangePriceFeed

    # Create trader
    trader = SelectiveTrader(capital=400.0)

    # Price feed
    price_feed = MultiExchangePriceFeed()
    price_feed.start()

    # Create pipeline with callback
    config = PipelineConfig(short_only=False)  # Enable both LONG and SHORT
    pipeline = MasterExchangePipeline(config)

    # Override signal callback
    original_print_signal = pipeline._print_signal

    def on_signal(signal):
        original_print_signal(signal)

        # Convert to flow type
        flow_type = 'INFLOW' if signal.direction == 'SHORT' else 'OUTFLOW'

        # Send to selective trader
        trader.on_signal(
            exchange=signal.exchange,
            direction=signal.direction,
            amount_btc=signal.amount_btc,
            price=signal.price,
            flow_type=flow_type
        )

    pipeline._print_signal = on_signal

    # Price update loop
    def update_loop():
        while True:
            prices = price_feed.get_all_prices()
            trader.update_prices(prices)
            time.sleep(1)

    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()

    # Stats loop
    def stats_loop():
        while True:
            time.sleep(60)
            trader.print_stats()

    stats_thread = threading.Thread(target=stats_loop, daemon=True)
    stats_thread.start()

    # Run pipeline
    try:
        pipeline.run()
    except KeyboardInterrupt:
        trader.print_stats()
        print('\nShutting down...')
