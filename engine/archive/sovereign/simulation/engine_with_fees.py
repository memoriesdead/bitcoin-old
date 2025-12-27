"""
Engine with Real Fee Accounting.

Deducts actual exchange fees from every trade.
True 1:1 simulation.
"""

import time
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

from .types import FormulaSignal, Position, TradeResult, ExitReason
from .formula_engine import ProductionFormulaEngine, PRODUCTION_FORMULA_IDS
from .trade_logger import TradeLogger
from .verifier import ExchangeVerifier, HistoricalVerifier
from .historical import HistoricalReplayer, HistoricalTick
from .live import LivePaperTrader, LiveTick
from .fees import EXCHANGE_FEES, get_slippage_estimate


@dataclass
class EngineConfigWithFees:
    """Engine configuration with fee settings."""
    initial_capital: float = 100.0
    kelly_fraction: float = 0.25
    max_positions: int = 5
    max_position_pct: float = 0.20
    formula_ids: List[int] = None
    db_path: str = "data/simulation_trades.db"

    # Fee settings
    exchange: str = "binance_us"  # Lowest fees
    use_taker_fees: bool = True   # Market orders
    include_slippage: bool = True

    def __post_init__(self):
        if self.formula_ids is None:
            self.formula_ids = PRODUCTION_FORMULA_IDS

    @property
    def fees(self):
        return EXCHANGE_FEES.get(self.exchange, EXCHANGE_FEES['binance_us'])


class SimulationEngineWithFees:
    """
    Simulation engine with real fee accounting.

    Every trade deducts:
    - Entry fee (maker or taker)
    - Exit fee (maker or taker)
    - Slippage estimate

    Target: 50.75% win rate after all costs.
    """

    def __init__(self, config: EngineConfigWithFees = None):
        self.config = config or EngineConfigWithFees()

        # Core components
        self.formula_engine = ProductionFormulaEngine(
            formula_ids=self.config.formula_ids,
            kelly_fraction=self.config.kelly_fraction,
        )
        self.logger = TradeLogger(self.config.db_path)

        # Verifier
        self.verifier = None

        # State
        self.capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.returns: List[float] = []

        # Fee tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        # Tracking
        self.current_price = 0.0
        self.tick_count = 0

    def _calculate_entry_cost(self, position_usd: float) -> Dict:
        """Calculate entry costs (fees + slippage)."""
        fees = self.config.fees

        if self.config.use_taker_fees:
            fee_pct = fees.taker_fee
        else:
            fee_pct = fees.maker_fee

        fee_usd = position_usd * fee_pct

        slippage_pct = 0.0
        slippage_usd = 0.0
        if self.config.include_slippage:
            slippage_pct = get_slippage_estimate(position_usd)
            slippage_usd = position_usd * slippage_pct

        return {
            'fee_pct': fee_pct,
            'fee_usd': fee_usd,
            'slippage_pct': slippage_pct,
            'slippage_usd': slippage_usd,
            'total_cost_usd': fee_usd + slippage_usd,
            'total_cost_pct': fee_pct + slippage_pct,
        }

    def _calculate_exit_cost(self, position_usd: float) -> Dict:
        """Calculate exit costs (fees + slippage)."""
        return self._calculate_entry_cost(position_usd)

    def _process_tick(self, timestamp: float, price: float, data: Dict = None):
        """Process single price tick with fee accounting."""
        self.current_price = price
        self.tick_count += 1

        if isinstance(self.verifier, HistoricalVerifier):
            self.verifier.set_price(price)

        self._check_exits(price, timestamp)

        signals = self.formula_engine.update(price, data)

        for signal in signals:
            self._execute_signal(signal, price, timestamp)

        if self.tick_count % 100 == 0:
            drawdown = ((self.peak_capital - self.capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0
            self.logger.log_equity(self.capital, drawdown)

    def _check_exits(self, price: float, timestamp: float):
        """Check all positions for exit conditions."""
        positions_to_close = []

        for trade_id, position in self.positions.items():
            exit_reason = position.check_exit(price, timestamp)
            if exit_reason:
                positions_to_close.append((trade_id, position, exit_reason))

        for trade_id, position, exit_reason in positions_to_close:
            self._close_position(trade_id, position, price, timestamp, exit_reason.value)

    def _execute_signal(self, signal: FormulaSignal, price: float, timestamp: float):
        """Execute trading signal with fee deduction."""
        if len(self.positions) >= self.config.max_positions:
            return

        for pos in self.positions.values():
            if pos.formula_id == signal.formula_id:
                return

        position_size_pct = min(signal.position_size_pct, self.config.max_position_pct)
        position_usd = self.capital * position_size_pct

        if position_usd < 1:
            return

        # Calculate entry costs
        entry_costs = self._calculate_entry_cost(position_usd)

        # Deduct fees from capital
        self.capital -= entry_costs['fee_usd']
        self.total_fees_paid += entry_costs['fee_usd']
        self.total_slippage_cost += entry_costs['slippage_usd']

        # Apply slippage to entry price
        slippage_mult = 1 + entry_costs['slippage_pct'] if signal.direction == 1 else 1 - entry_costs['slippage_pct']
        entry_price = price * slippage_mult

        position = Position(
            trade_id="",
            formula_id=signal.formula_id,
            formula_name=signal.formula_name,
            direction=signal.direction,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            position_size_pct=position_size_pct,
            position_btc=position_usd / entry_price,
            position_usd=position_usd,
            stop_loss_price=entry_price * (1 - signal.stop_loss_pct) if signal.direction == 1 else entry_price * (1 + signal.stop_loss_pct),
            take_profit_price=entry_price * (1 + signal.take_profit_pct) if signal.direction == 1 else entry_price * (1 - signal.take_profit_pct),
            max_exit_time=timestamp + signal.max_hold_seconds,
            signal_strength=signal.confidence,
        )

        trade_id = self.logger.log_signal(signal, price)
        position.trade_id = trade_id

        self.logger.log_entry(
            trade_id=trade_id,
            entry_price=entry_price,
            position_btc=position.position_btc,
            position_usd=position.position_usd,
            entry_timestamp=timestamp,
        )

        self.positions[trade_id] = position

        print(f"[FEE] {self.config.exchange}: Entry ${entry_costs['fee_usd']:.4f} ({entry_costs['fee_pct']*100:.2f}%)")

    def _close_position(
        self,
        trade_id: str,
        position: Position,
        price: float,
        timestamp: float,
        exit_reason: str
    ):
        """Close position with fee deduction."""
        exit_value = position.position_btc * price
        exit_costs = self._calculate_exit_cost(exit_value)

        # Deduct exit fees
        self.capital -= exit_costs['fee_usd']
        self.total_fees_paid += exit_costs['fee_usd']
        self.total_slippage_cost += exit_costs['slippage_usd']

        # Apply slippage to exit price
        slippage_mult = 1 - exit_costs['slippage_pct'] if position.direction == 1 else 1 + exit_costs['slippage_pct']
        exit_price = price * slippage_mult

        pnl_usd, pnl_pct = position.calculate_pnl(exit_price)
        self.capital += pnl_usd

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        self.logger.log_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_timestamp=timestamp,
        )

        del self.positions[trade_id]

        net_pnl = pnl_usd - exit_costs['fee_usd']
        print(f"[FEE] Exit: ${exit_costs['fee_usd']:.4f} | Gross: ${pnl_usd:.4f} | Net: ${net_pnl:.4f}")

    def run_historical(
        self,
        start_date: str = None,
        end_date: str = None,
        speed: float = 0,
        db_path: str = "data/unified_bitcoin.db"
    ) -> Dict:
        """Run historical simulation with fee accounting."""
        print("="*70)
        print(f"TRUE 1:1 SIMULATION - {self.config.fees.name}")
        print(f"Fees: {self.config.fees.taker_fee*100:.2f}% taker (round-trip: {self.config.fees.round_trip_taker*100:.2f}%)")
        print(f"Capital: ${self.config.initial_capital:.2f}")
        print("="*70)

        self.verifier = HistoricalVerifier()
        replayer = HistoricalReplayer(db_path)

        session = self.logger.create_session(
            mode="historical_with_fees",
            initial_capital=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction,
            formula_ids=self.config.formula_ids,
        )

        def on_tick(tick: HistoricalTick):
            self._process_tick(tick.timestamp, tick.price, tick.to_dict())

        stats = replayer.replay(
            callback=on_tick,
            start_date=start_date,
            end_date=end_date,
            speed=speed,
            progress_interval=500,
        )

        self._close_all_positions(stats.get('last_price', self.current_price), time.time())

        results = self._calculate_results(stats)
        self._print_results(results)

        return results

    def _close_all_positions(self, price: float, timestamp: float):
        for trade_id, position in list(self.positions.items()):
            self._close_position(trade_id, position, price, timestamp, "session_end")

    def _calculate_results(self, stats: Dict) -> Dict:
        session_obj = self.logger.session

        gross_pnl = self.capital - self.config.initial_capital + self.total_fees_paid + self.total_slippage_cost
        net_pnl = self.capital - self.config.initial_capital

        return {
            'initial_capital': self.config.initial_capital,
            'final_capital': self.capital,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'net_pnl_pct': (net_pnl / self.config.initial_capital) * 100,
            'total_fees': self.total_fees_paid,
            'total_slippage': self.total_slippage_cost,
            'total_costs': self.total_fees_paid + self.total_slippage_cost,
            'total_trades': session_obj.total_trades if session_obj else 0,
            'total_wins': session_obj.total_wins if session_obj else 0,
            'win_rate': session_obj.win_rate if session_obj else 0,
            'exchange': self.config.exchange,
            'fee_rate': self.config.fees.taker_fee,
        }

    def _print_results(self, results: Dict):
        print("\n" + "="*70)
        print("TRUE 1:1 RESULTS - WITH ALL COSTS")
        print("="*70)
        print(f"Exchange:        {results['exchange']}")
        print(f"Fee Rate:        {results['fee_rate']*100:.2f}% per trade")
        print(f"\nCapital:")
        print(f"  Initial:       ${results['initial_capital']:.2f}")
        print(f"  Final:         ${results['final_capital']:.2f}")
        print(f"\nPnL Breakdown:")
        print(f"  Gross PnL:     ${results['gross_pnl']:.2f}")
        print(f"  Total Fees:    -${results['total_fees']:.4f}")
        print(f"  Slippage:      -${results['total_slippage']:.4f}")
        print(f"  NET PnL:       ${results['net_pnl']:.2f} ({results['net_pnl_pct']:+.2f}%)")
        print(f"\nTrades:")
        print(f"  Total:         {results['total_trades']}")
        print(f"  Wins:          {results['total_wins']}")
        print(f"  Win Rate:      {results['win_rate']*100:.2f}%" if results['win_rate'] else "  Win Rate: N/A")
        print(f"\nTarget Win Rate: 50.75%")
        print(f"Actual Win Rate: {results['win_rate']*100:.2f}%" if results['win_rate'] else "Actual: N/A")
        if results['win_rate'] and results['win_rate'] > 0.5075:
            print(f"STATUS: BEATING TARGET BY {(results['win_rate'] - 0.5075)*100:.2f}%")
        print("="*70)

    def reset(self):
        self.capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.positions = {}
        self.returns = []
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0
        self.current_price = 0.0
        self.tick_count = 0
        self.formula_engine.reset()
