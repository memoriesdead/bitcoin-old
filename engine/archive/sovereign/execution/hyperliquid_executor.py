"""
Hyperliquid DEX Executor
========================

Professional-grade executor for Hyperliquid perpetuals.
No KYC, up to 50x leverage, US accessible.

Setup:
  pip install hyperliquid-python-sdk eth-account

Usage:
  executor = HyperliquidExecutor(private_key="0x...")
  executor.execute_signal(direction=1, confidence=0.8)
"""

import os
import time
import json
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Hyperliquid SDK
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    HAS_HYPERLIQUID = True
except ImportError:
    HAS_HYPERLIQUID = False
    Info = None
    Exchange = None
    constants = None

# Ethereum account for signing
try:
    from eth_account import Account
    HAS_ETH_ACCOUNT = True
except ImportError:
    HAS_ETH_ACCOUNT = False
    Account = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class HyperliquidConfig:
    """Hyperliquid configuration."""
    private_key: str
    mode: ExecutionMode = ExecutionMode.PAPER

    # Trading parameters
    symbol: str = "BTC"
    leverage: int = 10
    max_position_usd: float = 1000.0

    # Risk management
    stop_loss_pct: float = 0.003      # 0.3%
    take_profit_pct: float = 0.006    # 0.6%
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 0.10  # 10%
    cooldown_after_loss: int = 5      # trades

    # Signal thresholds
    min_confidence: float = 0.6
    strong_signal_threshold: float = 0.8


@dataclass
class Position:
    """Current position state."""
    size: float = 0.0           # Positive = long, negative = short
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    liquidation_price: float = 0.0


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order_id: Optional[str] = None
    side: str = ""
    size: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    volume: float = 0.0
    fees: float = 0.0
    consecutive_losses: int = 0
    date: str = ""


class HyperliquidExecutor:
    """
    Professional Hyperliquid perpetuals executor.

    Connects blockchain signals to DEX execution.
    """

    def __init__(self, config: HyperliquidConfig):
        """
        Initialize executor.

        Args:
            config: Hyperliquid configuration
        """
        self.config = config
        self.info: Optional[Info] = None
        self.exchange: Optional[Exchange] = None
        self.account_address: Optional[str] = None

        # State
        self.position = Position()
        self.daily_stats = DailyStats(date=time.strftime("%Y-%m-%d"))

        # Callbacks
        self.on_trade: Optional[Callable[[TradeResult], None]] = None
        self.on_position_update: Optional[Callable[[Position], None]] = None

        # Initialize
        self._check_dependencies()
        self._init_client()

    def _check_dependencies(self):
        """Check required dependencies."""
        if not HAS_HYPERLIQUID:
            raise ImportError(
                "Hyperliquid SDK not installed. Run:\n"
                "  pip install hyperliquid-python-sdk"
            )
        if not HAS_ETH_ACCOUNT:
            raise ImportError(
                "eth-account not installed. Run:\n"
                "  pip install eth-account"
            )

    def _init_client(self):
        """Initialize Hyperliquid client."""
        if self.config.mode == ExecutionMode.PAPER:
            logger.info("Running in PAPER mode - no real trades")
            return

        # Get account from private key
        account = Account.from_key(self.config.private_key)
        self.account_address = account.address

        # Select network
        if self.config.mode == ExecutionMode.TESTNET:
            base_url = constants.TESTNET_API_URL
            logger.info(f"Connecting to Hyperliquid TESTNET")
        else:
            base_url = constants.MAINNET_API_URL
            logger.info(f"Connecting to Hyperliquid MAINNET")

        # Initialize info client (read-only)
        self.info = Info(base_url, skip_ws=True)

        # Initialize exchange client (trading)
        self.exchange = Exchange(
            account,
            base_url,
            account_address=self.account_address
        )

        logger.info(f"Connected as: {self.account_address}")

        # Set leverage
        self._set_leverage()

    def _set_leverage(self):
        """Set leverage for trading."""
        if self.exchange is None:
            return

        try:
            # Hyperliquid uses cross margin by default
            # Leverage is per-asset
            self.exchange.update_leverage(
                self.config.leverage,
                self.config.symbol,
                is_cross=True
            )
            logger.info(f"Leverage set to {self.config.leverage}x for {self.config.symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")

    def get_account_state(self) -> Dict[str, Any]:
        """Get current account state."""
        if self.config.mode == ExecutionMode.PAPER:
            return {
                'equity': 100.0,
                'available': 100.0,
                'margin_used': 0.0,
                'positions': []
            }

        if self.info is None:
            return {}

        try:
            state = self.info.user_state(self.account_address)
            return {
                'equity': float(state.get('marginSummary', {}).get('accountValue', 0)),
                'available': float(state.get('marginSummary', {}).get('availableBalance', 0)),
                'margin_used': float(state.get('marginSummary', {}).get('totalMarginUsed', 0)),
                'positions': state.get('assetPositions', [])
            }
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return {}

    def get_position(self, symbol: str = None) -> Position:
        """Get current position for symbol."""
        symbol = symbol or self.config.symbol

        if self.config.mode == ExecutionMode.PAPER:
            return self.position

        if self.info is None:
            return Position()

        try:
            state = self.info.user_state(self.account_address)
            positions = state.get('assetPositions', [])

            for pos in positions:
                if pos.get('position', {}).get('coin') == symbol:
                    position_data = pos.get('position', {})
                    return Position(
                        size=float(position_data.get('szi', 0)),
                        entry_price=float(position_data.get('entryPx', 0)),
                        unrealized_pnl=float(position_data.get('unrealizedPnl', 0)),
                        leverage=int(position_data.get('leverage', {}).get('value', 1)),
                        liquidation_price=float(position_data.get('liquidationPx', 0) or 0)
                    )

            return Position()

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return Position()

    def get_price(self, symbol: str = None) -> float:
        """Get current mid price."""
        symbol = symbol or self.config.symbol

        if self.config.mode == ExecutionMode.PAPER:
            return 100000.0  # Placeholder

        if self.info is None:
            return 0.0

        try:
            # Get all mids
            mids = self.info.all_mids()
            return float(mids.get(symbol, 0))
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0

    def execute_signal(self,
                       direction: int,
                       confidence: float,
                       signal_data: Optional[Dict] = None) -> TradeResult:
        """
        Execute a trading signal.

        Args:
            direction: +1 = LONG, -1 = SHORT, 0 = CLOSE/NEUTRAL
            confidence: Signal confidence [0, 1]
            signal_data: Additional signal metadata

        Returns:
            TradeResult
        """
        # Check daily limits
        if not self._check_daily_limits():
            return TradeResult(
                success=False,
                error="Daily limits reached"
            )

        # Check confidence threshold
        if abs(direction) > 0 and confidence < self.config.min_confidence:
            return TradeResult(
                success=False,
                error=f"Confidence {confidence:.2f} below threshold {self.config.min_confidence}"
            )

        # Get current state
        current_price = self.get_price()
        current_position = self.get_position()
        account_state = self.get_account_state()

        if current_price <= 0:
            return TradeResult(success=False, error="Could not get price")

        # Determine action
        if direction == 0:
            # Close position
            if abs(current_position.size) > 0:
                return self._close_position(current_position, current_price)
            else:
                return TradeResult(success=True, error="No position to close")

        elif direction == 1:  # LONG
            if current_position.size < 0:
                # Close short first
                self._close_position(current_position, current_price)
            return self._open_long(confidence, current_price, account_state)

        elif direction == -1:  # SHORT
            if current_position.size > 0:
                # Close long first
                self._close_position(current_position, current_price)
            return self._open_short(confidence, current_price, account_state)

        return TradeResult(success=False, error=f"Invalid direction: {direction}")

    def _calculate_position_size(self,
                                  confidence: float,
                                  price: float,
                                  available: float) -> float:
        """Calculate position size based on confidence and account."""
        # Base size from config
        base_usd = min(self.config.max_position_usd, available * 0.9)

        # Scale by confidence
        if confidence >= self.config.strong_signal_threshold:
            size_usd = base_usd
        else:
            # Linear scale from 50% to 100% based on confidence
            scale = 0.5 + 0.5 * (confidence - self.config.min_confidence) / (
                self.config.strong_signal_threshold - self.config.min_confidence
            )
            size_usd = base_usd * scale

        # Convert to BTC
        size_btc = size_usd / price

        # Round to Hyperliquid precision (4 decimals for BTC)
        size_btc = round(size_btc, 4)

        return size_btc

    def _open_long(self,
                   confidence: float,
                   price: float,
                   account: Dict) -> TradeResult:
        """Open a long position."""
        available = account.get('available', 0)
        size = self._calculate_position_size(confidence, price, available)

        if size <= 0:
            return TradeResult(success=False, error="Position size too small")

        logger.info(f"Opening LONG: {size} BTC @ ~{price:.2f} (conf: {confidence:.2f})")

        return self._place_order(
            is_buy=True,
            size=size,
            price=price,
            reduce_only=False
        )

    def _open_short(self,
                    confidence: float,
                    price: float,
                    account: Dict) -> TradeResult:
        """Open a short position."""
        available = account.get('available', 0)
        size = self._calculate_position_size(confidence, price, available)

        if size <= 0:
            return TradeResult(success=False, error="Position size too small")

        logger.info(f"Opening SHORT: {size} BTC @ ~{price:.2f} (conf: {confidence:.2f})")

        return self._place_order(
            is_buy=False,
            size=size,
            price=price,
            reduce_only=False
        )

    def _close_position(self,
                        position: Position,
                        price: float) -> TradeResult:
        """Close existing position."""
        if abs(position.size) <= 0:
            return TradeResult(success=True, error="No position to close")

        # Close is opposite direction
        is_buy = position.size < 0
        size = abs(position.size)

        logger.info(f"Closing position: {size} BTC @ ~{price:.2f}")

        return self._place_order(
            is_buy=is_buy,
            size=size,
            price=price,
            reduce_only=True
        )

    def _place_order(self,
                     is_buy: bool,
                     size: float,
                     price: float,
                     reduce_only: bool = False) -> TradeResult:
        """
        Place order on Hyperliquid.

        Uses aggressive limit order (crosses spread slightly).
        """
        # Paper mode
        if self.config.mode == ExecutionMode.PAPER:
            return self._simulate_order(is_buy, size, price)

        if self.exchange is None:
            return TradeResult(success=False, error="Exchange not initialized")

        try:
            # Aggressive limit: cross the spread by 0.1%
            if is_buy:
                limit_price = price * 1.001
            else:
                limit_price = price * 0.999

            # Round price to appropriate precision
            limit_price = round(limit_price, 1)  # BTC usually $0.1 precision

            # Place order
            result = self.exchange.order(
                self.config.symbol,
                is_buy,
                size,
                limit_price,
                {"limit": {"tif": "Ioc"}},  # Immediate or cancel
                reduce_only=reduce_only
            )

            # Parse result
            if result.get('status') == 'ok':
                response = result.get('response', {})
                statuses = response.get('data', {}).get('statuses', [{}])

                if statuses and statuses[0].get('filled'):
                    filled = statuses[0].get('filled', {})
                    return TradeResult(
                        success=True,
                        order_id=str(filled.get('oid', '')),
                        side="buy" if is_buy else "sell",
                        size=float(filled.get('totalSz', size)),
                        price=float(filled.get('avgPx', limit_price)),
                        fee=float(filled.get('fee', 0)),
                    )
                elif statuses and statuses[0].get('resting'):
                    # Order resting, not filled
                    resting = statuses[0].get('resting', {})
                    return TradeResult(
                        success=True,
                        order_id=str(resting.get('oid', '')),
                        side="buy" if is_buy else "sell",
                        size=0,  # Not filled yet
                        price=limit_price,
                    )
                else:
                    return TradeResult(
                        success=False,
                        error=f"Order not filled: {statuses}"
                    )
            else:
                return TradeResult(
                    success=False,
                    error=f"Order failed: {result}"
                )

        except Exception as e:
            logger.error(f"Order error: {e}")
            return TradeResult(success=False, error=str(e))

    def _simulate_order(self,
                        is_buy: bool,
                        size: float,
                        price: float) -> TradeResult:
        """Simulate order for paper trading."""
        # Simulate slippage
        if is_buy:
            fill_price = price * 1.0005
        else:
            fill_price = price * 0.9995

        # Simulate fee (0.05% taker)
        fee = size * fill_price * 0.0005

        # Update paper position
        if is_buy:
            self.position.size += size
        else:
            self.position.size -= size

        self.position.entry_price = fill_price

        # Update stats
        self.daily_stats.trades += 1
        self.daily_stats.volume += size * fill_price
        self.daily_stats.fees += fee

        logger.info(f"[PAPER] {'BUY' if is_buy else 'SELL'} {size} @ {fill_price:.2f}")

        return TradeResult(
            success=True,
            order_id=f"PAPER_{int(time.time())}",
            side="buy" if is_buy else "sell",
            size=size,
            price=fill_price,
            fee=fee,
        )

    def set_stop_loss(self, price: float) -> bool:
        """Set stop loss order."""
        if self.config.mode == ExecutionMode.PAPER:
            logger.info(f"[PAPER] Stop loss set at {price:.2f}")
            return True

        if self.exchange is None:
            return False

        try:
            position = self.get_position()
            if abs(position.size) <= 0:
                return False

            # Stop loss is opposite direction
            is_buy = position.size < 0
            size = abs(position.size)

            result = self.exchange.order(
                self.config.symbol,
                is_buy,
                size,
                price,
                {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )

            return result.get('status') == 'ok'

        except Exception as e:
            logger.error(f"Stop loss error: {e}")
            return False

    def set_take_profit(self, price: float) -> bool:
        """Set take profit order."""
        if self.config.mode == ExecutionMode.PAPER:
            logger.info(f"[PAPER] Take profit set at {price:.2f}")
            return True

        if self.exchange is None:
            return False

        try:
            position = self.get_position()
            if abs(position.size) <= 0:
                return False

            is_buy = position.size < 0
            size = abs(position.size)

            result = self.exchange.order(
                self.config.symbol,
                is_buy,
                size,
                price,
                {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": "tp"}},
                reduce_only=True
            )

            return result.get('status') == 'ok'

        except Exception as e:
            logger.error(f"Take profit error: {e}")
            return False

    def execute_with_brackets(self,
                               direction: int,
                               confidence: float) -> Dict[str, TradeResult]:
        """
        Execute signal with automatic SL/TP brackets.

        This is the main entry point for signal execution.
        """
        results = {'entry': None, 'sl': None, 'tp': None}

        # Execute entry
        entry_result = self.execute_signal(direction, confidence)
        results['entry'] = entry_result

        if not entry_result.success or entry_result.size <= 0:
            return results

        # Calculate bracket prices
        entry_price = entry_result.price

        if direction == 1:  # LONG
            sl_price = entry_price * (1 - self.config.stop_loss_pct)
            tp_price = entry_price * (1 + self.config.take_profit_pct)
        else:  # SHORT
            sl_price = entry_price * (1 + self.config.stop_loss_pct)
            tp_price = entry_price * (1 - self.config.take_profit_pct)

        # Set brackets
        sl_success = self.set_stop_loss(round(sl_price, 1))
        tp_success = self.set_take_profit(round(tp_price, 1))

        results['sl'] = TradeResult(success=sl_success, price=sl_price)
        results['tp'] = TradeResult(success=tp_success, price=tp_price)

        logger.info(
            f"Position opened: Entry={entry_price:.2f}, "
            f"SL={sl_price:.2f}, TP={tp_price:.2f}"
        )

        return results

    def _check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        # Reset stats if new day
        today = time.strftime("%Y-%m-%d")
        if self.daily_stats.date != today:
            self.daily_stats = DailyStats(date=today)

        # Check trade count
        if self.daily_stats.trades >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_stats.trades}")
            return False

        # Check consecutive losses
        if self.daily_stats.consecutive_losses >= self.config.cooldown_after_loss:
            logger.warning(f"Consecutive loss limit: {self.daily_stats.consecutive_losses}")
            return False

        # Check daily loss
        account = self.get_account_state()
        if account.get('equity', 0) > 0:
            daily_loss_pct = -self.daily_stats.pnl / account['equity']
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                logger.warning(f"Daily loss limit: {daily_loss_pct:.1%}")
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        account = self.get_account_state()
        position = self.get_position()

        return {
            'mode': self.config.mode.value,
            'account': account,
            'position': {
                'size': position.size,
                'entry_price': position.entry_price,
                'unrealized_pnl': position.unrealized_pnl,
                'leverage': position.leverage,
            },
            'daily': {
                'trades': self.daily_stats.trades,
                'wins': self.daily_stats.wins,
                'losses': self.daily_stats.losses,
                'pnl': self.daily_stats.pnl,
                'volume': self.daily_stats.volume,
                'fees': self.daily_stats.fees,
            },
            'config': {
                'symbol': self.config.symbol,
                'leverage': self.config.leverage,
                'max_position_usd': self.config.max_position_usd,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
            }
        }


# =============================================================================
# SIGNAL BRIDGE - Connect to Sovereign Engine
# =============================================================================

class SignalBridge:
    """
    Bridge between Sovereign signal engine and Hyperliquid executor.

    Listens for signals and executes trades.
    """

    def __init__(self, executor: HyperliquidExecutor):
        self.executor = executor
        self.last_signal_time = 0
        self.min_signal_interval = 60  # Minimum seconds between signals

    def process_signal(self, signal: Dict) -> Optional[TradeResult]:
        """
        Process a signal from the engine.

        Expected signal format:
        {
            'direction': 1/-1/0,
            'confidence': 0.0-1.0,
            'source': 'ensemble',
            'components': {...}
        }
        """
        # Rate limit
        now = time.time()
        if now - self.last_signal_time < self.min_signal_interval:
            return None

        direction = signal.get('direction', 0)
        confidence = signal.get('confidence', 0)

        # Skip neutral signals
        if direction == 0:
            return None

        # Execute with brackets
        results = self.executor.execute_with_brackets(direction, confidence)

        if results['entry'] and results['entry'].success:
            self.last_signal_time = now

        return results.get('entry')


# =============================================================================
# MAIN - Run standalone or import
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYPERLIQUID EXECUTOR")
    print("=" * 60)

    # Check dependencies
    print(f"\nDependencies:")
    print(f"  hyperliquid-python-sdk: {'OK' if HAS_HYPERLIQUID else 'MISSING'}")
    print(f"  eth-account: {'OK' if HAS_ETH_ACCOUNT else 'MISSING'}")

    if not HAS_HYPERLIQUID or not HAS_ETH_ACCOUNT:
        print("\nInstall missing dependencies:")
        print("  pip install hyperliquid-python-sdk eth-account")
        exit(1)

    # Demo in paper mode
    print("\n" + "=" * 60)
    print("PAPER TRADING DEMO")
    print("=" * 60)

    config = HyperliquidConfig(
        private_key="0x" + "0" * 64,  # Dummy key for paper
        mode=ExecutionMode.PAPER,
        leverage=10,
        max_position_usd=100.0,
    )

    executor = HyperliquidExecutor(config)

    # Simulate signals
    print("\n[1] Simulating LONG signal (confidence: 0.8)")
    result = executor.execute_with_brackets(direction=1, confidence=0.8)
    print(f"    Entry: {result['entry']}")

    print(f"\n[2] Current stats:")
    stats = executor.get_stats()
    print(f"    Position: {stats['position']}")
    print(f"    Daily: {stats['daily']}")

    print("\n[3] Closing position")
    result = executor.execute_signal(direction=0, confidence=1.0)
    print(f"    Result: {result}")

    print("\n" + "=" * 60)
    print("To run LIVE, set:")
    print("  mode=ExecutionMode.LIVE")
    print("  private_key=<your wallet private key>")
    print("=" * 60)
