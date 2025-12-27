"""
Validation Script
=================

Validates the integrated trading system across modes:
1. Unit tests for each component
2. Integration tests
3. Dry-run validation
4. Performance benchmarks

Run: python -m engine.sovereign.validate
"""

import time
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ValidationResult:
    """Result of validation test."""
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = None


class Validator:
    """Validation framework."""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def run_test(self, name: str, test_func) -> ValidationResult:
        """Run a single test."""
        start = time.time()
        try:
            result = test_func()
            passed = result.get('passed', True) if isinstance(result, dict) else bool(result)
            message = result.get('message', 'OK') if isinstance(result, dict) else 'OK'
            details = result if isinstance(result, dict) else None
        except Exception as e:
            passed = False
            message = str(e)
            details = {'exception': type(e).__name__}

        duration = (time.time() - start) * 1000

        result = ValidationResult(
            name=name,
            passed=passed,
            duration_ms=duration,
            message=message,
            details=details,
        )
        self.results.append(result)
        return result

    def print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"{status} {r.name} ({r.duration_ms:.1f}ms)")
            if not r.passed:
                print(f"       {r.message}")

        print("=" * 60)
        print(f"TOTAL: {passed}/{total} passed")

        return passed == total


# =============================================================================
# COMPONENT TESTS
# =============================================================================

def test_pit_handler():
    """Test point-in-time data handler."""
    from .formulas.qlib_alpha import PointInTimeHandler

    handler = PointInTimeHandler(min_delay_seconds=60)

    # Test valid data (old enough)
    old_time = time.time() - 120  # 2 minutes ago
    now = time.time()
    valid = handler.is_valid(old_time, now)

    # Test invalid data (too new)
    recent_time = time.time() - 30  # 30 seconds ago
    invalid = not handler.is_valid(recent_time, now)

    return {
        'passed': valid and invalid,
        'message': 'PIT validation working correctly',
        'details': {'old_valid': valid, 'recent_invalid': invalid}
    }


def test_flow_expressions():
    """Test alpha expressions."""
    from .formulas.qlib_alpha import FlowMomentum, FlowZScore

    # Test FlowMomentum
    momentum = FlowMomentum(window=5)
    flows = [100, 110, 120, 115, 125, 130]
    result = momentum.compute(flows)

    # Test FlowZScore
    zscore = FlowZScore(window=5)
    z_result = zscore.compute(flows)

    return {
        'passed': result is not None and z_result is not None,
        'message': 'Alpha expressions computed successfully',
        'details': {'momentum': result, 'zscore': z_result}
    }


def test_lightgbm_classifier():
    """Test LightGBM classifier."""
    from .formulas.qlib_alpha.lightgbm_flow import LightGBMFlowClassifier

    clf = LightGBMFlowClassifier()

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Train
    clf.train(X, y)

    # Predict
    pred = clf.predict_proba(X[:10])

    return {
        'passed': len(pred) == 10 and all(0 <= p <= 1 for p in pred),
        'message': 'LightGBM classifier working',
        'details': {'predictions': pred[:5].tolist()}
    }


def test_rl_position_sizer():
    """Test RL position sizer."""
    from .formulas.finrl_rl import SACPositionSizer, TradingState

    sizer = SACPositionSizer(state_dim=11)

    # Create test state
    state = TradingState(
        price_change=0.01,
        volatility=0.02,
        volume_ratio=1.2,
        flow_imbalance=0.3,
        position_ratio=0.0,
        unrealized_pnl=0.0,
        drawdown=0.0,
        time_in_position=0.0,
        rsi=55.0,
        macd=0.001,
        signal_strength=0.7,
    )

    # Get position size
    size, conf = sizer.get_position_size(state)

    return {
        'passed': -1 <= size <= 1 and 0 <= conf <= 1,
        'message': 'SAC sizer working',
        'details': {'position_size': size, 'confidence': conf}
    }


def test_ml_enhancer():
    """Test ML enhancer."""
    from .formulas.ml_enhancer import MLEnhancer

    enhancer = MLEnhancer()

    # Test signal
    signal = {
        'direction': 1,
        'confidence': 0.7,
        'flow_imbalance': 0.3,
        'volatility': 0.02,
    }

    result = enhancer.enhance_signal(signal, price=42000.0)

    return {
        'passed': result.direction in [-1, 0, 1] and 0 <= result.confidence <= 1,
        'message': 'ML enhancer working',
        'details': {
            'direction': result.direction,
            'confidence': result.confidence,
            'alpha_count': len(result.features)
        }
    }


def test_order_manager():
    """Test order state machine."""
    from .execution import OrderManager, OrderState, OrderTransition

    manager = OrderManager()

    # Create order
    order = manager.create_order(
        client_order_id="TEST_001",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        amount=0.1,
        price=42000.0,
    )

    # Test transitions
    manager.transition("TEST_001", OrderTransition.SUBMIT)
    submitted = order.state == OrderState.SUBMITTED

    manager.transition("TEST_001", OrderTransition.CONFIRM)
    confirmed = order.state == OrderState.OPEN

    manager.transition("TEST_001", OrderTransition.FILL,
                      fill_amount=0.1, fill_price=42000.0)
    filled = order.state == OrderState.FILLED

    return {
        'passed': submitted and confirmed and filled,
        'message': 'Order state machine working',
        'details': {
            'submitted': submitted,
            'confirmed': confirmed,
            'filled': filled
        }
    }


def test_safety_manager():
    """Test safety mechanisms."""
    from .execution import SafetyManager, SafetyConfig

    config = SafetyConfig(
        max_position_usd=1000,
        max_daily_loss_usd=100,
        consecutive_loss_limit=3,
    )

    safety = SafetyManager(config)
    safety.set_capital(10000)

    # Test valid order
    result1 = safety.check_order("BTC/USDT", "buy", 0.02, 42000)  # $840
    valid_allowed = result1['allowed']

    # Test too large order
    result2 = safety.check_order("BTC/USDT", "buy", 0.05, 42000)  # $2100
    large_blocked = not result2['allowed']

    # Test consecutive losses
    for _ in range(4):
        safety.record_trade(-30, is_win=False)

    result3 = safety.check_order("BTC/USDT", "buy", 0.01, 42000)
    loss_blocked = not result3['allowed']

    return {
        'passed': valid_allowed and large_blocked and loss_blocked,
        'message': 'Safety manager working',
        'details': {
            'valid_allowed': valid_allowed,
            'large_blocked': large_blocked,
            'loss_blocked': loss_blocked
        }
    }


def test_execution_engine():
    """Test execution engine."""
    from .execution import ExecutionEngine, ExecutionMode

    engine = ExecutionEngine(mode=ExecutionMode.PAPER)
    engine.set_price("BTC/USDT", 42000.0)

    # Execute order
    result = engine.execute(
        symbol="BTC/USDT",
        side="buy",
        amount=0.01,
        order_type="market",
    )

    return {
        'passed': result.success and result.executed_amount > 0,
        'message': 'Execution engine working',
        'details': {
            'success': result.success,
            'price': result.executed_price,
            'amount': result.executed_amount,
            'fee': result.fee
        }
    }


def test_dry_run_executor():
    """Test dry run executor."""
    from .execution.dry_run import DryRunExecutor

    executor = DryRunExecutor(exchange='binance')

    fill = executor.simulate_fill(
        symbol="BTC/USDT",
        side="buy",
        amount=0.1,
        price=42000.0,
        order_type="market",
    )

    return {
        'passed': fill.amount > 0 and fill.slippage >= 0,
        'message': 'Dry run executor working',
        'details': {
            'price': fill.price,
            'slippage': fill.slippage,
            'fee': fill.fee,
            'latency_ms': fill.latency_ms
        }
    }


def test_config_manager():
    """Test configuration manager."""
    import os
    from .execution import ConfigManager

    # Set test env vars
    os.environ['TRADING_MODE'] = 'dry_run'
    os.environ['MAX_POSITION_USD'] = '500'

    manager = ConfigManager()
    config = manager.get_trading_config()

    return {
        'passed': config.mode == 'dry_run' and config.safety.max_position_usd == 500,
        'message': 'Config manager working',
        'details': {
            'mode': config.mode,
            'max_position': config.safety.max_position_usd
        }
    }


def test_orderbook_types():
    """Test order book types."""
    from .simulation.orderbook import OrderBookSnapshot, Level, Side

    # Create order book
    book = OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=time.time(),
        bids=[
            Level(price=41990, amount=1.0),
            Level(price=41980, amount=2.0),
        ],
        asks=[
            Level(price=42010, amount=1.0),
            Level(price=42020, amount=2.0),
        ],
    )

    mid = book.get_mid_price()
    spread = book.get_spread()
    imbalance = book.get_imbalance()

    return {
        'passed': 41990 < mid < 42010 and spread > 0,
        'message': 'Order book types working',
        'details': {
            'mid_price': mid,
            'spread': spread,
            'imbalance': imbalance
        }
    }


def test_hft_backtester():
    """Test HFT backtester."""
    from .simulation.orderbook import HFTBacktester, BacktestConfig
    from .simulation.orderbook.loader import InMemoryOrderBookLoader, generate_synthetic_orderbook

    # Create synthetic data
    loader = InMemoryOrderBookLoader()
    price = 42000.0
    for i in range(100):
        price += np.random.normal(0, 5)
        book = generate_synthetic_orderbook("BTCUSDT", price, timestamp=1700000000 + i * 60)
        loader.add_snapshot(book)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000,
        max_position=0.1,
        use_latency=False,
    )

    # Simple strategy
    def strategy(book, state):
        from .simulation.orderbook import Order, Side
        orders = []
        if state.position < 0.05 and book.get_imbalance() > 0.2:
            orders.append(Order(
                order_id="",
                symbol=book.symbol,
                side=Side.BUY,
                order_type="market",
                amount=0.01,
                price=book.get_mid_price(),
                timestamp=book.timestamp,
            ))
        return orders

    backtester = HFTBacktester(config, loader)
    backtester.set_strategy(strategy)
    result = backtester.run("BTCUSDT")

    return {
        'passed': len(result.equity_curve) > 0,
        'message': 'HFT backtester working',
        'details': {
            'total_return': f"{result.total_return*100:.2f}%",
            'trades': result.total_trades,
            'sharpe': result.sharpe_ratio
        }
    }


def test_integration():
    """Test full integration."""
    from .integration import create_trading_system

    system = create_trading_system(mode="paper")
    system.start()

    # Send test signal
    signal = {
        'direction': 1,
        'confidence': 0.75,
        'flow_imbalance': 0.3,
        'volatility': 0.02,
    }

    result = system.on_flow_signal(signal, price=42000.0)

    # Get stats
    stats = system.get_stats()

    system.stop()

    return {
        'passed': stats['is_running'] == False and result is not None,
        'message': 'Integration working',
        'details': {
            'signal_direction': result.final_direction if result else None,
            'components': stats['components']
        }
    }


# =============================================================================
# MAIN
# =============================================================================

def run_validation():
    """Run all validation tests."""
    print("RenTech Integration Validation")
    print("=" * 60)

    validator = Validator()

    # Component tests
    tests = [
        ("PIT Handler", test_pit_handler),
        ("Flow Expressions", test_flow_expressions),
        ("LightGBM Classifier", test_lightgbm_classifier),
        ("RL Position Sizer", test_rl_position_sizer),
        ("ML Enhancer", test_ml_enhancer),
        ("Order Manager", test_order_manager),
        ("Safety Manager", test_safety_manager),
        ("Execution Engine", test_execution_engine),
        ("Dry Run Executor", test_dry_run_executor),
        ("Config Manager", test_config_manager),
        ("Order Book Types", test_orderbook_types),
        ("HFT Backtester", test_hft_backtester),
        ("Full Integration", test_integration),
    ]

    for name, test_func in tests:
        print(f"Testing {name}...", end=" ", flush=True)
        result = validator.run_test(name, test_func)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.duration_ms:.1f}ms)")

    # Print summary
    all_passed = validator.print_results()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_validation())
