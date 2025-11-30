# Freqtrade Integration Plan
## Renaissance Trading System + Freqtrade Hybrid Architecture

---

## Executive Summary

This document outlines the careful integration of Freqtrade (45K+ GitHub stars, 321 contributors) into our existing Renaissance Trading System (433 formulas, 62 files). The goal is to leverage Freqtrade's proven backtesting, hyperopt, and exchange connectivity while preserving our custom formula engine and blockchain data capabilities.

---

## Current Architecture (What We Have)

```
livetrading/
├── formulas/           # 433 academic trading formulas
│   ├── base.py         # BaseFormula + FormulaRegistry
│   ├── statistical.py  # IDs 1-30
│   ├── timeseries.py   # IDs 31-60
│   └── ... (20+ formula files)
│
├── engine/
│   ├── live_engine_v1.py      # Production engine (2098 lines)
│   ├── explosive_engine.py    # High-frequency variant
│   └── blockchain_live_engine.py
│
├── blockchain/
│   ├── blockchain_feed.py     # WebSocket feeds
│   ├── blockchain_market_data.py
│   └── real_price_feed.py
│
└── data/
    ├── btc_history.csv
    └── pipeline.py
```

---

## Target Architecture (After Integration)

```
livetrading/
├── vendor/
│   └── freqtrade/             # Cloned Freqtrade (DO NOT MODIFY CORE)
│       ├── freqtrade/
│       │   ├── strategy/
│       │   ├── optimize/      # Hyperopt engine
│       │   ├── data/          # Data providers
│       │   └── exchange/      # Exchange connectors
│       └── requirements.txt
│
├── freqtrade_bridge/          # OUR ADAPTER LAYER (NEW)
│   ├── __init__.py
│   ├── formula_adapter.py     # BaseFormula → Freqtrade indicator
│   ├── blockchain_provider.py # Blockchain data → Freqtrade
│   ├── tick_provider.py       # Tick data support
│   ├── strategy_wrapper.py    # Wraps our formulas as FT strategy
│   └── hyperopt_spaces.py     # Formula parameter optimization
│
├── formulas/                  # UNCHANGED - Our 433 formulas
├── engine/                    # UNCHANGED - Can run independently
├── blockchain/                # UNCHANGED - Our blockchain feeds
└── data/                      # UNCHANGED
```

---

## Phase 1: Clone Freqtrade (Safe Isolation)

### Commands
```bash
cd C:\Users\kevin\livetrading
mkdir vendor
cd vendor
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable  # Use stable branch, not develop
```

### Why vendor/ Directory?
- **Isolation**: Keeps Freqtrade code separate from our code
- **Upgradeable**: Can `git pull` updates without conflicts
- **No Modification**: We adapt TO Freqtrade, not modify it
- **Clean Separation**: Our code in `freqtrade_bridge/`, their code in `vendor/`

---

## Phase 2: Create Adapter Layer

### File: `freqtrade_bridge/formula_adapter.py`

This adapter converts our BaseFormula interface to Freqtrade's indicator format.

```python
"""
Adapter that converts Renaissance BaseFormula → Freqtrade indicators
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Type
import sys
sys.path.insert(0, '../')  # Access our formulas

from formulas.base import BaseFormula, FORMULA_REGISTRY

class FormulaToIndicator:
    """
    Converts a BaseFormula class to a Freqtrade-compatible indicator function.

    Freqtrade expects: populate_indicators(dataframe) → dataframe with new columns
    Our formulas expect: update(price, volume, timestamp) → signal
    """

    def __init__(self, formula_ids: List[int] = None):
        """
        Args:
            formula_ids: List of formula IDs to use. None = all formulas.
        """
        self.formula_ids = formula_ids or list(FORMULA_REGISTRY.keys())
        self.formula_instances: Dict[int, BaseFormula] = {}
        self._initialize_formulas()

    def _initialize_formulas(self):
        """Create instances of all selected formulas."""
        for fid in self.formula_ids:
            if fid in FORMULA_REGISTRY:
                try:
                    self.formula_instances[fid] = FORMULA_REGISTRY[fid]()
                except Exception as e:
                    print(f"Warning: Could not instantiate formula {fid}: {e}")

    def compute_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataframe through our formulas.

        Args:
            dataframe: Freqtrade OHLCV dataframe with columns:
                       [date, open, high, low, close, volume]

        Returns:
            dataframe with additional columns for each formula signal
        """
        # Initialize signal columns
        for fid in self.formula_instances:
            dataframe[f'formula_{fid}'] = 0.0

        # Process each row (simulating live data flow)
        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx]
            price = row['close']
            volume = row['volume']
            timestamp = row['date'].timestamp() if hasattr(row['date'], 'timestamp') else 0

            # Update each formula and capture signal
            for fid, formula in self.formula_instances.items():
                try:
                    signal = formula.update(price, volume, timestamp)
                    dataframe.loc[dataframe.index[idx], f'formula_{fid}'] = signal or 0.0
                except Exception:
                    pass  # Formula failed, keep 0

        return dataframe

    def get_aggregated_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute aggregated signal from all formulas (Condorcet voting).

        Returns:
            dataframe with 'agg_signal' column (-1 to 1)
        """
        formula_cols = [c for c in dataframe.columns if c.startswith('formula_')]

        if formula_cols:
            # Simple average (can be enhanced with weighted voting)
            dataframe['agg_signal'] = dataframe[formula_cols].mean(axis=1)
        else:
            dataframe['agg_signal'] = 0.0

        return dataframe
```

### File: `freqtrade_bridge/strategy_wrapper.py`

```python
"""
Freqtrade Strategy that wraps our Renaissance formulas.
"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import sys
sys.path.insert(0, '../')

from freqtrade_bridge.formula_adapter import FormulaToIndicator

class RenaissanceStrategy(IStrategy):
    """
    Freqtrade strategy powered by 433 Renaissance formulas.
    """

    # Strategy settings
    INTERFACE_VERSION = 3
    timeframe = '1m'
    stoploss = -0.02  # 2% stop loss

    # Our formula adapter
    formula_adapter = None

    # Which formula IDs to use (None = all 433)
    formula_ids = None  # Set to list like [1, 5, 101, 259] to filter

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.formula_adapter = FormulaToIndicator(self.formula_ids)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Run all Renaissance formulas on the dataframe."""
        dataframe = self.formula_adapter.compute_signals(dataframe)
        dataframe = self.formula_adapter.get_aggregated_signal(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate buy signals from aggregated formula output."""
        dataframe.loc[
            (dataframe['agg_signal'] > 0.3),  # Threshold tunable via hyperopt
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate sell signals."""
        dataframe.loc[
            (dataframe['agg_signal'] < -0.1),
            'exit_long'
        ] = 1
        return dataframe
```

---

## Phase 3: Blockchain Data Provider

### File: `freqtrade_bridge/blockchain_provider.py`

```python
"""
Injects blockchain data (mempool, fees, blocks) into Freqtrade dataframes.
"""
import asyncio
from typing import Dict, Any
import sys
sys.path.insert(0, '../')

from blockchain.blockchain_feed import BlockchainFeed
from blockchain.blockchain_market_data import BlockchainMarketData

class BlockchainDataProvider:
    """
    Provides blockchain signals as additional dataframe columns.

    Freqtrade doesn't natively support blockchain data, so we inject it
    as custom indicators during populate_indicators().
    """

    def __init__(self):
        self.feed = BlockchainFeed()
        self.market_data = BlockchainMarketData()
        self._cache: Dict[str, Any] = {}

    async def start(self):
        """Start blockchain data feeds."""
        await self.feed.start()

    def get_current_metrics(self) -> Dict[str, float]:
        """Get latest blockchain metrics."""
        return {
            'mempool_size': self.market_data.mempool_size,
            'fee_rate_fast': self.market_data.fee_rate_fast,
            'fee_rate_medium': self.market_data.fee_rate_medium,
            'block_height': self.market_data.block_height,
            'hashrate': self.market_data.hashrate,
            'difficulty': self.market_data.difficulty,
            'mempool_tx_count': self.market_data.mempool_tx_count,
        }

    def enrich_dataframe(self, dataframe, metadata: dict):
        """
        Add blockchain columns to Freqtrade dataframe.

        Note: For backtesting, we'd need historical blockchain data.
        For live trading, we inject real-time values.
        """
        metrics = self.get_current_metrics()

        # Add as constant columns (real-time value)
        for key, value in metrics.items():
            dataframe[f'blockchain_{key}'] = value

        return dataframe
```

---

## Phase 4: Tick-Level Data Support

### Challenge
Freqtrade operates on candles (1m minimum). Our HFT formulas need tick data.

### Solution: Tick Aggregation Layer

```python
"""
freqtrade_bridge/tick_provider.py

Aggregates tick data into features that can be used in candle-based backtesting.
"""
import numpy as np
from collections import deque
from typing import List, Tuple

class TickAggregator:
    """
    Collects ticks within a candle period and computes HFT-relevant features.

    Features computed per candle:
    - tick_count: Number of ticks in candle
    - tick_imbalance: Buy vs sell pressure
    - microprice: Volume-weighted mid
    - tick_volatility: Price variance of ticks
    - max_drawdown_ticks: Intra-candle drawdown
    """

    def __init__(self, candle_seconds: int = 60):
        self.candle_seconds = candle_seconds
        self.current_ticks: List[Tuple[float, float, float]] = []  # (price, volume, timestamp)

    def add_tick(self, price: float, volume: float, timestamp: float):
        """Add a tick to current candle accumulator."""
        self.current_ticks.append((price, volume, timestamp))

    def flush_candle(self) -> dict:
        """
        Compute tick-derived features for the candle and reset.

        Returns dict of features to add to dataframe row.
        """
        if not self.current_ticks:
            return self._empty_features()

        prices = np.array([t[0] for t in self.current_ticks])
        volumes = np.array([t[1] for t in self.current_ticks])

        features = {
            'tick_count': len(self.current_ticks),
            'tick_volatility': np.std(prices) if len(prices) > 1 else 0,
            'tick_range': np.max(prices) - np.min(prices),
            'vwap_tick': np.average(prices, weights=volumes) if volumes.sum() > 0 else prices.mean(),
            'volume_concentration': np.max(volumes) / volumes.sum() if volumes.sum() > 0 else 0,
        }

        # Reset for next candle
        self.current_ticks = []
        return features

    def _empty_features(self) -> dict:
        return {
            'tick_count': 0,
            'tick_volatility': 0,
            'tick_range': 0,
            'vwap_tick': 0,
            'volume_concentration': 0,
        }
```

---

## Phase 5: Hyperopt Integration

### File: `freqtrade_bridge/hyperopt_spaces.py`

```python
"""
Hyperopt parameter spaces for Renaissance formulas.

This allows Freqtrade's optimizer to tune formula thresholds.
"""
from freqtrade.optimize.space import Categorical, Dimension, Integer, Real

def renaissance_hyperopt_space() -> list:
    """
    Define tunable parameters for our strategy.
    """
    return [
        # Signal threshold for entry
        Real(0.1, 0.5, name='entry_threshold', default=0.3),

        # Signal threshold for exit
        Real(-0.3, 0.0, name='exit_threshold', default=-0.1),

        # Number of formulas to use (sampling)
        Integer(50, 433, name='formula_count', default=200),

        # Minimum formula agreement
        Real(0.3, 0.8, name='min_agreement', default=0.5),

        # Position sizing (Kelly fraction)
        Real(0.1, 0.5, name='kelly_fraction', default=0.25),
    ]
```

---

## Phase 6: Configuration Files

### File: `freqtrade_bridge/config.json`

```json
{
    "strategy": "RenaissanceStrategy",
    "strategy_path": "../freqtrade_bridge/",

    "max_open_trades": 1,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",

    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",

    "dry_run": true,
    "dry_run_wallet": 10000,

    "exchange": {
        "name": "kraken",
        "key": "",
        "secret": "",
        "ccxt_config": {},
        "ccxt_async_config": {}
    },

    "pairlists": [
        {"method": "StaticPairList"}
    ],

    "pair_whitelist": [
        "BTC/USDT"
    ],

    "telegram": {
        "enabled": false
    },

    "bot_name": "Renaissance_Freqtrade",

    "initial_state": "running",

    "internals": {
        "process_throttle_secs": 5
    }
}
```

---

## Phase 7: Testing Commands

### Backtest with our formulas
```bash
cd vendor/freqtrade
python -m freqtrade backtesting \
    --config ../../freqtrade_bridge/config.json \
    --strategy RenaissanceStrategy \
    --timeframe 1m \
    --timerange 20240101-20241201
```

### Hyperopt optimization
```bash
python -m freqtrade hyperopt \
    --config ../../freqtrade_bridge/config.json \
    --strategy RenaissanceStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces all \
    --epochs 1000
```

### Paper trading
```bash
python -m freqtrade trade \
    --config ../../freqtrade_bridge/config.json \
    --strategy RenaissanceStrategy \
    --dry-run
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing code | All new code in `freqtrade_bridge/`, no modifications to `formulas/` or `engine/` |
| Freqtrade updates breaking integration | Use stable branch, vendor directory allows easy updates |
| Formula computation slow in backtest | Use numpy vectorization, optional numba JIT |
| Missing historical blockchain data | For backtesting, use proxy metrics; live trading uses real data |
| HFT formulas not validatable | Tick aggregator computes derived features per candle |

---

## Execution Checklist

- [ ] Phase 1: Clone Freqtrade to vendor/
- [ ] Phase 2: Create freqtrade_bridge/formula_adapter.py
- [ ] Phase 2: Create freqtrade_bridge/strategy_wrapper.py
- [ ] Phase 3: Create freqtrade_bridge/blockchain_provider.py
- [ ] Phase 4: Create freqtrade_bridge/tick_provider.py
- [ ] Phase 5: Create freqtrade_bridge/hyperopt_spaces.py
- [ ] Phase 6: Create freqtrade_bridge/config.json
- [ ] Phase 7: Run first backtest
- [ ] Phase 7: Run hyperopt with 100 epochs
- [ ] Validate results match expectations

---

## Timeline-Free Next Steps

1. Execute Phase 1 (clone)
2. Create adapter files
3. Run basic backtest to verify integration
4. Iterate on formula selection
5. Run hyperopt to find optimal parameters
6. Compare results to current live_engine_v1 performance
