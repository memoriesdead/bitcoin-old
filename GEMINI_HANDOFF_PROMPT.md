# Gemini 2.5 Pro - Trading Engine Restructure Prompt

## CONTEXT

I have a Bitcoin HFT trading engine running on a VPS (31.97.211.217) that achieved 100% win rate on LONG and SHORT signals. The codebase has grown to 1,178 Python files (4.5GB) and needs serious restructuring to reduce complexity while preserving the working logic.

**DO NOT DELETE CODE** - Archive deprecated files, restructure and redesign for clarity.

---

## THE PROVEN EDGE (This logic works - preserve it exactly)

```
INFLOW to exchange  → Deposit to SELL → Price DOWN → SHORT
OUTFLOW from exchange → Buyer withdrawal → Price UP → LONG

Pattern is tradeable when:
  samples >= 10 AND correlation >= 0.7 AND win_rate >= 0.9
```

---

## CURRENT ARCHITECTURE (Working but messy)

### Signal Flow
```
Bitcoin Core ZMQ → C++ Runner (8μs latency) → Python Bridge → CCXT Exchange API
```

### VPS Location: /root/sovereign/

**Core Components (KEEP):**
```
cpp_runner/
├── build/blockchain_runner    # C++ binary - DO NOT MODIFY
├── deterministic_bridge.py    # Main trading bridge (works)
├── exchange_flow_tracker.py   # Flow detection
└── signal_bridge.py           # Signal parsing

blockchain/
├── config.py                  # Configuration
├── correlation_formula.py     # Signal generation
├── cpp_master_pipeline.py     # Orchestration
├── deterministic_trader.py    # Position management
└── multi_price_feed.py        # 12 exchange prices via CCXT
```

**Databases (KEEP):**
```
walletexplorer_addresses.db    # 8.6M exchange addresses
correlation.db                 # Flow→price patterns
trades.db                      # Trade history
addresses.bin                  # Compiled address cache for C++
```

**Deprecated (ARCHIVE):**
- 1,100+ other Python files
- Multiple duplicate implementations
- Experimental strategies that didn't work

---

## TRADING CONFIGURATION

```python
initial_capital = 100.0      # USD
max_leverage = 125
max_positions = 8
position_size_pct = 0.125    # 12.5% per position
exit_timeout_seconds = 300   # 5 min
stop_loss_pct = 0.01         # 1%
take_profit_pct = 0.02       # 2%
```

---

## SIGNAL TYPES (From C++ Runner)

```python
SHORT_INTERNAL  # Exchange consolidating internally → about to sell → SHORT
LONG_EXTERNAL   # Customer withdrawal → already bought → LONG
INFLOW_SHORT    # Deposit to exchange → about to sell → SHORT
```

---

## YOUR TASK

### 1. Restructure to This Clean Architecture:

```
sovereign/
├── README.md                    # Documentation
├── config.py                    # Single config file
├── run.py                       # Entry point
│
├── core/                        # Core logic (< 5 files)
│   ├── __init__.py
│   ├── signals.py               # Signal dataclasses
│   ├── trader.py                # Position management
│   ├── price_feed.py            # CCXT price feeds
│   └── correlation.py           # Pattern matching
│
├── bridge/                      # C++ integration
│   ├── __init__.py
│   ├── cpp_runner.py            # C++ process management
│   └── signal_parser.py         # Parse C++ output
│
├── cpp_runner/                  # C++ code (unchanged)
│   ├── build/
│   ├── src/
│   └── include/
│
├── data/                        # Databases
│   ├── addresses.bin
│   ├── walletexplorer_addresses.db
│   ├── correlation.db
│   └── trades.db
│
└── archive/                     # All deprecated code
    └── (move 1,100+ files here)
```

### 2. Requirements:

- **Total Python code: < 2,500 lines** across all active files
- **Single entry point:** `python3 run.py --paper` or `python3 run.py --live`
- **Preserve exact signal logic** - the INFLOW/OUTFLOW edge works
- **Keep C++ runner unchanged** - just clean Python wrapper
- **Use relative imports** within packages
- **Cross-platform compatibility** (runs on Linux VPS and Windows dev)

### 3. Key Classes to Preserve:

```python
@dataclass
class Signal:
    timestamp: datetime
    signal_type: str      # SHORT_INTERNAL, LONG_EXTERNAL, INFLOW_SHORT
    action: str           # SHORT or LONG
    source: str
    dest_exchanges: list
    txid: str
    latency_ns: int

@dataclass
class Position:
    entry_time: datetime
    direction: str        # SHORT or LONG
    entry_price: float
    size_usd: float
    signal_type: str
    txid: str

class Trader:
    def open_position(self, signal: Signal, price: float) -> Position
    def check_exits(self, current_price: float) -> list[Position]
    def close_position(self, position: Position, price: float, reason: str)
```

### 4. File-by-File Instructions:

| Current File | Action |
|--------------|--------|
| cpp_runner/deterministic_bridge.py | Refactor → core/trader.py + bridge/signal_parser.py |
| cpp_runner/exchange_flow_tracker.py | Refactor → core/signals.py |
| blockchain/config.py | Keep → config.py (root) |
| blockchain/correlation_formula.py | Refactor → core/correlation.py |
| blockchain/cpp_master_pipeline.py | Refactor → run.py |
| blockchain/deterministic_trader.py | Merge → core/trader.py |
| blockchain/multi_price_feed.py | Refactor → core/price_feed.py |
| Everything else in blockchain/ | Move → archive/ |
| Everything else in /root/sovereign/ | Move → archive/ |

---

## VERIFICATION

After restructure, these must work:

```bash
# Paper trading
python3 run.py --paper

# Check structure
find . -name "*.py" -not -path "./archive/*" | wc -l  # Should be < 15 files

# Line count
find . -name "*.py" -not -path "./archive/*" -exec wc -l {} + | tail -1  # Should be < 2500 lines
```

---

## DO NOT

- Delete any code (archive instead)
- Modify C++ runner source
- Change the signal detection logic
- Add new features
- Over-engineer with abstractions
- Create unnecessary class hierarchies

---

## IMPORTANT CONTEXT

The trading edge WORKS when patterns meet criteria. Recent tests showed 42% win rate because the system was trading ALL signals instead of only high-correlation patterns. The fix is in correlation.py - only trade when:

```python
if pattern.sample_count >= 10 and pattern.correlation >= 0.7 and pattern.win_rate >= 0.9:
    execute_trade(signal)
```

This filter was missing/disabled. Ensure it's enforced in the restructured code.

---

## START

Begin by:
1. SSH to root@31.97.211.217
2. List all files: `find /root/sovereign -name "*.py" | head -100`
3. Create archive folder: `mkdir -p /root/sovereign/archive`
4. Move deprecated files systematically
5. Consolidate core logic into clean structure
6. Test with `python3 run.py --paper`
