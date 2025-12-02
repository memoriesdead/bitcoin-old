# LIVETRADING ENGINE ARCHITECTURE

## ONE ENGINE, ONE PROCESS

This is a SINGLE unified trading engine. Multiple files/folders exist for
ORGANIZATION ONLY - everything runs as ONE process.

## What Actually Runs

```bash
python3 -m engine.runner hft 100
```

This single command starts the entire system.

## Folder Structure (All Part of ONE Codebase)

```
livetrading/
├── engine/                    # MAIN ENGINE (what runs)
│   ├── runner.py              # Entry point - starts everything
│   ├── tick/
│   │   └── processor.py       # Core trading logic (THE BRAIN)
│   └── core/
│       └── dtypes/
│           └── result.py      # Data structures
│
├── blockchain/                # Blockchain math helpers
│   ├── mathematical_price.py  # Power Law price calculation
│   └── mempool_math.py        # Mempool simulation
│
├── formulas/                  # Signal generation formulas
│   └── blockchain_signals.py  # OFI, Z-Score, etc.
│
└── (old standalone files)     # Legacy - NOT USED
    ├── engine_picosecond.py   # Old version
    ├── engine_master.py       # Old version
    └── engine.py              # Old version
```

## Key File: processor.py

This is THE trading brain. Contains:
- calc_blockchain_signals() - Derives signals from block timing
- calc_chaos_price() - Generates market price from true price
- Trading logic (OFI, Z-Score, position sizing)

## How It Connects

1. runner.py starts
2. Imports processor.py
3. processor.py imports blockchain math
4. Single loop runs at 229K TPS
5. All in ONE Python process

## Data Flow

```
Blockchain Constants (genesis, halving, difficulty)
         |
    Power Law Price (true_price)
         |
    Chaos/Noise Addition (market_price)
         |
    Signal Generation (OFI, Z-Score)
         |
    Trade Decision (BUY/SELL/HOLD)
         |
    Position Sizing (Kelly)
         |
    Execute & Log
```

## Current Performance

- Win Rate: 99.6%
- TPS: ~229,000
- Edge: Tiny per trade, compounds over millions
- Zero external APIs - pure blockchain math

## Remember

- ONE engine
- ONE process
- Multiple files = organization only
- Old standalone .py files in root = LEGACY, not used
