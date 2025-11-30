# RENAISSANCE TRADING SYSTEM v2.0

## Pure Blockchain HFT Trading Engine

### NO Third-Party Exchange APIs
All data from Bitcoin blockchain: mempool.space, blockstream.info, blockchain.info

### Folder Structure
```
organized/
├── blockchain/          # Pure blockchain data pipeline
│   ├── blockchain_feed.py         # WebSocket/REST endpoints
│   ├── blockchain_market_data.py  # Trading signals
│   └── blockchain_price_engine.py # Price derivation
│
├── engine/              # Trading engines
│   ├── live_engine_v1.py          # Main production engine
│   ├── blockchain_live_engine.py  # Pure blockchain engine
│   └── hft/                       # Numba JIT engine
│
├── formulas/            # 300+ academic trading formulas
│
├── data/                # Data utilities
│
└── run.py               # Main entry point
```

### Quick Start
```bash
python run.py live 60    # Run 60 second live test
python run.py live 300   # Run 5 minute test
```
