# Claude Code Settings for Renaissance Trading System

## PRIMARY DEVELOPMENT ENVIRONMENT
**ALL code changes, testing, and execution should happen on HOSTINGER:**
- Server: root@31.97.211.217
- Path: /root/livetrading/

## DO NOT:
- Edit files locally in VS Code as the primary action
- Run strategies locally
- Test code locally

## ALWAYS:
- SSH to Hostinger for all file operations
- Execute strategies on Hostinger
- Sync FROM Hostinger if needed (not TO)

## Active Strategy Versions
- V1, V2, V3, V4 only (V1-V4 MAX)

## Key Paths on Hostinger
- Strategies: /root/livetrading/
- Historical Data: /root/livetrading/data/historical/
- Renaissance Module: /root/livetrading/renaissance/

## Credentials
- Located at: /root/livetrading/renaissance/core/credentials.py
- APIs: Kraken, Coinbase, Binance

## Historical Data Foundation
- 1,879,385 candles loaded
- Daily: 2014-2025
- Hourly: 2017-2025
- Minute: 2019-2022
