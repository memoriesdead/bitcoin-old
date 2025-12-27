#!/usr/bin/env python3
"""
Sovereign Engine - Unified CLI Entry Point
===========================================

Single entry point for all trading modes:
    python -m engine.sovereign.run_sovereign --mode paper --capital 10000
    python -m engine.sovereign.run_sovereign --mode dry_run --engines rentech
    python -m engine.sovereign.run_sovereign --mode live --capital 5000

Usage:
    --mode          Execution mode: paper, dry_run, live, onchain
    --capital       Initial capital in USD
    --duration      Trading duration in seconds (0 = infinite)
    --engines       Comma-separated engines: adaptive,pattern,rentech
    --config        Path to config file
    --telegram      Enable Telegram notifications
    --verbose       Verbose output
"""
import argparse
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sovereign Engine - Unified Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Paper trading with default settings:
    python -m engine.sovereign.run_sovereign --mode paper

  Dry run with RenTech engine only:
    python -m engine.sovereign.run_sovereign --mode dry_run --engines rentech

  Live trading (requires API keys):
    python -m engine.sovereign.run_sovereign --mode live --capital 5000

  On-chain (Solana):
    python -m engine.sovereign.run_sovereign --mode onchain --chain solana
"""
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["paper", "dry_run", "live", "onchain", "backtest"],
        default="paper",
        help="Execution mode (default: paper)"
    )

    # Capital
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000)"
    )

    # Duration
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=3600,
        help="Trading duration in seconds (0 = infinite, default: 3600)"
    )

    # Engines
    parser.add_argument(
        "--engines", "-e",
        type=str,
        default="adaptive,pattern,rentech",
        help="Comma-separated engines to enable (default: all)"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )

    # Exchanges
    parser.add_argument(
        "--exchanges",
        type=str,
        default="binance",
        help="Comma-separated exchanges (default: binance)"
    )

    # Chain (for on-chain mode)
    parser.add_argument(
        "--chain",
        type=str,
        choices=["solana", "polygon", "base", "ethereum"],
        default="solana",
        help="Blockchain for on-chain execution (default: solana)"
    )

    # Telegram
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Telegram notifications"
    )

    # Verbose
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    # Claude AI
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude AI for signal validation and trade confirmation"
    )

    # Symbols
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT",
        help="Comma-separated trading symbols (default: BTC/USDT)"
    )

    # Data source
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["blockchain", "historical", "live", "simulated"],
        default="simulated",
        help="Data source (default: simulated)"
    )

    return parser.parse_args()


def build_config(args):
    """Build SovereignConfig from command line arguments."""
    from engine.sovereign.core.config import (
        SovereignConfig, load_config,
        DataConfig, EnginesConfig, EngineConfig,
        ExecutionConfig, TelegramConfig, LoggingConfig
    )
    from engine.sovereign.core.types import ExecutionMode, DataSource

    # Load base config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = SovereignConfig()

    # Override with CLI args
    config.initial_capital = args.capital
    config.duration_seconds = args.duration

    # Execution mode
    mode_map = {
        "paper": ExecutionMode.PAPER,
        "dry_run": ExecutionMode.DRY_RUN,
        "live": ExecutionMode.LIVE,
        "onchain": ExecutionMode.ONCHAIN,
        "backtest": ExecutionMode.BACKTEST,
    }
    config.execution.mode = mode_map[args.mode]

    # Exchanges
    config.execution.exchanges = args.exchanges.split(",")
    config.execution.default_exchange = args.exchanges.split(",")[0]

    # Data source
    source_map = {
        "blockchain": DataSource.BLOCKCHAIN,
        "historical": DataSource.HISTORICAL,
        "live": DataSource.LIVE,
        "simulated": DataSource.SIMULATED,
    }
    config.data.source = source_map[args.data_source]
    config.data.symbols = args.symbols.split(",")

    # Engines
    enabled_engines = args.engines.split(",")
    config.engines.adaptive.enabled = "adaptive" in enabled_engines
    config.engines.pattern.enabled = "pattern" in enabled_engines
    config.engines.rentech.enabled = "rentech" in enabled_engines
    config.engines.qlib.enabled = "qlib" in enabled_engines

    # Telegram
    if args.telegram:
        config.telegram.enabled = True
        # Load token/chat_id from environment
        config.telegram.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        config.telegram.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Logging
    config.logging.verbose = args.verbose

    # Claude AI
    if args.claude:
        config.claude.enabled = True
        config.claude.model = "sonnet"
        config.claude.validate_signals = True
        config.claude.confirm_trades = True
        config.claude.risk_assessment = True
        print("[Config] Claude AI enabled (Sonnet model)")

    return config


def print_banner():
    """Print startup banner."""
    banner = """
+=========================================================================+
|                                                                         |
|   SOVEREIGN ENGINE - Renaissance-Grade Bitcoin Trading                  |
|                                                                         |
|   * Adaptive Formulas (10001-10005)                                     |
|   * Pattern Recognition (20001-20012)                                   |
|   * RenTech Patterns (72001-72099)                                      |
|   * 900+ Total Formulas with Ensemble Voting                            |
|                                                                         |
+=========================================================================+
"""
    print(banner)


def print_config_summary(config, args):
    """Print configuration summary."""
    print(f"\n{'-'*60}")
    print(f"CONFIGURATION")
    print(f"{'-'*60}")
    print(f"  Mode:        {args.mode.upper()}")
    print(f"  Capital:     ${config.initial_capital:,.2f}")
    print(f"  Duration:    {config.duration_seconds}s ({config.duration_seconds/3600:.1f}h)")
    print(f"  Engines:     {args.engines}")
    print(f"  Data:        {args.data_source}")
    print(f"  Exchanges:   {args.exchanges}")
    print(f"  Symbols:     {args.symbols}")
    print(f"  Telegram:    {'Enabled' if config.telegram.enabled else 'Disabled'}")
    print(f"  Claude AI:   {'Enabled (Sonnet)' if config.claude.enabled else 'Disabled'}")
    print(f"{'-'*60}\n")


def confirm_live_trading(config):
    """Confirm before live trading."""
    if config.execution.mode.value == "live":
        print("\n" + "="*60)
        print("!!! WARNING: LIVE TRADING MODE !!!")
        print("="*60)
        print(f"Capital at risk: ${config.initial_capital:,.2f}")
        print(f"Max position:    ${config.safety.max_position_usd:,.2f}")
        print(f"Max daily loss:  ${config.safety.max_daily_loss_usd:,.2f}")
        print("="*60)

        response = input("\nType 'CONFIRM' to proceed with live trading: ")
        if response.strip() != "CONFIRM":
            print("Live trading cancelled.")
            sys.exit(0)

        print("Live trading confirmed. Starting...\n")


def main():
    """Main entry point."""
    print_banner()

    args = parse_args()
    config = build_config(args)

    print_config_summary(config, args)

    # Confirm live trading
    confirm_live_trading(config)

    # Import and run engine
    from engine.sovereign.core.sovereign_engine import SovereignEngine

    engine = SovereignEngine(config)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print final stats
    stats = engine.get_stats()
    print(f"\nFinal PnL: ${stats.total_pnl:+,.2f} ({stats.return_pct:+.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
