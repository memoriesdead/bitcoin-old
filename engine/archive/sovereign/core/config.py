"""
Unified Configuration - Sovereign Engine
==========================================

All configuration dataclasses for the unified engine.
Loaded from config/sovereign.json or passed programmatically.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .types import ExecutionMode, DataSource


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    source: DataSource = DataSource.BLOCKCHAIN
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    # Historical data
    historical_db: str = "data/bitcoin_2021_2025.db"
    features_db: str = "data/bitcoin_features.db"
    flows_db: str = "data/exchange_flows_2022_2025.db"

    # Real-time
    websocket_exchanges: List[str] = field(default_factory=lambda: ["binance", "coinbase"])
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])

    # Replay settings
    replay_speed: float = 1.0      # 1.0 = real-time, 10.0 = 10x speed
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class EngineConfig:
    """Individual engine configuration."""
    enabled: bool = True
    formulas: List[int] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class EnginesConfig:
    """All formula engines configuration."""
    adaptive: EngineConfig = field(default_factory=lambda: EngineConfig(
        enabled=True,
        formulas=[10001, 10002, 10003, 10004, 10005],
        weight=0.3
    ))
    pattern: EngineConfig = field(default_factory=lambda: EngineConfig(
        enabled=True,
        formulas=[20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012],
        weight=0.3
    ))
    rentech: EngineConfig = field(default_factory=lambda: EngineConfig(
        enabled=True,
        formulas=[],  # All 72001-72099
        weight=0.4
    ))
    qlib: EngineConfig = field(default_factory=lambda: EngineConfig(
        enabled=False,
        formulas=[70001, 70002, 70003, 70004, 70005],
        weight=0.0
    ))


@dataclass
class EnsembleConfig:
    """Ensemble voting configuration."""
    method: str = "weighted_vote"  # "weighted_vote", "majority", "confidence_weighted"
    min_agreement: int = 2         # Minimum engines that must agree
    confidence_threshold: float = 0.5
    boost_unanimous: float = 1.5   # Boost when all engines agree
    boost_majority: float = 1.3    # Boost when majority agrees


@dataclass
class SafetyConfig:
    """Risk management and safety limits."""
    # Position limits
    max_position_usd: float = 1000.0
    max_position_pct: float = 0.10  # 10% of capital
    max_exposure_usd: float = 5000.0

    # Daily limits
    max_daily_trades: int = 50
    max_daily_loss_usd: float = 200.0
    max_daily_loss_pct: float = 0.10  # 10% of capital

    # Drawdown
    max_drawdown_pct: float = 0.10
    consecutive_loss_limit: int = 5

    # Kill switch
    kill_switch_enabled: bool = True
    kill_switch_loss_pct: float = 0.15  # 15% triggers kill switch

    # Time restrictions
    trading_hours_only: bool = False
    blackout_hours: List[int] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    mode: ExecutionMode = ExecutionMode.PAPER

    # Exchange settings
    exchanges: List[str] = field(default_factory=lambda: ["binance", "coinbase"])
    default_exchange: str = "binance"

    # API credentials (loaded from env)
    api_key: str = ""
    api_secret: str = ""

    # Order settings
    default_order_type: str = "market"
    max_slippage_pct: float = 0.01  # 1%

    # Paper/Dry run settings
    paper_slippage_pct: float = 0.0001  # 0.01%
    paper_fee_pct: float = 0.001        # 0.1%

    # On-chain settings
    onchain_enabled: bool = False
    chain: str = "solana"               # "solana", "polygon", "base"
    wallet_address: str = ""
    rpc_endpoint: str = ""


@dataclass
class RLConfig:
    """RL position sizing configuration."""
    enabled: bool = False
    agent: str = "sac"                  # "sac", "ppo"
    model_path: Optional[str] = None

    # Training
    train_on_outcomes: bool = True
    min_samples_for_training: int = 100
    retrain_interval: int = 1000        # Trades between retraining

    # Position sizing bounds
    min_position_pct: float = 0.01      # 1%
    max_position_pct: float = 0.10      # 10%

    # Kelly fallback
    kelly_enabled: bool = True
    kelly_fraction: float = 0.25        # Quarter Kelly


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    enabled: bool = False
    token: str = ""
    chat_id: str = ""
    allowed_chat_ids: List[str] = field(default_factory=list)

    # Notification settings
    notify_trades: bool = True
    notify_errors: bool = True
    notify_daily_summary: bool = True


@dataclass
class ClaudeConfig:
    """Claude AI integration configuration."""
    enabled: bool = False              # Master switch
    model: str = "sonnet"              # Model: "sonnet" (fast) or "opus" (powerful)
    validate_signals: bool = True      # Signal validation after ensemble
    confirm_trades: bool = True        # Trade confirmation before execution
    risk_assessment: bool = True       # Dynamic risk/Kelly adjustment
    market_context: bool = False       # Market regime analysis (slower)

    timeout: int = 5                   # CLI timeout in seconds
    fallback_on_timeout: bool = True   # Proceed without validation on timeout
    log_responses: bool = True         # Log all Claude responses

    min_confidence_for_claude: float = 0.5  # Only call Claude above this confidence


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = "logs/sovereign.log"
    log_trades: bool = True
    log_signals: bool = True
    verbose: bool = False


@dataclass
class SovereignConfig:
    """
    Master configuration for the Sovereign Engine.

    Aggregates all sub-configurations.
    """
    # Capital
    initial_capital: float = 10000.0

    # Duration
    duration_seconds: int = 3600        # Default 1 hour

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_json(cls, path: str) -> "SovereignConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SovereignConfig":
        """Create config from dictionary."""
        config = cls()

        # Top-level
        config.initial_capital = data.get("initial_capital", config.initial_capital)
        config.duration_seconds = data.get("duration_seconds", config.duration_seconds)

        # Data config
        if "data" in data:
            d = data["data"]
            config.data.source = DataSource(d.get("source", "blockchain"))
            config.data.zmq_endpoint = d.get("zmq_endpoint", config.data.zmq_endpoint)
            config.data.symbols = d.get("symbols", config.data.symbols)

        # Engines config
        if "engines" in data:
            for engine_name in ["adaptive", "pattern", "rentech", "qlib"]:
                if engine_name in data["engines"]:
                    e = data["engines"][engine_name]
                    engine_cfg = getattr(config.engines, engine_name)
                    engine_cfg.enabled = e.get("enabled", engine_cfg.enabled)
                    engine_cfg.formulas = e.get("formulas", engine_cfg.formulas)
                    engine_cfg.weight = e.get("weight", engine_cfg.weight)

        # Ensemble config
        if "ensemble" in data:
            e = data["ensemble"]
            config.ensemble.method = e.get("method", config.ensemble.method)
            config.ensemble.min_agreement = e.get("min_agreement", config.ensemble.min_agreement)
            config.ensemble.confidence_threshold = e.get("confidence_threshold", config.ensemble.confidence_threshold)

        # Safety config
        if "safety" in data:
            s = data["safety"]
            config.safety.max_position_usd = s.get("max_position_usd", config.safety.max_position_usd)
            config.safety.max_daily_loss_pct = s.get("max_daily_loss_pct", config.safety.max_daily_loss_pct)
            config.safety.max_drawdown_pct = s.get("max_drawdown_pct", config.safety.max_drawdown_pct)
            config.safety.consecutive_loss_limit = s.get("consecutive_loss_limit", config.safety.consecutive_loss_limit)

        # Execution config
        if "execution" in data:
            e = data["execution"]
            mode_str = e.get("mode", "paper")
            config.execution.mode = ExecutionMode(mode_str)
            config.execution.exchanges = e.get("exchanges", config.execution.exchanges)
            config.execution.default_exchange = e.get("default_exchange", config.execution.default_exchange)

        # RL config
        if "rl" in data:
            r = data["rl"]
            config.rl.enabled = r.get("enabled", config.rl.enabled)
            config.rl.agent = r.get("agent", config.rl.agent)
            config.rl.max_position_pct = r.get("max_position", config.rl.max_position_pct)

        # Telegram config
        if "telegram" in data:
            t = data["telegram"]
            config.telegram.enabled = t.get("enabled", config.telegram.enabled)
            config.telegram.token = t.get("token", config.telegram.token)
            config.telegram.chat_id = t.get("chat_id", config.telegram.chat_id)

        # Logging config
        if "logging" in data:
            l = data["logging"]
            config.logging.level = l.get("level", config.logging.level)
            config.logging.verbose = l.get("verbose", config.logging.verbose)

        # Claude AI config
        if "claude" in data:
            c = data["claude"]
            config.claude.enabled = c.get("enabled", config.claude.enabled)
            config.claude.validate_signals = c.get("validate_signals", config.claude.validate_signals)
            config.claude.confirm_trades = c.get("confirm_trades", config.claude.confirm_trades)
            config.claude.risk_assessment = c.get("risk_assessment", config.claude.risk_assessment)
            config.claude.market_context = c.get("market_context", config.claude.market_context)
            config.claude.timeout = c.get("timeout", config.claude.timeout)
            config.claude.fallback_on_timeout = c.get("fallback_on_timeout", config.claude.fallback_on_timeout)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "duration_seconds": self.duration_seconds,
            "data": {
                "source": self.data.source.value,
                "zmq_endpoint": self.data.zmq_endpoint,
                "symbols": self.data.symbols,
            },
            "engines": {
                "adaptive": {
                    "enabled": self.engines.adaptive.enabled,
                    "formulas": self.engines.adaptive.formulas,
                    "weight": self.engines.adaptive.weight,
                },
                "pattern": {
                    "enabled": self.engines.pattern.enabled,
                    "formulas": self.engines.pattern.formulas,
                    "weight": self.engines.pattern.weight,
                },
                "rentech": {
                    "enabled": self.engines.rentech.enabled,
                    "formulas": self.engines.rentech.formulas,
                    "weight": self.engines.rentech.weight,
                },
            },
            "ensemble": {
                "method": self.ensemble.method,
                "min_agreement": self.ensemble.min_agreement,
                "confidence_threshold": self.ensemble.confidence_threshold,
            },
            "safety": {
                "max_position_usd": self.safety.max_position_usd,
                "max_daily_loss_pct": self.safety.max_daily_loss_pct,
                "max_drawdown_pct": self.safety.max_drawdown_pct,
            },
            "execution": {
                "mode": self.execution.mode.value,
                "exchanges": self.execution.exchanges,
            },
            "rl": {
                "enabled": self.rl.enabled,
                "agent": self.rl.agent,
            },
            "claude": {
                "enabled": self.claude.enabled,
                "validate_signals": self.claude.validate_signals,
                "confirm_trades": self.claude.confirm_trades,
                "risk_assessment": self.claude.risk_assessment,
                "market_context": self.claude.market_context,
                "timeout": self.claude.timeout,
            },
        }

    def save(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def load_config(path: Optional[str] = None) -> SovereignConfig:
    """
    Load configuration from file or create default.

    Args:
        path: Path to config file. If None, uses default location.

    Returns:
        SovereignConfig instance
    """
    if path is None:
        path = "engine/sovereign/config/sovereign.json"

    if Path(path).exists():
        return SovereignConfig.from_json(path)
    else:
        # Return default config
        return SovereignConfig()


def create_paper_config(capital: float = 10000.0) -> SovereignConfig:
    """Create configuration for paper trading."""
    config = SovereignConfig()
    config.initial_capital = capital
    config.execution.mode = ExecutionMode.PAPER
    return config


def create_live_config(
    capital: float,
    api_key: str,
    api_secret: str,
    exchange: str = "binance"
) -> SovereignConfig:
    """Create configuration for live trading."""
    config = SovereignConfig()
    config.initial_capital = capital
    config.execution.mode = ExecutionMode.LIVE
    config.execution.api_key = api_key
    config.execution.api_secret = api_secret
    config.execution.default_exchange = exchange

    # Stricter safety for live
    config.safety.max_position_usd = min(1000.0, capital * 0.05)
    config.safety.max_daily_loss_pct = 0.05
    config.safety.kill_switch_enabled = True

    return config
