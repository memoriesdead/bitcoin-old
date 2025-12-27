"""
Configuration Manager
====================

Secure configuration management.

Sources (in priority order):
1. Environment variables
2. .env file
3. config.json
4. Defaults

Freqtrade pattern: Never hardcode secrets.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""
    exchange_id: str
    api_key: str
    secret: str
    password: Optional[str] = None
    sandbox: bool = False


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    token: str
    chat_id: str
    enabled: bool = True

    # Command permissions
    allow_stop: bool = True
    allow_trade: bool = False  # Dangerous - disabled by default

    # Notification settings
    notify_trades: bool = True
    notify_errors: bool = True
    notify_daily_summary: bool = True


@dataclass
class SafetyLimits:
    """Trading safety limits."""
    max_position_usd: float = 1000.0
    max_total_exposure_usd: float = 5000.0
    max_daily_trades: int = 100
    max_daily_loss_usd: float = 200.0
    max_daily_loss_pct: float = 0.10
    consecutive_loss_limit: int = 5
    max_drawdown_pct: float = 0.10


@dataclass
class TradingConfig:
    """Full trading configuration."""
    # Mode
    mode: str = "paper"  # paper, dry_run, live

    # Exchange
    exchange: Optional[ExchangeCredentials] = None

    # Telegram
    telegram: Optional[TelegramConfig] = None

    # Safety
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    # Trading params
    symbols: list = field(default_factory=lambda: ["BTC/USDT"])
    base_position_size: float = 0.01  # BTC

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


class ConfigManager:
    """
    Configuration manager.

    Loads config from multiple sources with secure defaults.
    """

    # Environment variable mappings
    ENV_MAPPINGS = {
        # Exchange
        'EXCHANGE_ID': 'exchange.exchange_id',
        'EXCHANGE_API_KEY': 'exchange.api_key',
        'EXCHANGE_SECRET': 'exchange.secret',
        'EXCHANGE_PASSWORD': 'exchange.password',
        'EXCHANGE_SANDBOX': 'exchange.sandbox',

        # Telegram
        'TELEGRAM_TOKEN': 'telegram.token',
        'TELEGRAM_CHAT_ID': 'telegram.chat_id',
        'TELEGRAM_ENABLED': 'telegram.enabled',

        # Trading
        'TRADING_MODE': 'mode',
        'TRADING_SYMBOLS': 'symbols',
        'POSITION_SIZE': 'base_position_size',

        # Safety
        'MAX_POSITION_USD': 'safety.max_position_usd',
        'MAX_EXPOSURE_USD': 'safety.max_total_exposure_usd',
        'MAX_DAILY_LOSS': 'safety.max_daily_loss_usd',
    }

    def __init__(self, config_path: Optional[str] = None,
                 env_file: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to config.json
            env_file: Path to .env file
        """
        self.config_path = config_path
        self.env_file = env_file
        self._config: Dict[str, Any] = {}

        # Load in order
        self._load_defaults()
        if config_path:
            self._load_json(config_path)
        if env_file:
            self._load_env_file(env_file)
        self._load_environment()

    def _load_defaults(self):
        """Load default configuration."""
        self._config = {
            'mode': 'paper',
            'symbols': ['BTC/USDT'],
            'base_position_size': 0.01,
            'log_level': 'INFO',
            'exchange': {},
            'telegram': {},
            'safety': {
                'max_position_usd': 1000.0,
                'max_total_exposure_usd': 5000.0,
                'max_daily_trades': 100,
                'max_daily_loss_usd': 200.0,
                'max_daily_loss_pct': 0.10,
                'consecutive_loss_limit': 5,
                'max_drawdown_pct': 0.10,
            },
        }

    def _load_json(self, path: str):
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                json_config = json.load(f)
                self._merge_config(json_config)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            print(f"[CONFIG] Error parsing {path}: {e}")

    def _load_env_file(self, path: str):
        """Load configuration from .env file."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
        except FileNotFoundError:
            pass

    def _load_environment(self):
        """Load configuration from environment variables."""
        for env_var, config_path in self.ENV_MAPPINGS.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, self._parse_value(value))

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]

        return value

    def _set_nested(self, path: str, value: Any):
        """Set nested config value using dot notation."""
        parts = path.split('.')
        current = self._config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _merge_config(self, new_config: Dict):
        """Merge new config into existing."""
        def merge(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value

        merge(self._config, new_config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Args:
            key: Config key (e.g., "telegram.token")
            default: Default value if not found

        Returns:
            Config value
        """
        parts = key.split('.')
        current = self._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def get_trading_config(self) -> TradingConfig:
        """
        Build TradingConfig from loaded configuration.

        Returns:
            TradingConfig instance
        """
        # Exchange credentials
        exchange = None
        if self.get('exchange.api_key'):
            exchange = ExchangeCredentials(
                exchange_id=self.get('exchange.exchange_id', 'binance'),
                api_key=self.get('exchange.api_key', ''),
                secret=self.get('exchange.secret', ''),
                password=self.get('exchange.password'),
                sandbox=self.get('exchange.sandbox', False),
            )

        # Telegram config
        telegram = None
        if self.get('telegram.token'):
            telegram = TelegramConfig(
                token=self.get('telegram.token', ''),
                chat_id=self.get('telegram.chat_id', ''),
                enabled=self.get('telegram.enabled', True),
                allow_stop=self.get('telegram.allow_stop', True),
                allow_trade=self.get('telegram.allow_trade', False),
                notify_trades=self.get('telegram.notify_trades', True),
                notify_errors=self.get('telegram.notify_errors', True),
                notify_daily_summary=self.get('telegram.notify_daily_summary', True),
            )

        # Safety limits
        safety = SafetyLimits(
            max_position_usd=self.get('safety.max_position_usd', 1000.0),
            max_total_exposure_usd=self.get('safety.max_total_exposure_usd', 5000.0),
            max_daily_trades=self.get('safety.max_daily_trades', 100),
            max_daily_loss_usd=self.get('safety.max_daily_loss_usd', 200.0),
            max_daily_loss_pct=self.get('safety.max_daily_loss_pct', 0.10),
            consecutive_loss_limit=self.get('safety.consecutive_loss_limit', 5),
            max_drawdown_pct=self.get('safety.max_drawdown_pct', 0.10),
        )

        return TradingConfig(
            mode=self.get('mode', 'paper'),
            exchange=exchange,
            telegram=telegram,
            safety=safety,
            symbols=self.get('symbols', ['BTC/USDT']),
            base_position_size=self.get('base_position_size', 0.01),
            log_level=self.get('log_level', 'INFO'),
            log_file=self.get('log_file'),
        )

    def validate(self) -> tuple:
        """
        Validate configuration.

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        mode = self.get('mode', 'paper')

        # Live mode requires exchange credentials
        if mode == 'live':
            if not self.get('exchange.api_key'):
                errors.append("Live mode requires EXCHANGE_API_KEY")
            if not self.get('exchange.secret'):
                errors.append("Live mode requires EXCHANGE_SECRET")

        # Telegram requires token and chat_id
        if self.get('telegram.enabled', False):
            if not self.get('telegram.token'):
                errors.append("Telegram enabled but TELEGRAM_TOKEN not set")
            if not self.get('telegram.chat_id'):
                errors.append("Telegram enabled but TELEGRAM_CHAT_ID not set")

        # Safety limits validation
        max_pos = self.get('safety.max_position_usd', 0)
        max_exp = self.get('safety.max_total_exposure_usd', 0)

        if max_pos > max_exp:
            errors.append("max_position_usd cannot exceed max_total_exposure_usd")

        return len(errors) == 0, errors

    def print_config(self, hide_secrets: bool = True):
        """Print current configuration (with secrets masked)."""
        def mask_secrets(d: Dict, path: str = "") -> Dict:
            result = {}
            secret_keys = {'api_key', 'secret', 'password', 'token'}

            for key, value in d.items():
                full_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    result[key] = mask_secrets(value, full_path)
                elif hide_secrets and key in secret_keys and value:
                    result[key] = f"***{str(value)[-4:]}" if len(str(value)) > 4 else "****"
                else:
                    result[key] = value

            return result

        masked = mask_secrets(self._config)
        print(json.dumps(masked, indent=2))


def load_config(config_path: Optional[str] = None,
                env_file: Optional[str] = None) -> TradingConfig:
    """
    Load trading configuration.

    Convenience function for simple usage.

    Args:
        config_path: Path to config.json
        env_file: Path to .env file

    Returns:
        TradingConfig
    """
    # Default paths
    if config_path is None:
        for path in ['config.json', 'config/trading.json']:
            if Path(path).exists():
                config_path = path
                break

    if env_file is None:
        for path in ['.env', 'config/.env']:
            if Path(path).exists():
                env_file = path
                break

    manager = ConfigManager(config_path, env_file)

    is_valid, errors = manager.validate()
    if not is_valid:
        for error in errors:
            print(f"[CONFIG] Warning: {error}")

    return manager.get_trading_config()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Configuration Manager Demo")
    print("=" * 50)

    # Set some test env vars
    os.environ['TRADING_MODE'] = 'dry_run'
    os.environ['TELEGRAM_TOKEN'] = 'test_token_12345'
    os.environ['TELEGRAM_CHAT_ID'] = '123456789'
    os.environ['MAX_POSITION_USD'] = '500'

    manager = ConfigManager()

    print("\nLoaded configuration:")
    manager.print_config(hide_secrets=True)

    print("\nValidation:")
    is_valid, errors = manager.validate()
    print(f"  Valid: {is_valid}")
    if errors:
        for e in errors:
            print(f"  - {e}")

    print("\nTradingConfig:")
    config = manager.get_trading_config()
    print(f"  Mode: {config.mode}")
    print(f"  Symbols: {config.symbols}")
    print(f"  Telegram enabled: {config.telegram is not None}")
    print(f"  Max position: ${config.safety.max_position_usd}")
