"""
Telegram Bot Controller
======================

Remote monitoring and control via Telegram.

Commands:
- /status - Current trading status
- /profit - PnL summary
- /stop - Emergency stop (kill switch)
- /start - Resume trading
- /positions - Open positions
- /config - Current configuration

Freqtrade pattern: Telegram for remote control.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque
import threading
import queue

# Try to import telegram
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application, CommandHandler, ContextTypes,
        MessageHandler, filters
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    Update = None
    Bot = None
    Application = None
    CommandHandler = None
    MessageHandler = None
    filters = None
    # Create mock ContextTypes for type hints when telegram not installed
    class _MockContextTypes:
        DEFAULT_TYPE = None
    ContextTypes = _MockContextTypes


@dataclass
class TradingStatus:
    """Current trading status for reporting."""
    mode: str = "paper"
    is_running: bool = False
    kill_switch: bool = False

    # Performance
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trades_today: int = 0
    total_trades: int = 0

    # Positions
    open_positions: int = 0
    total_exposure: float = 0.0

    # Risk
    current_drawdown: float = 0.0
    consecutive_losses: int = 0

    # System
    uptime_hours: float = 0.0
    last_signal_time: Optional[float] = None
    errors_today: int = 0


class TelegramNotifier:
    """
    Sends notifications to Telegram.

    Use this for one-way notifications without running full bot.
    """

    def __init__(self, token: str, chat_id: str):
        """
        Initialize notifier.

        Args:
            token: Bot token from @BotFather
            chat_id: Chat ID to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self._enabled = HAS_TELEGRAM

        if self._enabled:
            self.bot = Bot(token=token)

    async def send_message(self, text: str, parse_mode: str = "HTML"):
        """Send message asynchronously."""
        if not self._enabled:
            print(f"[TELEGRAM] (disabled) {text}")
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as e:
            print(f"[TELEGRAM] Error sending: {e}")

    def send_sync(self, text: str):
        """Send message synchronously."""
        asyncio.run(self.send_message(text))

    def notify_trade(self, symbol: str, side: str, amount: float,
                     price: float, pnl: Optional[float] = None):
        """Send trade notification."""
        emoji = "🟢" if side == "buy" else "🔴"
        pnl_str = f"\nPnL: ${pnl:+.2f}" if pnl is not None else ""

        text = f"""
{emoji} <b>Trade Executed</b>
Symbol: {symbol}
Side: {side.upper()}
Amount: {amount:.6f}
Price: ${price:,.2f}{pnl_str}
"""
        self.send_sync(text.strip())

    def notify_error(self, error: str, severity: str = "warning"):
        """Send error notification."""
        emoji = "⚠️" if severity == "warning" else "🚨"

        text = f"""
{emoji} <b>Alert: {severity.upper()}</b>
{error}
"""
        self.send_sync(text.strip())

    def notify_daily_summary(self, status: TradingStatus):
        """Send daily summary."""
        emoji = "📈" if status.daily_pnl >= 0 else "📉"

        text = f"""
{emoji} <b>Daily Summary</b>
PnL: ${status.daily_pnl:+.2f}
Trades: {status.trades_today}
Win Rate: {status.win_rate*100:.1f}%
Drawdown: {status.current_drawdown*100:.1f}%
"""
        self.send_sync(text.strip())


class TelegramBot:
    """
    Full Telegram bot with command handling.

    Runs in background thread, processes commands.
    """

    def __init__(self, token: str, chat_id: str,
                 allowed_chat_ids: Optional[List[str]] = None):
        """
        Initialize bot.

        Args:
            token: Bot token
            chat_id: Primary chat ID
            allowed_chat_ids: Additional allowed chat IDs (security)
        """
        self.token = token
        self.chat_id = chat_id
        self.allowed_chat_ids = set(allowed_chat_ids or [chat_id])
        self.allowed_chat_ids.add(chat_id)

        self._enabled = HAS_TELEGRAM
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[Any] = None

        # Command queue for main thread
        self._command_queue: queue.Queue = queue.Queue()

        # Callbacks
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_start: Optional[Callable[[], None]] = None
        self.get_status: Optional[Callable[[], TradingStatus]] = None
        self.get_positions: Optional[Callable[[], List[Dict]]] = None
        self.get_config: Optional[Callable[[], Dict]] = None

        # Notifier for outgoing messages
        if self._enabled:
            self.notifier = TelegramNotifier(token, chat_id)
        else:
            self.notifier = None

    def start(self):
        """Start bot in background thread."""
        if not self._enabled:
            print("[TELEGRAM] Telegram not available (pip install python-telegram-bot)")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True)
        self._thread.start()
        print("[TELEGRAM] Bot started")

    def stop(self):
        """Stop bot."""
        self._running = False
        if self._app:
            asyncio.run(self._app.stop())
        print("[TELEGRAM] Bot stopped")

    def _run_bot(self):
        """Run bot event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._async_run())
        except Exception as e:
            print(f"[TELEGRAM] Bot error: {e}")

    async def _async_run(self):
        """Async bot runner."""
        self._app = Application.builder().token(self.token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("profit", self._cmd_profit))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("config", self._cmd_config))
        self._app.add_handler(CommandHandler("help", self._cmd_help))

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Keep running
        while self._running:
            await asyncio.sleep(1)

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def _check_auth(self, update: Update) -> bool:
        """Check if user is authorized."""
        chat_id = str(update.effective_chat.id)
        return chat_id in self.allowed_chat_ids

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._check_auth(update):
            await update.message.reply_text("Unauthorized")
            return

        if self.get_status:
            status = self.get_status()
        else:
            status = TradingStatus()

        running_emoji = "✅" if status.is_running else "⏸️"
        kill_emoji = "🚨 KILL SWITCH ACTIVE" if status.kill_switch else ""

        text = f"""
<b>Trading Status</b> {running_emoji}
{kill_emoji}

<b>Mode:</b> {status.mode.upper()}
<b>Uptime:</b> {status.uptime_hours:.1f}h

<b>Performance:</b>
Daily PnL: ${status.daily_pnl:+.2f}
Total PnL: ${status.total_pnl:+.2f}
Win Rate: {status.win_rate*100:.1f}%
Trades Today: {status.trades_today}

<b>Risk:</b>
Drawdown: {status.current_drawdown*100:.1f}%
Consecutive Losses: {status.consecutive_losses}
Open Positions: {status.open_positions}
Exposure: ${status.total_exposure:.2f}

<b>System:</b>
Errors Today: {status.errors_today}
"""
        await update.message.reply_text(text.strip(), parse_mode="HTML")

    async def _cmd_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /profit command."""
        if not self._check_auth(update):
            return

        if self.get_status:
            status = self.get_status()
        else:
            status = TradingStatus()

        emoji = "📈" if status.total_pnl >= 0 else "📉"

        text = f"""
{emoji} <b>Profit Summary</b>

<b>Today:</b> ${status.daily_pnl:+.2f}
<b>Total:</b> ${status.total_pnl:+.2f}

<b>Trades:</b>
Today: {status.trades_today}
Total: {status.total_trades}
Win Rate: {status.win_rate*100:.1f}%
"""
        await update.message.reply_text(text.strip(), parse_mode="HTML")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - emergency stop."""
        if not self._check_auth(update):
            return

        if self.on_stop:
            self.on_stop()
            await update.message.reply_text("🛑 <b>KILL SWITCH ACTIVATED</b>\nTrading stopped.", parse_mode="HTML")
        else:
            await update.message.reply_text("Stop handler not configured")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - resume trading."""
        if not self._check_auth(update):
            return

        if self.on_start:
            self.on_start()
            await update.message.reply_text("✅ <b>Trading Resumed</b>", parse_mode="HTML")
        else:
            await update.message.reply_text("Start handler not configured")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not self._check_auth(update):
            return

        if self.get_positions:
            positions = self.get_positions()
        else:
            positions = []

        if not positions:
            await update.message.reply_text("No open positions")
            return

        text = "<b>Open Positions:</b>\n\n"
        for pos in positions:
            emoji = "🟢" if pos.get('pnl', 0) >= 0 else "🔴"
            text += f"{emoji} {pos.get('symbol', '???')}\n"
            text += f"   Size: {pos.get('size', 0):.6f}\n"
            text += f"   Entry: ${pos.get('entry_price', 0):,.2f}\n"
            text += f"   PnL: ${pos.get('pnl', 0):+.2f}\n\n"

        await update.message.reply_text(text.strip(), parse_mode="HTML")

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command."""
        if not self._check_auth(update):
            return

        if self.get_config:
            config = self.get_config()
        else:
            config = {}

        text = "<b>Current Configuration:</b>\n\n"

        # Safe keys only
        safe_keys = ['mode', 'symbols', 'base_position_size',
                     'max_position_usd', 'max_daily_loss_usd']

        for key in safe_keys:
            if key in config:
                text += f"<b>{key}:</b> {config[key]}\n"

        await update.message.reply_text(text.strip(), parse_mode="HTML")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self._check_auth(update):
            return

        text = """
<b>Available Commands:</b>

/status - Current trading status
/profit - PnL summary
/positions - Open positions
/config - Current configuration
/stop - Emergency stop (kill switch)
/start - Resume trading
/help - This message
"""
        await update.message.reply_text(text.strip(), parse_mode="HTML")

    def notify_trade(self, symbol: str, side: str, amount: float,
                     price: float, pnl: Optional[float] = None):
        """Send trade notification."""
        if self.notifier:
            self.notifier.notify_trade(symbol, side, amount, price, pnl)

    def notify_error(self, error: str, severity: str = "warning"):
        """Send error notification."""
        if self.notifier:
            self.notifier.notify_error(error, severity)

    def notify_daily_summary(self, status: TradingStatus):
        """Send daily summary."""
        if self.notifier:
            self.notifier.notify_daily_summary(status)


class MockTelegramBot:
    """
    Mock bot for testing without Telegram.

    Logs messages to console instead.
    """

    def __init__(self, *args, **kwargs):
        self.messages = []
        self.on_stop = None
        self.on_start = None
        self.get_status = None

    def start(self):
        print("[MOCK TELEGRAM] Bot started")

    def stop(self):
        print("[MOCK TELEGRAM] Bot stopped")

    def notify_trade(self, symbol: str, side: str, amount: float,
                     price: float, pnl: Optional[float] = None):
        msg = f"[TRADE] {symbol} {side} {amount} @ ${price}"
        if pnl:
            msg += f" PnL: ${pnl:+.2f}"
        print(f"[MOCK TELEGRAM] {msg}")
        self.messages.append(msg)

    def notify_error(self, error: str, severity: str = "warning"):
        print(f"[MOCK TELEGRAM] [{severity.upper()}] {error}")
        self.messages.append(f"[{severity}] {error}")

    def notify_daily_summary(self, status: TradingStatus):
        msg = f"[SUMMARY] PnL: ${status.daily_pnl:+.2f}, Trades: {status.trades_today}"
        print(f"[MOCK TELEGRAM] {msg}")
        self.messages.append(msg)


def create_telegram_bot(token: str, chat_id: str,
                        mock: bool = False) -> TelegramBot:
    """
    Factory function to create Telegram bot.

    Args:
        token: Bot token
        chat_id: Chat ID
        mock: Use mock bot for testing

    Returns:
        TelegramBot or MockTelegramBot
    """
    if mock or not HAS_TELEGRAM:
        return MockTelegramBot(token, chat_id)
    return TelegramBot(token, chat_id)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Telegram Bot Demo")
    print("=" * 50)
    print(f"python-telegram-bot available: {HAS_TELEGRAM}")

    # Create mock bot
    bot = create_telegram_bot("test_token", "123456789", mock=True)

    # Simulate status callback
    def get_status():
        return TradingStatus(
            mode="dry_run",
            is_running=True,
            daily_pnl=150.0,
            total_pnl=1250.0,
            win_rate=0.65,
            trades_today=12,
            total_trades=156,
            current_drawdown=0.02,
        )

    bot.get_status = get_status

    # Test notifications
    print("\nTest notifications:")
    bot.notify_trade("BTC/USDT", "buy", 0.05, 42000.0)
    bot.notify_trade("BTC/USDT", "sell", 0.05, 42500.0, pnl=25.0)
    bot.notify_error("High latency detected", "warning")
    bot.notify_daily_summary(get_status())

    print(f"\nMessages sent: {len(bot.messages)}")
