"""
Claude Adapter - AI Integration for Sovereign Engine
=====================================================

Provides trading-specific methods using Claude subscription via CLI.
All Claude calls are wrapped with timeouts and graceful fallbacks.

INTEGRATION POINTS:
1. Signal Validation - After ensemble voting, before gate
2. Trade Confirmation - Before execute_trade()
3. Risk Assessment - Dynamic Kelly adjustment
4. Market Context - Regime analysis (optional)
"""
import json
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import sys
from pathlib import Path

# Add tools to path for ClaudeClient import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from tools.claude_subscription import ClaudeClient


@dataclass
class ClaudeConfig:
    """Configuration for Claude AI integration."""
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
class ClaudeResponse:
    """Standardized Claude response."""
    success: bool
    action: str = "APPROVE"           # APPROVE, ADJUST, REJECT
    confidence_adjustment: float = 1.0
    size_adjustment: float = 1.0
    reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    latency_ms: int = 0
    error: Optional[str] = None


class ClaudeAdapter:
    """
    Adapter for Claude subscription client with trading-specific methods.

    Thread-safe with graceful fallbacks for production use.
    Uses your existing $20/month Claude subscription via CLI.
    """

    SYSTEM_PROMPT = """You are a quantitative trading analyst for the Sovereign Engine.
You analyze Bitcoin trading signals from blockchain exchange flow data.
Your role is to validate signals, confirm trades, and assess risk.

CRITICAL RULES:
1. Be concise - respond in JSON format only, no markdown
2. Be conservative - when in doubt, recommend reducing position size
3. Never recommend position sizes above 10% of capital
4. Flag any red flags or anomalies immediately
5. Consider recent win rate and drawdown in all recommendations"""

    def __init__(self, config: ClaudeConfig = None):
        """
        Initialize Claude adapter.

        Args:
            config: Claude configuration (uses defaults if None)
        """
        self.config = config or ClaudeConfig()
        self.client: Optional[ClaudeClient] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()

        # Stats
        self.calls_made = 0
        self.calls_succeeded = 0
        self.calls_failed = 0
        self.calls_timed_out = 0
        self.total_latency_ms = 0

        # Response cache (short TTL)
        self._cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self._cache_ttl = 30  # 30 seconds

        if self.config.enabled:
            self._init_client()

    def _init_client(self):
        """Initialize Claude client."""
        try:
            self.client = ClaudeClient(
                timeout=self.config.timeout,
                model=self.config.model
            )
            print(f"[CLAUDE] AI integration initialized (model: {self.config.model})")
        except FileNotFoundError as e:
            print(f"[CLAUDE] CLI not found: {e}")
            self.client = None
            self.config.enabled = False
        except Exception as e:
            print(f"[CLAUDE] Init error: {e}")
            self.client = None
            self.config.enabled = False

    def _call_claude(self, message: str, cache_key: str = None) -> ClaudeResponse:
        """
        Make a Claude API call with timeout and caching.

        Args:
            message: The prompt to send
            cache_key: Optional key for caching (skips cache if None)

        Returns:
            ClaudeResponse with results or fallback
        """
        if not self.config.enabled or not self.client:
            return ClaudeResponse(success=True, reasoning="Claude disabled")

        # Check cache
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        start = time.time()

        try:
            with self.lock:
                self.calls_made += 1

            # Run with timeout using thread pool
            future = self.executor.submit(
                self.client.chat,
                message,
                system=self.SYSTEM_PROMPT
            )

            response_text = future.result(timeout=self.config.timeout)
            latency = int((time.time() - start) * 1000)

            # Parse JSON response
            response = self._parse_response(response_text, latency)

            with self.lock:
                self.calls_succeeded += 1
                self.total_latency_ms += latency

            # Cache successful response
            if cache_key:
                self._set_cached(cache_key, response)

            if self.config.log_responses:
                print(f"[CLAUDE] {latency}ms | {response.action} | {response.reasoning[:50]}...")

            return response

        except FuturesTimeoutError:
            with self.lock:
                self.calls_timed_out += 1

            if self.config.fallback_on_timeout:
                return ClaudeResponse(
                    success=True,
                    reasoning="Timeout - proceeding with default",
                    warnings=["Claude timeout - using defaults"]
                )
            else:
                return ClaudeResponse(
                    success=False,
                    action="REJECT",
                    error="Timeout"
                )

        except Exception as e:
            with self.lock:
                self.calls_failed += 1

            if self.config.fallback_on_timeout:
                return ClaudeResponse(
                    success=True,
                    reasoning=f"Error fallback: {str(e)[:50]}",
                    warnings=[f"Claude error: {str(e)[:50]}"]
                )
            else:
                return ClaudeResponse(
                    success=False,
                    action="REJECT",
                    error=str(e)
                )

    def _parse_response(self, text: str, latency: int) -> ClaudeResponse:
        """Parse Claude's JSON response into ClaudeResponse."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1

            if start >= 0 and end > start:
                data = json.loads(text[start:end])

                return ClaudeResponse(
                    success=True,
                    action=data.get("action", "APPROVE").upper(),
                    confidence_adjustment=float(data.get("confidence_adjustment", 1.0)),
                    size_adjustment=float(data.get("size_adjustment", 1.0)),
                    reasoning=data.get("reasoning", data.get("reason", "")),
                    warnings=data.get("warnings", data.get("red_flags", [])),
                    latency_ms=latency
                )
            else:
                # No JSON found - treat as approval with reasoning
                return ClaudeResponse(
                    success=True,
                    reasoning=text[:200],
                    latency_ms=latency
                )

        except json.JSONDecodeError:
            return ClaudeResponse(
                success=True,
                reasoning=text[:200],
                latency_ms=latency
            )

    def _get_cached(self, key: str) -> Optional[ClaudeResponse]:
        """Get cached response if still valid."""
        if key in self._cache:
            response, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return response
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, response: ClaudeResponse):
        """Cache a response."""
        self._cache[key] = (response, time.time())

        # Clean old entries
        now = time.time()
        self._cache = {k: v for k, v in self._cache.items()
                       if now - v[1] < self._cache_ttl}

    # =========================================================================
    # TRADING-SPECIFIC METHODS
    # =========================================================================

    def validate_signal(self, signal: Dict, context: Dict = None) -> ClaudeResponse:
        """
        Validate ensemble signal before gate.

        Called after 3-way ensemble voting, before breakeven gate.

        Args:
            signal: The ensemble signal dict with direction, confidence, etc.
            context: Additional context (recent trades, engine states, etc.)

        Returns:
            ClaudeResponse with validation result
        """
        if not self.config.validate_signals:
            return ClaudeResponse(success=True, reasoning="Validation disabled")

        # Skip low-confidence signals
        if signal.get('confidence', 0) < self.config.min_confidence_for_claude:
            return ClaudeResponse(success=True, reasoning="Below confidence threshold")

        context = context or {}

        prompt = f"""Analyze this trading signal:
- Direction: {"LONG" if signal.get('direction') == 1 else "SHORT"}
- Confidence: {signal.get('confidence', 0):.2f}
- Engine Agreement: {signal.get('vote_count', 0)}/{signal.get('total_engines', 3)}
- Ensemble Type: {signal.get('ensemble_type', 'unknown')}
- Flow: {signal.get('btc_amount', 0):.2f} BTC from {signal.get('exchange', 'unknown')}
- Regime: {signal.get('regime', 'UNKNOWN')}
- Recent Win Rate: {context.get('win_rate', 50):.1f}%

Questions:
1. Is the signal coherent? (engines agreeing for valid reasons)
2. Any contradictions or red flags?
3. Confidence adjustment recommendation (0.5-1.5)?

Respond in JSON format:
{{"action": "APPROVE|ADJUST|REJECT", "confidence_adjustment": 1.0, "reasoning": "...", "warnings": []}}"""

        cache_key = f"signal_{signal.get('direction')}_{signal.get('vote_count')}"
        return self._call_claude(prompt, cache_key)

    def confirm_trade(self, signal: Dict, account: Dict) -> ClaudeResponse:
        """
        Pre-execution trade confirmation.

        Called right before execute_trade().

        Args:
            signal: The trade signal
            account: Account state (capital, drawdown, win_rate, recent_trades)

        Returns:
            ClaudeResponse with APPROVE/ADJUST/REJECT
        """
        if not self.config.confirm_trades:
            return ClaudeResponse(success=True, reasoning="Confirmation disabled")

        direction = "LONG" if signal.get('direction', 0) == 1 else "SHORT"
        position_pct = signal.get('position_size', 0.05) * 100

        # Format recent trades
        recent = account.get('recent_trades', [])
        recent_str = ", ".join([f"{r:+.2f}%" for r in recent[-5:]]) if recent else "none"

        prompt = f"""Should we execute this trade?
- Type: {direction}
- Position: {position_pct:.1f}% of capital
- Entry Price: ${signal.get('price', 0):,.0f}
- Stop Loss: {signal.get('stop_loss', 0.003) * 100:.1f}%
- Take Profit: {signal.get('take_profit', 0.005) * 100:.1f}%
- Account Drawdown: {account.get('drawdown', 0) * 100:.1f}%
- Win Rate: {account.get('win_rate', 50):.1f}%
- Last 5 trades: [{recent_str}]

Provide:
1. Action: APPROVE/ADJUST/REJECT
2. If ADJUST: size_adjustment multiplier (0.5-2.0)
3. Risk assessment
4. Reasoning (1 sentence)

Respond in JSON format:
{{"action": "APPROVE", "size_adjustment": 1.0, "reasoning": "..."}}"""

        return self._call_claude(prompt)

    def assess_risk(self, portfolio_state: Dict) -> ClaudeResponse:
        """
        Portfolio risk assessment and Kelly adjustment.

        Called periodically or before trades.

        Args:
            portfolio_state: Dict with drawdown, win_rate, volatility, kelly, etc.

        Returns:
            ClaudeResponse with Kelly adjustment recommendation
        """
        if not self.config.risk_assessment:
            return ClaudeResponse(success=True, reasoning="Risk assessment disabled")

        prompt = f"""Assess portfolio risk and recommend Kelly adjustment:

Current State:
- Kelly Fraction: {portfolio_state.get('kelly', 0.25):.2f}
- Drawdown: {portfolio_state.get('drawdown', 0) * 100:.1f}%
- Win Rate: {portfolio_state.get('win_rate', 50):.1f}%
- Sharpe Ratio: {portfolio_state.get('sharpe', 0):.2f}
- Consecutive Losses: {portfolio_state.get('consecutive_losses', 0)}
- Trades Today: {portfolio_state.get('trades_today', 0)}

Recommend Kelly adjustment multiplier (0.1 to 1.5):
- Below 1.0 = reduce risk
- Above 1.0 = increase risk (only if metrics are strong)

Respond in JSON format:
{{"action": "ADJUST", "size_adjustment": 0.8, "reasoning": "...", "warnings": []}}"""

        cache_key = f"risk_{int(portfolio_state.get('drawdown', 0) * 100)}"
        return self._call_claude(prompt, cache_key)

    def analyze_market(self, market_context: Dict) -> ClaudeResponse:
        """
        Market regime and context analysis.

        Optional - only enabled if config.market_context = True.

        Args:
            market_context: Dict with regime, flow data, price action

        Returns:
            ClaudeResponse with market assessment
        """
        if not self.config.market_context:
            return ClaudeResponse(success=True, reasoning="Market context disabled")

        prompt = f"""Analyze current market context:

Data:
- Current Regime: {market_context.get('regime', 'UNKNOWN')}
- HMM Confidence: {market_context.get('hmm_confidence', 0) * 100:.0f}%
- Net Flow (1h): {market_context.get('net_flow_1h', 0):.1f} BTC
- Price 24h Change: {market_context.get('price_change_24h', 0):.1f}%
- Volatility: {market_context.get('volatility', 0) * 100:.2f}%

Provide:
1. Market narrative (1 sentence)
2. Recommended gate threshold adjustment (0.9-1.1)
3. Any special circumstances

Respond in JSON format:
{{"regime_narrative": "...", "threshold_adjustment": 1.0, "special_notes": "..."}}"""

        cache_key = f"market_{market_context.get('regime', 'UNKNOWN')}"
        return self._call_claude(prompt, cache_key)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get Claude adapter statistics."""
        with self.lock:
            avg_latency = (self.total_latency_ms / self.calls_succeeded
                          if self.calls_succeeded > 0 else 0)

            return {
                "enabled": self.config.enabled,
                "client_ready": self.client is not None,
                "calls_made": self.calls_made,
                "calls_succeeded": self.calls_succeeded,
                "calls_failed": self.calls_failed,
                "calls_timed_out": self.calls_timed_out,
                "success_rate": (self.calls_succeeded / self.calls_made * 100
                                if self.calls_made > 0 else 0),
                "avg_latency_ms": avg_latency,
                "cache_entries": len(self._cache),
            }

    def test_connection(self) -> bool:
        """Test Claude connection."""
        if not self.client:
            return False

        try:
            response = self.client.chat("Say 'OK' in one word.")
            return "OK" in response.upper()
        except Exception as e:
            print(f"[CLAUDE] Connection test failed: {e}")
            return False

    def shutdown(self):
        """Clean shutdown."""
        self.executor.shutdown(wait=False)
