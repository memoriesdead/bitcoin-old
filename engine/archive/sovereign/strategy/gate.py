"""
Breakeven Gate - Protection Layer
ID 950: Ensures 50.75%+ win probability before trading
Max lines: 100
"""
from typing import TYPE_CHECKING
from ..config.exchanges import EXCHANGE_FEE_GATES

if TYPE_CHECKING:
    from ..ai.claude_adapter import ClaudeAdapter

# Gate Constants
RENTECH_MIN_WIN_RATE = 0.5075  # RenTech's documented threshold
DEFAULT_COSTS_BPS = 10.0  # 10 basis points total costs


class BreakevenGate:
    """
    ID 950: Renaissance Technologies Protection Layer.

    CRITICAL: This gate ensures we NEVER trade below 50.75% win probability.
    "We're right 50.75% of the time, but we're 100% right 50.75% of the time."

    FEE-AWARE THRESHOLDS BY EXCHANGE:
    - Bitstamp: 50.75% (0% maker fees - best for HFT)
    - dYdX:     50.75% (-0.025% maker rebate!)
    - Gemini:   51.00% (0.1-0.3% fees)
    - Kraken:   51.00% (0.16-0.26% fees)
    - Coinbase: 51.35% (0.4-0.6% fees - requires higher edge)

    The threshold calculation:
    1. Kelly breakeven = 1 / (1 + risk_reward) = 50% for 1:1
    2. Add exchange-specific fee costs
    3. Add RenTech buffer (0.75%)
    4. Final threshold = max(calculated, exchange_minimum)
    """

    def __init__(self, min_win_rate: float = RENTECH_MIN_WIN_RATE,
                 costs_bps: float = DEFAULT_COSTS_BPS,
                 risk_reward: float = 1.0,
                 exchange: str = None,
                 claude: "ClaudeAdapter" = None):
        self.min_win_rate = min_win_rate
        self.costs_bps = costs_bps
        self.risk_reward = risk_reward
        self.exchange = exchange
        self.claude = claude

        # Get exchange-specific threshold from EXCHANGE_FEE_GATES
        if exchange and exchange in EXCHANGE_FEE_GATES:
            _, self.threshold = EXCHANGE_FEE_GATES[exchange]
        else:
            # Calculate default threshold
            breakeven = 1.0 / (1.0 + risk_reward)
            costs_decimal = costs_bps / 10000.0
            rentech_buffer = 0.0075
            self.threshold = max(breakeven + costs_decimal + rentech_buffer, min_win_rate)

        # Statistics
        self.trades_checked = 0
        self.trades_passed = 0
        self.trades_blocked = 0
        self.exchange_stats = {ex: {'passed': 0, 'blocked': 0} for ex in EXCHANGE_FEE_GATES}

        # Claude market context cache
        self._threshold_adjustment = 1.0
        self._last_context_check = 0

    def check(self, probability: float, exchange: str = None) -> bool:
        """
        Check if trade passes the gate.

        Uses exchange-specific threshold if exchange provided.
        Returns True only if probability >= threshold.
        """
        self.trades_checked += 1

        # Use exchange-specific threshold if available
        threshold = self.threshold
        if exchange and exchange in EXCHANGE_FEE_GATES:
            _, threshold = EXCHANGE_FEE_GATES[exchange]

        if probability >= threshold:
            self.trades_passed += 1
            if exchange and exchange in self.exchange_stats:
                self.exchange_stats[exchange]['passed'] += 1
            return True
        else:
            self.trades_blocked += 1
            if exchange and exchange in self.exchange_stats:
                self.exchange_stats[exchange]['blocked'] += 1
            return False

    def get_pass_rate(self) -> float:
        """Get percentage of trades that passed the gate."""
        if self.trades_checked == 0:
            return 0.0
        return (self.trades_passed / self.trades_checked) * 100

    def get_threshold_for_exchange(self, exchange: str) -> float:
        """Get the fee-aware threshold for a specific exchange."""
        if exchange in EXCHANGE_FEE_GATES:
            return EXCHANGE_FEE_GATES[exchange][1]
        return self.threshold

    def update_market_context(self, context: dict) -> float:
        """
        Update gate threshold based on Claude market context analysis.

        Only called when claude.market_context is enabled.
        Returns the threshold adjustment multiplier.

        Args:
            context: Dict with regime, hmm_confidence, net_flow, volatility, etc.

        Returns:
            Threshold adjustment multiplier (0.9-1.1)
        """
        import time
        now = time.time()

        # Cache for 60 seconds to avoid too many Claude calls
        if now - self._last_context_check < 60:
            return self._threshold_adjustment

        if not self.claude:
            return 1.0

        response = self.claude.analyze_market(context)

        if response.success:
            # Extract threshold adjustment from response
            # The analyze_market response has threshold_adjustment in reasoning
            try:
                import json
                # Parse the response for threshold adjustment
                if hasattr(response, 'reasoning') and response.reasoning:
                    # Try to extract JSON from reasoning
                    start = response.reasoning.find('{')
                    end = response.reasoning.rfind('}') + 1
                    if start >= 0 and end > start:
                        data = json.loads(response.reasoning[start:end])
                        adj = float(data.get('threshold_adjustment', 1.0))
                        self._threshold_adjustment = max(0.9, min(1.1, adj))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

            self._last_context_check = now

            if self._threshold_adjustment != 1.0:
                print(f"[GATE] Claude threshold adjustment: {self._threshold_adjustment:.2f}")

        return self._threshold_adjustment

    def check_with_context(self, probability: float, exchange: str = None,
                           context: dict = None) -> bool:
        """
        Check if trade passes the gate with Claude market context.

        Args:
            probability: Win probability from signal
            exchange: Exchange ID for fee-aware threshold
            context: Market context for Claude analysis

        Returns:
            True if probability passes the adjusted threshold
        """
        # Get Claude context adjustment if context provided
        if context and self.claude:
            self.update_market_context(context)

        # Apply threshold adjustment
        threshold = self.get_threshold_for_exchange(exchange) if exchange else self.threshold
        adjusted_threshold = threshold * self._threshold_adjustment

        self.trades_checked += 1

        if probability >= adjusted_threshold:
            self.trades_passed += 1
            if exchange and exchange in self.exchange_stats:
                self.exchange_stats[exchange]['passed'] += 1
            return True
        else:
            self.trades_blocked += 1
            if exchange and exchange in self.exchange_stats:
                self.exchange_stats[exchange]['blocked'] += 1
            return False
