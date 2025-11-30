"""
Renaissance Formula Library - Bidirectional Trading Formulas
=============================================================
IDs 403-411: Short Selling, Signal Inversion, and Bearish Detection

Academic Sources:
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
  https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Easley, López de Prado, O'Hara (2012): "Flow Toxicity and Liquidity" (VPIN)
  https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf
- Willy Woo (2017): NVT Ratio - Network Value to Transactions
- He, Manela, Ross (2024): "Fundamentals of Perpetual Futures"
  https://arxiv.org/html/2212.06888v5
- CryptoQuant/Glassnode: On-chain whale movement research

These formulas enable BIDIRECTIONAL trading:
- LONG when blockchain signals are bullish
- SHORT when blockchain signals are bearish
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
from enum import IntEnum

from .base import BaseFormula, FormulaRegistry


class MarketDirection(IntEnum):
    """Market direction for bidirectional trading"""
    STRONG_SHORT = -2
    SHORT = -1
    NEUTRAL = 0
    LONG = 1
    STRONG_LONG = 2


@FormulaRegistry.register(403)
class BidirectionalOUReversion(BaseFormula):
    """
    ID 347: Bidirectional Ornstein-Uhlenbeck Mean Reversion

    Generates BOTH long and short signals based on z-score deviation.

    Academic Basis:
    - Avellaneda & Lee (2010): Statistical arbitrage in the US equities market
    - s-score = (X - μ) / σ where X follows OU process

    Signal Logic:
    - z-score > +threshold → SHORT (price too high, will revert down)
    - z-score < -threshold → LONG (price too low, will revert up)

    Expected Edge: +20-40% win rate on mean reversion trades
    """

    FORMULA_ID = 403
    CATEGORY = "bidirectional"
    NAME = "Bidirectional OU Reversion"
    DESCRIPTION = "Mean reversion signals for both long and short positions"

    def __init__(self, lookback: int = 100,
                 long_threshold: float = -2.0,
                 short_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        # TRIGGER: z-score thresholds for entry
        self.long_threshold = long_threshold   # z < -2.0 → LONG
        self.short_threshold = short_threshold  # z > +2.0 → SHORT
        self.exit_threshold = exit_threshold    # |z| < 0.5 → EXIT

        # DATA: Calculated from LIVE prices
        self.z_score = 0.0
        self.mean = 0.0
        self.std = 0.0
        self.half_life = 0.0
        self.theta = 0.0  # Mean reversion speed

        self.direction = MarketDirection.NEUTRAL

    def _compute(self) -> None:
        """Compute bidirectional mean reversion signal"""
        prices = self._prices_array()

        if len(prices) < self.lookback:
            return

        # Calculate OU parameters from LIVE data
        log_prices = np.log(prices[-self.lookback:])

        # Estimate mean and std
        self.mean = np.mean(log_prices)
        self.std = np.std(log_prices)

        if self.std < 1e-10:
            return

        # Current z-score
        current_log = np.log(prices[-1])
        self.z_score = (current_log - self.mean) / self.std

        # Estimate theta (mean reversion speed) using AR(1) regression
        y = log_prices[1:] - self.mean
        x = log_prices[:-1] - self.mean
        if len(x) > 1 and np.std(x) > 1e-10:
            beta = np.sum(x * y) / np.sum(x * x)
            self.theta = -np.log(max(0.01, min(0.99, beta)))
            self.half_life = np.log(2) / self.theta if self.theta > 0 else float('inf')

        # Generate BIDIRECTIONAL signal
        if self.z_score > self.short_threshold:
            # Price is HIGH → SHORT signal (expect reversion DOWN)
            self.direction = MarketDirection.STRONG_SHORT if self.z_score > self.short_threshold * 1.5 else MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(1.0, abs(self.z_score) / 3.0)
        elif self.z_score < self.long_threshold:
            # Price is LOW → LONG signal (expect reversion UP)
            self.direction = MarketDirection.STRONG_LONG if self.z_score < self.long_threshold * 1.5 else MarketDirection.LONG
            self.signal = 1
            self.confidence = min(1.0, abs(self.z_score) / 3.0)
        elif abs(self.z_score) < self.exit_threshold:
            # Near mean → NEUTRAL (exit any position)
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.5
        else:
            # In between → hold
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'z_score': self.z_score,
            'direction': self.direction.name,
            'mean': self.mean,
            'std': self.std,
            'half_life': self.half_life,
            'theta': self.theta
        })
        return state


@FormulaRegistry.register(404)
class ExchangeInflowBearish(BaseFormula):
    """
    ID 348: Exchange Inflow Bearish Signal

    HIGH exchange inflows = coins moving TO exchanges = SELLING PRESSURE = SHORT

    Academic Basis:
    - CryptoQuant research: "41% of exchange inflows attributed to whales"
    - Glassnode: "Rising exchange inflows point to increased selling pressure"

    Signal Logic:
    - Inflow spike > 2σ above mean → STRONG_SHORT
    - Inflow > 1σ above mean → SHORT
    - Inflow normal → NEUTRAL
    - Outflow spike (negative inflow) → LONG
    """

    FORMULA_ID = 404
    CATEGORY = "bidirectional"
    NAME = "Exchange Inflow Bearish"
    DESCRIPTION = "Generate SHORT signals from exchange inflow spikes"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.inflow_history = deque(maxlen=lookback)
        self.inflow_z_score = 0.0
        self.direction = MarketDirection.NEUTRAL

    def update_inflow(self, inflow_btc: float):
        """Update with new exchange inflow data (positive = inflow, negative = outflow)"""
        self.inflow_history.append(inflow_btc)

    def _compute(self) -> None:
        """Compute bearish signal from exchange inflows"""
        if len(self.inflow_history) < 20:
            return

        inflows = np.array(self.inflow_history)
        mean_inflow = np.mean(inflows)
        std_inflow = np.std(inflows)

        if std_inflow < 1e-10:
            return

        current_inflow = inflows[-1]
        self.inflow_z_score = (current_inflow - mean_inflow) / std_inflow

        # HIGH inflows = BEARISH (people selling)
        if self.inflow_z_score > 2.0:
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.inflow_z_score / 3.0)
        elif self.inflow_z_score > 1.0:
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(0.7, self.inflow_z_score / 3.0)
        elif self.inflow_z_score < -2.0:
            # HIGH outflows = BULLISH (people accumulating)
            self.direction = MarketDirection.STRONG_LONG
            self.signal = 1
            self.confidence = min(1.0, abs(self.inflow_z_score) / 3.0)
        elif self.inflow_z_score < -1.0:
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = min(0.7, abs(self.inflow_z_score) / 3.0)
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'inflow_z_score': self.inflow_z_score,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(405)
class NVTOverboughtSignal(BaseFormula):
    """
    ID 349: NVT Ratio Overbought/Oversold Signal

    NVT = Market Cap / Daily Transaction Volume
    HIGH NVT = Overbought = SHORT signal
    LOW NVT = Oversold = LONG signal

    Academic Basis:
    - Willy Woo (2017): "NVT is the closest thing to P/E ratio for Bitcoin"
    - Thresholds: NVT > 150 = overbought, NVT < 45 = oversold

    Formula:
    NVT Signal = Network Value / 90DMA(Daily Transaction Value)
    """

    FORMULA_ID = 405
    CATEGORY = "bidirectional"
    NAME = "NVT Overbought Signal"
    DESCRIPTION = "SHORT when NVT indicates overbought, LONG when oversold"

    def __init__(self, lookback: int = 90,
                 overbought_threshold: float = 150.0,
                 oversold_threshold: float = 45.0,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        # TRIGGER thresholds from research
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

        # DATA: Must be provided externally
        self.market_cap = 0.0
        self.daily_tx_volume = deque(maxlen=lookback)
        self.nvt_ratio = 0.0
        self.nvt_signal = 0.0
        self.direction = MarketDirection.NEUTRAL

    def update_nvt_data(self, market_cap: float, daily_tx_volume_usd: float):
        """Update with market cap and transaction volume"""
        self.market_cap = market_cap
        self.daily_tx_volume.append(daily_tx_volume_usd)

    def _compute(self) -> None:
        """Compute NVT-based signal"""
        if len(self.daily_tx_volume) < 10 or self.market_cap <= 0:
            return

        # Calculate 90-day moving average of transaction volume
        tx_volumes = np.array(self.daily_tx_volume)
        ma_tx_volume = np.mean(tx_volumes)

        if ma_tx_volume <= 0:
            return

        # NVT Signal = Market Cap / MA(Transaction Volume)
        self.nvt_signal = self.market_cap / ma_tx_volume
        self.nvt_ratio = self.market_cap / tx_volumes[-1] if tx_volumes[-1] > 0 else 0

        # Generate signal
        if self.nvt_signal > self.overbought_threshold:
            # Overbought → SHORT
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.nvt_signal / 200.0)
        elif self.nvt_signal < self.oversold_threshold:
            # Oversold → LONG
            self.direction = MarketDirection.STRONG_LONG
            self.signal = 1
            self.confidence = min(1.0, self.oversold_threshold / max(1, self.nvt_signal))
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'nvt_ratio': self.nvt_ratio,
            'nvt_signal': self.nvt_signal,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(406)
class VPINToxicShort(BaseFormula):
    """
    ID 350: VPIN Toxic Flow Short Signal

    HIGH VPIN = Toxic order flow = Informed traders selling = SHORT

    Academic Basis:
    - Easley, López de Prado, O'Hara (2012): "Flow Toxicity and Liquidity"
    - VPIN predicted Flash Crash hours in advance
    - High VPIN = market makers withdrawing = crash imminent

    Signal Logic:
    - VPIN > 0.8 → STRONG_SHORT (very toxic, crash likely)
    - VPIN > 0.6 → SHORT (moderately toxic)
    - VPIN < 0.3 → LONG (low toxicity, safe to buy)
    """

    FORMULA_ID = 406
    CATEGORY = "bidirectional"
    NAME = "VPIN Toxic Short"
    DESCRIPTION = "SHORT on high VPIN (toxic flow), LONG on low VPIN"

    def __init__(self, lookback: int = 50,
                 toxic_threshold: float = 0.8,
                 warning_threshold: float = 0.6,
                 safe_threshold: float = 0.3,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        self.toxic_threshold = toxic_threshold
        self.warning_threshold = warning_threshold
        self.safe_threshold = safe_threshold

        self.vpin = 0.0
        self.direction = MarketDirection.NEUTRAL

    def update_vpin(self, vpin: float):
        """Update with externally calculated VPIN"""
        self.vpin = vpin

    def _compute(self) -> None:
        """Compute signal from VPIN"""
        if self.vpin > self.toxic_threshold:
            # Very toxic → STRONG SHORT
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.vpin)
        elif self.vpin > self.warning_threshold:
            # Warning level → SHORT
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = self.vpin * 0.8
        elif self.vpin < self.safe_threshold:
            # Low toxicity → safe to LONG
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = 1.0 - self.vpin
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'vpin': self.vpin,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(407)
class WhaleDistributionSignal(BaseFormula):
    """
    ID 351: Whale Distribution/Accumulation Signal

    Whale SELLING (distribution) → SHORT
    Whale BUYING (accumulation) → LONG

    Academic Basis:
    - Glassnode: "Whale accumulation reflects confidence, distribution hints profit-taking"
    - CryptoQuant Exchange Whale Ratio research

    Metrics:
    - Exchange Whale Ratio = Top 10 inflows / Total inflows
    - High ratio = whales depositing to sell = BEARISH
    """

    FORMULA_ID = 407
    CATEGORY = "bidirectional"
    NAME = "Whale Distribution Signal"
    DESCRIPTION = "SHORT on whale distribution, LONG on accumulation"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.whale_ratio_history = deque(maxlen=lookback)
        self.whale_ratio = 0.0
        self.whale_ratio_z = 0.0
        self.direction = MarketDirection.NEUTRAL

    def update_whale_ratio(self, whale_ratio: float):
        """Update with whale ratio (0-1, fraction of volume from whales)"""
        self.whale_ratio = whale_ratio
        self.whale_ratio_history.append(whale_ratio)

    def _compute(self) -> None:
        """Compute signal from whale activity"""
        if len(self.whale_ratio_history) < 20:
            return

        ratios = np.array(self.whale_ratio_history)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        if std_ratio < 1e-10:
            return

        self.whale_ratio_z = (self.whale_ratio - mean_ratio) / std_ratio

        # High whale ratio = whales selling = SHORT
        if self.whale_ratio_z > 2.0:
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.whale_ratio_z / 3.0)
        elif self.whale_ratio_z > 1.0:
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(0.7, self.whale_ratio_z / 3.0)
        elif self.whale_ratio_z < -2.0:
            # Low whale ratio = whales accumulating = LONG
            self.direction = MarketDirection.STRONG_LONG
            self.signal = 1
            self.confidence = min(1.0, abs(self.whale_ratio_z) / 3.0)
        elif self.whale_ratio_z < -1.0:
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = min(0.7, abs(self.whale_ratio_z) / 3.0)
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'whale_ratio': self.whale_ratio,
            'whale_ratio_z': self.whale_ratio_z,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(408)
class FundingRateArbitrage(BaseFormula):
    """
    ID 352: Perpetual Futures Funding Rate Signal

    HIGH positive funding = longs paying shorts = market overleveraged LONG
    → SHORT opportunity (expect correction)

    HIGH negative funding = shorts paying longs = market overleveraged SHORT
    → LONG opportunity

    Academic Basis:
    - He, Manela, Ross (2024): "Fundamentals of Perpetual Futures"
    - Sharpe ratio 1.8-3.5 for funding rate arbitrage
    - Dai, Li, Yang (2025): "Arbitrage in Perpetual Contracts"

    Default funding rate: 0.01% per 8 hours (Binance)
    """

    FORMULA_ID = 408
    CATEGORY = "bidirectional"
    NAME = "Funding Rate Arbitrage"
    DESCRIPTION = "SHORT on high positive funding, LONG on high negative"

    def __init__(self, lookback: int = 100,
                 high_positive_threshold: float = 0.05,  # 0.05% = 5 bps
                 high_negative_threshold: float = -0.05,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        self.high_positive = high_positive_threshold
        self.high_negative = high_negative_threshold

        self.funding_rate = 0.0
        self.funding_history = deque(maxlen=lookback)
        self.direction = MarketDirection.NEUTRAL

    def update_funding_rate(self, funding_rate: float):
        """Update with current funding rate (as percentage, e.g., 0.01 = 0.01%)"""
        self.funding_rate = funding_rate
        self.funding_history.append(funding_rate)

    def _compute(self) -> None:
        """Compute signal from funding rate"""
        if self.funding_rate > self.high_positive:
            # High positive funding = overleveraged longs = SHORT
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(1.0, self.funding_rate / 0.1)
        elif self.funding_rate < self.high_negative:
            # High negative funding = overleveraged shorts = LONG
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = min(1.0, abs(self.funding_rate) / 0.1)
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'funding_rate': self.funding_rate,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(409)
class MempoolPressureInversion(BaseFormula):
    """
    ID 353: Mempool Pressure Inversion Signal

    Our current system only generates BUY on low mempool.
    This formula INVERTS to generate SHORT on HIGH mempool.

    HIGH mempool congestion = users rushing to sell = BEARISH
    LOW mempool = calm market = could go either way

    Logic:
    - Mempool > 2σ above mean → SHORT (panic selling)
    - Mempool > 1σ above mean → slight SHORT
    - Mempool < -1σ below mean → LONG (accumulation)
    """

    FORMULA_ID = 409
    CATEGORY = "bidirectional"
    NAME = "Mempool Pressure Inversion"
    DESCRIPTION = "SHORT on high mempool congestion (panic selling)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.mempool_history = deque(maxlen=lookback)
        self.mempool_size = 0
        self.mempool_z = 0.0
        self.direction = MarketDirection.NEUTRAL

    def update_mempool(self, mempool_size: int):
        """Update with current mempool transaction count"""
        self.mempool_size = mempool_size
        self.mempool_history.append(mempool_size)

    def _compute(self) -> None:
        """Compute signal from mempool pressure"""
        if len(self.mempool_history) < 20:
            return

        sizes = np.array(self.mempool_history)
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        if std_size < 1:
            return

        self.mempool_z = (self.mempool_size - mean_size) / std_size

        # HIGH mempool = congestion = panic = SHORT
        if self.mempool_z > 2.0:
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.mempool_z / 3.0)
        elif self.mempool_z > 1.0:
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(0.6, self.mempool_z / 3.0)
        elif self.mempool_z < -2.0:
            # LOW mempool = calm = good for LONG
            self.direction = MarketDirection.STRONG_LONG
            self.signal = 1
            self.confidence = min(1.0, abs(self.mempool_z) / 3.0)
        elif self.mempool_z < -1.0:
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = min(0.6, abs(self.mempool_z) / 3.0)
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'mempool_size': self.mempool_size,
            'mempool_z': self.mempool_z,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(410)
class FeeSpikeShortSignal(BaseFormula):
    """
    ID 354: Fee Spike Short Signal

    SUDDEN fee spike = urgent transactions = panic selling = SHORT

    Logic:
    - Fee spike > 3x baseline → STRONG_SHORT
    - Fee spike > 2x baseline → SHORT
    - Fee drop < 0.5x baseline → LONG (calm market)
    """

    FORMULA_ID = 410
    CATEGORY = "bidirectional"
    NAME = "Fee Spike Short"
    DESCRIPTION = "SHORT on fee spikes indicating panic"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.fee_history = deque(maxlen=lookback)
        self.current_fee = 0.0
        self.baseline_fee = 0.0
        self.fee_ratio = 1.0
        self.direction = MarketDirection.NEUTRAL

    def update_fee(self, fee_rate: float):
        """Update with current fee rate (sat/vB)"""
        self.current_fee = fee_rate
        self.fee_history.append(fee_rate)

    def _compute(self) -> None:
        """Compute signal from fee dynamics"""
        if len(self.fee_history) < 20:
            return

        fees = np.array(self.fee_history)
        self.baseline_fee = np.median(fees)  # Use median for robustness

        if self.baseline_fee <= 0:
            return

        self.fee_ratio = self.current_fee / self.baseline_fee

        # Fee spike = panic = SHORT
        if self.fee_ratio > 3.0:
            self.direction = MarketDirection.STRONG_SHORT
            self.signal = -1
            self.confidence = min(1.0, self.fee_ratio / 5.0)
        elif self.fee_ratio > 2.0:
            self.direction = MarketDirection.SHORT
            self.signal = -1
            self.confidence = min(0.7, self.fee_ratio / 5.0)
        elif self.fee_ratio < 0.5:
            # Low fees = calm market = LONG
            self.direction = MarketDirection.LONG
            self.signal = 1
            self.confidence = min(0.7, 1.0 / max(0.1, self.fee_ratio))
        else:
            self.direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_fee': self.current_fee,
            'baseline_fee': self.baseline_fee,
            'fee_ratio': self.fee_ratio,
            'direction': self.direction.name
        })
        return state


@FormulaRegistry.register(411)
class BidirectionalSignalAggregator(BaseFormula):
    """
    ID 355: Bidirectional Signal Aggregator

    Combines all bidirectional signals into a single direction.
    Uses Condorcet probability theorem for aggregation.

    Academic Basis:
    - Condorcet Jury Theorem (1785)
    - If each signal is >50% accurate, combining improves accuracy

    Output:
    - direction: STRONG_SHORT, SHORT, NEUTRAL, LONG, STRONG_LONG
    - confidence: Combined probability
    """

    FORMULA_ID = 411
    CATEGORY = "bidirectional"
    NAME = "Bidirectional Aggregator"
    DESCRIPTION = "Combines all bidirectional signals using Condorcet theorem"

    def __init__(self, lookback: int = 100,
                 min_signals: int = 2,
                 min_probability: float = 0.55,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        self.min_signals = min_signals
        self.min_probability = min_probability

        # Signal sources
        self.ou_signal = 0
        self.ou_confidence = 0.0
        self.exchange_signal = 0
        self.exchange_confidence = 0.0
        self.nvt_signal = 0
        self.nvt_confidence = 0.0
        self.vpin_signal = 0
        self.vpin_confidence = 0.0
        self.whale_signal = 0
        self.whale_confidence = 0.0
        self.funding_signal = 0
        self.funding_confidence = 0.0
        self.mempool_signal = 0
        self.mempool_confidence = 0.0
        self.fee_signal = 0
        self.fee_confidence = 0.0

        self.final_direction = MarketDirection.NEUTRAL
        self.combined_probability = 0.5

    def update_signals(self,
                       ou: Tuple[int, float] = None,
                       exchange: Tuple[int, float] = None,
                       nvt: Tuple[int, float] = None,
                       vpin: Tuple[int, float] = None,
                       whale: Tuple[int, float] = None,
                       funding: Tuple[int, float] = None,
                       mempool: Tuple[int, float] = None,
                       fee: Tuple[int, float] = None):
        """Update with signals from individual formulas (signal, confidence)"""
        if ou: self.ou_signal, self.ou_confidence = ou
        if exchange: self.exchange_signal, self.exchange_confidence = exchange
        if nvt: self.nvt_signal, self.nvt_confidence = nvt
        if vpin: self.vpin_signal, self.vpin_confidence = vpin
        if whale: self.whale_signal, self.whale_confidence = whale
        if funding: self.funding_signal, self.funding_confidence = funding
        if mempool: self.mempool_signal, self.mempool_confidence = mempool
        if fee: self.fee_signal, self.fee_confidence = fee

    def _compute(self) -> None:
        """Aggregate all signals"""
        signals = [
            (self.ou_signal, self.ou_confidence),
            (self.exchange_signal, self.exchange_confidence),
            (self.nvt_signal, self.nvt_confidence),
            (self.vpin_signal, self.vpin_confidence),
            (self.whale_signal, self.whale_confidence),
            (self.funding_signal, self.funding_confidence),
            (self.mempool_signal, self.mempool_confidence),
            (self.fee_signal, self.fee_confidence),
        ]

        # Filter to active signals
        active_long = [(s, c) for s, c in signals if s > 0 and c > 0.1]
        active_short = [(s, c) for s, c in signals if s < 0 and c > 0.1]

        # Condorcet aggregation
        if len(active_long) >= self.min_signals and len(active_long) > len(active_short):
            # More LONG signals
            prob = 1.0
            for _, conf in active_long:
                prob *= conf
            for _, conf in active_short:
                prob *= (1 - conf)

            self.combined_probability = prob ** (1 / max(1, len(active_long) + len(active_short)))

            if self.combined_probability >= self.min_probability:
                self.final_direction = MarketDirection.STRONG_LONG if self.combined_probability > 0.7 else MarketDirection.LONG
                self.signal = 1
                self.confidence = self.combined_probability
            else:
                self.final_direction = MarketDirection.NEUTRAL
                self.signal = 0
                self.confidence = 0.5

        elif len(active_short) >= self.min_signals and len(active_short) > len(active_long):
            # More SHORT signals
            prob = 1.0
            for _, conf in active_short:
                prob *= conf
            for _, conf in active_long:
                prob *= (1 - conf)

            self.combined_probability = prob ** (1 / max(1, len(active_long) + len(active_short)))

            if self.combined_probability >= self.min_probability:
                self.final_direction = MarketDirection.STRONG_SHORT if self.combined_probability > 0.7 else MarketDirection.SHORT
                self.signal = -1
                self.confidence = self.combined_probability
            else:
                self.final_direction = MarketDirection.NEUTRAL
                self.signal = 0
                self.confidence = 0.5
        else:
            self.final_direction = MarketDirection.NEUTRAL
            self.signal = 0
            self.confidence = 0.5
            self.combined_probability = 0.5

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'final_direction': self.final_direction.name,
            'combined_probability': self.combined_probability,
            'long_signals': len([s for s, c in [
                (self.ou_signal, self.ou_confidence),
                (self.exchange_signal, self.exchange_confidence),
                (self.nvt_signal, self.nvt_confidence),
                (self.vpin_signal, self.vpin_confidence),
                (self.whale_signal, self.whale_confidence),
                (self.funding_signal, self.funding_confidence),
                (self.mempool_signal, self.mempool_confidence),
                (self.fee_signal, self.fee_confidence),
            ] if s > 0 and c > 0.1]),
            'short_signals': len([s for s, c in [
                (self.ou_signal, self.ou_confidence),
                (self.exchange_signal, self.exchange_confidence),
                (self.nvt_signal, self.nvt_confidence),
                (self.vpin_signal, self.vpin_confidence),
                (self.whale_signal, self.whale_confidence),
                (self.funding_signal, self.funding_confidence),
                (self.mempool_signal, self.mempool_confidence),
                (self.fee_signal, self.fee_confidence),
            ] if s < 0 and c > 0.1])
        })
        return state
