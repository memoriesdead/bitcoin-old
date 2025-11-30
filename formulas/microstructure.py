"""
Market Microstructure Formulas (IDs 101-130)
============================================
Kyle's Lambda, VPIN, OFI, bid-ask analysis, and market making signals.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# KYLE'S LAMBDA AND PRICE IMPACT (101-110)
# =============================================================================

@FormulaRegistry.register(101)
class KylesLambda(BaseFormula):
    """ID 101: Kyle's Lambda - Price impact coefficient"""

    CATEGORY = "microstructure"
    NAME = "KylesLambda"
    DESCRIPTION = "λ = Cov(ΔP, Q) / Var(Q) - Price impact per unit volume"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.price_changes = deque(maxlen=lookback)
        self.order_flows = deque(maxlen=lookback)
        self.lambda_value = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 2 or len(self.volumes) < 2:
            return
        dp = self.prices[-1] - self.prices[-2]
        vol = self.volumes[-1]
        direction = 1 if dp > 0 else -1
        order_flow = direction * vol
        self.price_changes.append(dp)
        self.order_flows.append(order_flow)
        if len(self.price_changes) < 20:
            return
        dp_arr = np.array(self.price_changes)
        of_arr = np.array(self.order_flows)
        cov = np.cov(dp_arr, of_arr)[0, 1]
        var_of = np.var(of_arr)
        self.lambda_value = cov / (var_of + 1e-10)
        if self.lambda_value > 0.0001:
            self.signal = -1
            self.confidence = min(self.lambda_value * 1000, 1.0)
        elif self.lambda_value < -0.0001:
            self.signal = 1
            self.confidence = min(abs(self.lambda_value) * 1000, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(102)
class KyleObizhaeva(BaseFormula):
    """ID 102: Kyle-Obizhaeva Market Impact Model"""

    CATEGORY = "microstructure"
    NAME = "KyleObizhaeva"
    DESCRIPTION = "Impact = σ × sqrt(Q / ADV) - Square root law"

    def __init__(self, lookback: int = 100, adv: float = 1e9, **kwargs):
        super().__init__(lookback, **kwargs)
        self.adv = adv
        self.cumulative_impact = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 10 or len(self.volumes) < 2:
            return
        returns = self._returns_array()
        sigma = np.std(returns) + 1e-10
        vol = self.volumes[-1]
        impact = sigma * np.sqrt(vol / self.adv)
        self.cumulative_impact = 0.9 * self.cumulative_impact + 0.1 * impact
        if self.cumulative_impact > sigma * 0.5:
            self.signal = -1
            self.confidence = min(self.cumulative_impact / sigma, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(103)
class AlmgrenChriss(BaseFormula):
    """ID 103: Almgren-Chriss Optimal Execution"""

    CATEGORY = "microstructure"
    NAME = "AlmgrenChriss"
    DESCRIPTION = "Optimal trade schedule minimizing cost + risk"

    def __init__(self, lookback: int = 100, risk_aversion: float = 1e-6, **kwargs):
        super().__init__(lookback, **kwargs)
        self.risk_aversion = risk_aversion
        self.execution_urgency = 0.5

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        sigma = np.std(returns) + 1e-10
        eta = 0.01 * sigma
        gamma = 0.1 * sigma
        kappa = np.sqrt(self.risk_aversion * sigma**2 / eta)
        self.execution_urgency = np.tanh(kappa)
        momentum = np.mean(returns[-5:])
        if momentum > 0 and self.execution_urgency > 0.6:
            self.signal = 1
            self.confidence = self.execution_urgency
        elif momentum < 0 and self.execution_urgency > 0.6:
            self.signal = -1
            self.confidence = self.execution_urgency
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(104)
class LinearPriceImpact(BaseFormula):
    """ID 104: Linear Temporary Price Impact"""

    CATEGORY = "microstructure"
    NAME = "LinearPriceImpact"
    DESCRIPTION = "ΔP = η × v - Linear impact model"

    def __init__(self, lookback: int = 100, eta: float = 1e-8, **kwargs):
        super().__init__(lookback, **kwargs)
        self.eta = eta
        self.impact_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.volumes) < 2 or len(self.prices) < 2:
            return
        vol = self.volumes[-1]
        dp = self.prices[-1] - self.prices[-2]
        expected_impact = self.eta * vol
        actual_impact = abs(dp)
        self.impact_history.append(actual_impact / (expected_impact + 1e-10))
        if len(self.impact_history) < 10:
            return
        impact_ratio = np.mean(self.impact_history)
        if impact_ratio > 1.5:
            self.signal = -1
            self.confidence = min(impact_ratio / 3.0, 1.0)
        elif impact_ratio < 0.5:
            self.signal = 1
            self.confidence = min(1 / (impact_ratio + 0.1), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(105)
class SquareRootImpact(BaseFormula):
    """ID 105: Square Root Price Impact"""

    CATEGORY = "microstructure"
    NAME = "SquareRootImpact"
    DESCRIPTION = "ΔP = η × sqrt(v) - Square root law"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.impact_coefficient = 0.0

    def _compute(self) -> None:
        if len(self.volumes) < 10 or len(self.prices) < 10:
            return
        price_changes = np.diff(list(self.prices)[-10:])
        volumes = np.array(list(self.volumes)[-9:])
        sqrt_volumes = np.sqrt(volumes + 1)
        if np.std(sqrt_volumes) > 0:
            self.impact_coefficient = np.corrcoef(np.abs(price_changes), sqrt_volumes)[0, 1]
        if self.impact_coefficient > 0.5:
            self.signal = -1
            self.confidence = self.impact_coefficient
        elif self.impact_coefficient < -0.3:
            self.signal = 1
            self.confidence = abs(self.impact_coefficient)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(106)
class PermanentImpact(BaseFormula):
    """ID 106: Permanent Price Impact Model"""

    CATEGORY = "microstructure"
    NAME = "PermanentImpact"
    DESCRIPTION = "Long-term price shift from trades"

    def __init__(self, lookback: int = 100, decay_rate: float = 0.95, **kwargs):
        super().__init__(lookback, **kwargs)
        self.decay_rate = decay_rate
        self.cumulative_permanent = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 2 or len(self.volumes) < 2:
            return
        ret = self.returns[-1]
        vol = self.volumes[-1]
        impact = ret * np.log1p(vol / 1e6)
        self.cumulative_permanent = self.decay_rate * self.cumulative_permanent + impact
        if self.cumulative_permanent > 0.001:
            self.signal = 1
            self.confidence = min(abs(self.cumulative_permanent) * 100, 1.0)
        elif self.cumulative_permanent < -0.001:
            self.signal = -1
            self.confidence = min(abs(self.cumulative_permanent) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(107)
class TemporaryImpact(BaseFormula):
    """ID 107: Temporary Price Impact (Mean Reverting)"""

    CATEGORY = "microstructure"
    NAME = "TemporaryImpact"
    DESCRIPTION = "Short-term impact that decays"

    def __init__(self, lookback: int = 100, half_life: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.half_life = half_life
        self.impact_buffer = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2 or len(self.volumes) < 2:
            return
        ret = self.returns[-1]
        vol = self.volumes[-1]
        impact = ret * vol / 1e6
        self.impact_buffer.append(impact)
        if len(self.impact_buffer) < self.half_life:
            return
        decay = 0.5 ** (1.0 / self.half_life)
        weighted_impact = 0.0
        weight_sum = 0.0
        for i, imp in enumerate(reversed(list(self.impact_buffer))):
            w = decay ** i
            weighted_impact += w * imp
            weight_sum += w
        avg_impact = weighted_impact / (weight_sum + 1e-10)
        if avg_impact > 0.0001:
            self.signal = -1
            self.confidence = min(abs(avg_impact) * 1000, 1.0)
        elif avg_impact < -0.0001:
            self.signal = 1
            self.confidence = min(abs(avg_impact) * 1000, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(108)
class PropagatorModel(BaseFormula):
    """ID 108: Price Impact Propagator"""

    CATEGORY = "microstructure"
    NAME = "PropagatorModel"
    DESCRIPTION = "Impact propagation through time"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.propagator_kernel = None
        self.trade_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2 or len(self.volumes) < 2:
            return
        sign = 1 if self.returns[-1] > 0 else -1
        trade = sign * self.volumes[-1]
        self.trade_history.append(trade)
        if len(self.trade_history) < 20:
            return
        trades = np.array(self.trade_history)
        kernel_size = min(10, len(trades))
        kernel = np.exp(-np.arange(kernel_size) / 3.0)
        kernel /= kernel.sum()
        propagated = np.convolve(trades, kernel, mode='valid')
        current_signal = propagated[-1] if len(propagated) > 0 else 0
        if current_signal > 1e5:
            self.signal = 1
            self.confidence = min(current_signal / 1e6, 1.0)
        elif current_signal < -1e5:
            self.signal = -1
            self.confidence = min(abs(current_signal) / 1e6, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(109)
class TransientImpact(BaseFormula):
    """ID 109: Transient Impact Response"""

    CATEGORY = "microstructure"
    NAME = "TransientImpact"
    DESCRIPTION = "Impulse response of price to trades"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.response_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 5 or len(self.volumes) < 5:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        vol_spikes = volumes > np.mean(volumes) * 1.5
        responses = []
        for i in range(5, len(vol_spikes)):
            if vol_spikes[i-5]:
                response = np.sum(returns[i-4:i+1])
                responses.append(response)
        if len(responses) < 3:
            return
        avg_response = np.mean(responses)
        self.response_history.append(avg_response)
        if len(self.response_history) < 5:
            return
        trend = np.mean(list(self.response_history)[-5:])
        if trend > 0.001:
            self.signal = 1
            self.confidence = min(abs(trend) * 100, 1.0)
        elif trend < -0.001:
            self.signal = -1
            self.confidence = min(abs(trend) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(110)
class MetaOrderImpact(BaseFormula):
    """ID 110: Meta-Order Impact Detection"""

    CATEGORY = "microstructure"
    NAME = "MetaOrderImpact"
    DESCRIPTION = "Detect large hidden orders from impact pattern"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.impact_pattern = deque(maxlen=lookback)
        self.meta_order_detected = False

    def _compute(self) -> None:
        if len(self.returns) < 10 or len(self.volumes) < 10:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        recent_volumes = volumes[-10:]
        recent_returns = returns[-10:]
        vol_increasing = all(recent_volumes[i] <= recent_volumes[i+1]
                            for i in range(len(recent_volumes)-1))
        same_direction = all(r >= 0 for r in recent_returns) or all(r <= 0 for r in recent_returns)
        self.meta_order_detected = vol_increasing and same_direction
        if self.meta_order_detected:
            if np.mean(recent_returns) > 0:
                self.signal = 1
            else:
                self.signal = -1
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# VPIN AND TOXICITY (111-120)
# =============================================================================

@FormulaRegistry.register(111)
class VPIN(BaseFormula):
    """ID 111: Volume-Synchronized Probability of Informed Trading"""

    CATEGORY = "microstructure"
    NAME = "VPIN"
    DESCRIPTION = "VPIN = |V_buy - V_sell| / V_total"

    def __init__(self, lookback: int = 100, bucket_size: float = 1e6, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bucket_size = bucket_size
        self.current_bucket = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.vpin_history = deque(maxlen=50)
        self.vpin = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        vol = self.volumes[-1]
        ret = self.returns[-1] if len(self.returns) > 0 else 0
        if ret >= 0:
            self.buy_volume += vol
        else:
            self.sell_volume += vol
        self.current_bucket += vol
        if self.current_bucket >= self.bucket_size:
            total = self.buy_volume + self.sell_volume
            if total > 0:
                self.vpin = abs(self.buy_volume - self.sell_volume) / total
                self.vpin_history.append(self.vpin)
            self.current_bucket = 0.0
            self.buy_volume = 0.0
            self.sell_volume = 0.0
        if len(self.vpin_history) < 5:
            return
        avg_vpin = np.mean(self.vpin_history)
        if avg_vpin > 0.7:
            self.signal = -1
            self.confidence = avg_vpin
        elif avg_vpin < 0.3:
            self.signal = 1
            self.confidence = 1 - avg_vpin
        else:
            self.signal = 0
            self.confidence = 0.5


@FormulaRegistry.register(112)
class BulkVPIN(BaseFormula):
    """ID 112: Bulk Volume VPIN Classification"""

    CATEGORY = "microstructure"
    NAME = "BulkVPIN"
    DESCRIPTION = "Classify bulk volume as informed/uninformed"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bulk_classifier = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 10 or len(self.volumes) < 10:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        sigma = np.std(returns) + 1e-10
        for i in range(len(returns)):
            z = returns[i] / sigma
            prob_buy = self._sigmoid(z * 2)
            buy_vol = prob_buy * volumes[i]
            sell_vol = (1 - prob_buy) * volumes[i]
            self.bulk_classifier.append((buy_vol, sell_vol))
        if len(self.bulk_classifier) < 20:
            return
        recent = list(self.bulk_classifier)[-20:]
        total_buy = sum(b for b, s in recent)
        total_sell = sum(s for b, s in recent)
        total = total_buy + total_sell + 1e-10
        imbalance = (total_buy - total_sell) / total
        if imbalance > 0.3:
            self.signal = 1
            self.confidence = min(abs(imbalance), 1.0)
        elif imbalance < -0.3:
            self.signal = -1
            self.confidence = min(abs(imbalance), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(113)
class ToxicityIndex(BaseFormula):
    """ID 113: Flow Toxicity Index"""

    CATEGORY = "microstructure"
    NAME = "ToxicityIndex"
    DESCRIPTION = "Measure of adverse selection risk"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.toxicity = 0.0
        self.toxicity_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        sigma = np.std(returns) + 1e-10
        abs_returns = np.abs(returns[-20:])
        normal_abs = np.sqrt(2/np.pi) * sigma
        self.toxicity = np.mean(abs_returns) / normal_abs - 1
        self.toxicity_history.append(self.toxicity)
        avg_toxicity = np.mean(self.toxicity_history) if len(self.toxicity_history) > 5 else self.toxicity
        if avg_toxicity > 0.5:
            self.signal = -1
            self.confidence = min(avg_toxicity, 1.0)
        elif avg_toxicity < -0.3:
            self.signal = 1
            self.confidence = min(abs(avg_toxicity), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(114)
class InformedTraderProbability(BaseFormula):
    """ID 114: Probability of Informed Trader (PIN)"""

    CATEGORY = "microstructure"
    NAME = "InformedTraderProbability"
    DESCRIPTION = "PIN = αμ / (αμ + ε_b + ε_s)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.alpha = 0.3
        self.mu = 1.0
        self.epsilon_b = 0.5
        self.epsilon_s = 0.5
        self.pin = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 20 or len(self.volumes) < 20:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        buy_trades = np.sum(returns > 0)
        sell_trades = np.sum(returns < 0)
        total_trades = len(returns)
        if total_trades > 0:
            self.epsilon_b = buy_trades / total_trades
            self.epsilon_s = sell_trades / total_trades
        abs_returns = np.abs(returns)
        unusual_moves = np.sum(abs_returns > 2 * np.std(returns))
        self.alpha = unusual_moves / (total_trades + 1)
        denominator = self.alpha * self.mu + self.epsilon_b + self.epsilon_s
        self.pin = (self.alpha * self.mu) / (denominator + 1e-10)
        if self.pin > 0.5:
            self.signal = -1
            self.confidence = self.pin
        else:
            self.signal = 0
            self.confidence = 1 - self.pin


@FormulaRegistry.register(115)
class AdverseSelection(BaseFormula):
    """ID 115: Adverse Selection Component"""

    CATEGORY = "microstructure"
    NAME = "AdverseSelection"
    DESCRIPTION = "Measure adverse selection from spread"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.adverse_component = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        signed_returns = returns[-20:]
        autocov = np.cov(signed_returns[:-1], signed_returns[1:])[0, 1]
        var_ret = np.var(signed_returns) + 1e-10
        self.adverse_component = -autocov / var_ret
        if self.adverse_component > 0.3:
            self.signal = -1
            self.confidence = min(self.adverse_component, 1.0)
        elif self.adverse_component < -0.1:
            self.signal = 1
            self.confidence = min(abs(self.adverse_component), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(116)
class InventoryComponent(BaseFormula):
    """ID 116: Inventory Component of Spread"""

    CATEGORY = "microstructure"
    NAME = "InventoryComponent"
    DESCRIPTION = "Spread component from inventory risk"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.inventory = 0.0
        self.inventory_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2 or len(self.volumes) < 2:
            return
        sign = 1 if self.returns[-1] > 0 else -1
        vol = self.volumes[-1]
        self.inventory += sign * vol
        self.inventory_history.append(self.inventory)
        if len(self.inventory_history) < 10:
            return
        inv_arr = np.array(self.inventory_history)
        inv_std = np.std(inv_arr) + 1e-10
        inv_z = (self.inventory - np.mean(inv_arr)) / inv_std
        if inv_z > 2:
            self.signal = -1
            self.confidence = min(abs(inv_z) / 4, 1.0)
        elif inv_z < -2:
            self.signal = 1
            self.confidence = min(abs(inv_z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(117)
class OrderProcessingCost(BaseFormula):
    """ID 117: Order Processing Cost Component"""

    CATEGORY = "microstructure"
    NAME = "OrderProcessingCost"
    DESCRIPTION = "Fixed cost component of spread"

    def __init__(self, lookback: int = 100, base_cost: float = 0.0001, **kwargs):
        super().__init__(lookback, **kwargs)
        self.base_cost = base_cost
        self.effective_cost = base_cost

    def _compute(self) -> None:
        if len(self.returns) < 20 or len(self.volumes) < 20:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        avg_volume = np.mean(volumes)
        current_volume = volumes[-1]
        volume_ratio = current_volume / (avg_volume + 1)
        self.effective_cost = self.base_cost / np.sqrt(volume_ratio + 0.1)
        if self.effective_cost > self.base_cost * 2:
            self.signal = -1
            self.confidence = min(self.effective_cost / self.base_cost / 5, 1.0)
        elif self.effective_cost < self.base_cost * 0.5:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(118)
class RealizedSpread(BaseFormula):
    """ID 118: Realized Spread Analysis"""

    CATEGORY = "microstructure"
    NAME = "RealizedSpread"
    DESCRIPTION = "Actual spread realized by market makers"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.realized_spreads = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 10:
            return
        prices = self._prices_array()
        midpoints = (prices[:-1] + prices[1:]) / 2
        for i in range(len(midpoints) - 1):
            trade_price = prices[i + 1]
            future_mid = midpoints[min(i + 5, len(midpoints) - 1)]
            spread = 2 * (trade_price - midpoints[i]) * np.sign(trade_price - midpoints[i])
            realized = spread - 2 * (future_mid - midpoints[i]) * np.sign(trade_price - midpoints[i])
            self.realized_spreads.append(realized)
        if len(self.realized_spreads) < 10:
            return
        avg_spread = np.mean(self.realized_spreads)
        if avg_spread > 0.001:
            self.signal = -1
            self.confidence = min(avg_spread * 100, 1.0)
        elif avg_spread < -0.0005:
            self.signal = 1
            self.confidence = min(abs(avg_spread) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(119)
class EffectiveSpread(BaseFormula):
    """ID 119: Effective Spread Measure"""

    CATEGORY = "microstructure"
    NAME = "EffectiveSpread"
    DESCRIPTION = "2 × |P_trade - P_mid|"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.effective_spreads = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 5:
            return
        prices = self._prices_array()
        mid = (np.max(prices[-5:]) + np.min(prices[-5:])) / 2
        current = prices[-1]
        eff_spread = 2 * abs(current - mid)
        self.effective_spreads.append(eff_spread)
        if len(self.effective_spreads) < 10:
            return
        avg_spread = np.mean(self.effective_spreads)
        spread_std = np.std(self.effective_spreads) + 1e-10
        z_spread = (eff_spread - avg_spread) / spread_std
        if z_spread > 2:
            self.signal = -1
            self.confidence = min(z_spread / 4, 1.0)
        elif z_spread < -1:
            self.signal = 1
            self.confidence = min(abs(z_spread) / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(120)
class QuotedSpread(BaseFormula):
    """ID 120: Quoted Spread Dynamics"""

    CATEGORY = "microstructure"
    NAME = "QuotedSpread"
    DESCRIPTION = "Bid-Ask spread from price range"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.quoted_spreads = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 10:
            return
        prices = self._prices_array()
        high = np.max(prices[-10:])
        low = np.min(prices[-10:])
        mid = (high + low) / 2
        spread = (high - low) / mid if mid > 0 else 0
        self.quoted_spreads.append(spread)
        if len(self.quoted_spreads) < 10:
            return
        avg_spread = np.mean(self.quoted_spreads)
        if spread > avg_spread * 1.5:
            self.signal = -1
            self.confidence = min(spread / avg_spread / 3, 1.0)
        elif spread < avg_spread * 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# ORDER FLOW IMBALANCE (121-130)
# =============================================================================

@FormulaRegistry.register(121)
class OrderFlowImbalance(BaseFormula):
    """ID 121: Order Flow Imbalance (OFI)"""

    CATEGORY = "microstructure"
    NAME = "OrderFlowImbalance"
    DESCRIPTION = "OFI = Σ(signed_volume)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.ofi = 0.0
        self.ofi_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        sign = 1 if self.returns[-1] >= 0 else -1
        vol = self.volumes[-1]
        self.ofi += sign * vol
        self.ofi_history.append(sign * vol)
        if len(self.ofi_history) < 10:
            return
        recent_ofi = np.sum(list(self.ofi_history)[-10:])
        avg_vol = np.mean(list(self.volumes)[-10:])
        normalized_ofi = recent_ofi / (avg_vol * 10 + 1)
        if normalized_ofi > 0.3:
            self.signal = 1
            self.confidence = min(normalized_ofi, 1.0)
        elif normalized_ofi < -0.3:
            self.signal = -1
            self.confidence = min(abs(normalized_ofi), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(122)
class CumulativeOFI(BaseFormula):
    """ID 122: Cumulative Order Flow Imbalance"""

    CATEGORY = "microstructure"
    NAME = "CumulativeOFI"
    DESCRIPTION = "Running sum of signed volume"

    def __init__(self, lookback: int = 100, decay: float = 0.99, **kwargs):
        super().__init__(lookback, **kwargs)
        self.decay = decay
        self.cumulative_ofi = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        sign = 1 if self.returns[-1] >= 0 else -1
        vol = self.volumes[-1]
        self.cumulative_ofi = self.decay * self.cumulative_ofi + sign * vol
        avg_vol = np.mean(list(self.volumes)) if len(self.volumes) > 0 else 1
        normalized = self.cumulative_ofi / (avg_vol * len(self.volumes) + 1)
        if normalized > 0.5:
            self.signal = 1
            self.confidence = min(normalized, 1.0)
        elif normalized < -0.5:
            self.signal = -1
            self.confidence = min(abs(normalized), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(123)
class NormalizedOFI(BaseFormula):
    """ID 123: Normalized Order Flow Imbalance"""

    CATEGORY = "microstructure"
    NAME = "NormalizedOFI"
    DESCRIPTION = "OFI / σ(OFI)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.ofi_values = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        sign = 1 if self.returns[-1] >= 0 else -1
        vol = self.volumes[-1]
        ofi = sign * vol
        self.ofi_values.append(ofi)
        if len(self.ofi_values) < 20:
            return
        ofi_arr = np.array(self.ofi_values)
        mean_ofi = np.mean(ofi_arr)
        std_ofi = np.std(ofi_arr) + 1e-10
        z_ofi = (ofi - mean_ofi) / std_ofi
        if z_ofi > 2:
            self.signal = 1
            self.confidence = min(z_ofi / 4, 1.0)
        elif z_ofi < -2:
            self.signal = -1
            self.confidence = min(abs(z_ofi) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(124)
class DeltaOFI(BaseFormula):
    """ID 124: Change in Order Flow Imbalance"""

    CATEGORY = "microstructure"
    NAME = "DeltaOFI"
    DESCRIPTION = "ΔOFI = OFI_t - OFI_{t-1}"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.ofi_values = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        sign = 1 if self.returns[-1] >= 0 else -1
        vol = self.volumes[-1]
        ofi = sign * vol
        self.ofi_values.append(ofi)
        if len(self.ofi_values) < 5:
            return
        current_ofi = np.sum(list(self.ofi_values)[-5:])
        previous_ofi = np.sum(list(self.ofi_values)[-10:-5]) if len(self.ofi_values) >= 10 else 0
        delta_ofi = current_ofi - previous_ofi
        avg_vol = np.mean(list(self.volumes)) if len(self.volumes) > 0 else 1
        normalized_delta = delta_ofi / (avg_vol * 5 + 1)
        if normalized_delta > 0.5:
            self.signal = 1
            self.confidence = min(normalized_delta, 1.0)
        elif normalized_delta < -0.5:
            self.signal = -1
            self.confidence = min(abs(normalized_delta), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(125)
class TradeArrivalRate(BaseFormula):
    """ID 125: Trade Arrival Rate (Intensity)"""

    CATEGORY = "microstructure"
    NAME = "TradeArrivalRate"
    DESCRIPTION = "λ = N_trades / Δt"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.arrival_times = deque(maxlen=lookback)
        self.intensity = 0.0

    def _compute(self) -> None:
        if len(self.timestamps) < 2:
            return
        self.arrival_times.append(self.timestamps[-1])
        if len(self.arrival_times) < 10:
            return
        times = np.array(self.arrival_times)
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if len(intervals) < 5:
            return
        avg_interval = np.mean(intervals)
        self.intensity = 1.0 / (avg_interval + 1e-10)
        intensity_history = 1.0 / (np.mean(intervals[-20:]) + 1e-10) if len(intervals) >= 20 else self.intensity
        if self.intensity > intensity_history * 1.5:
            recent_return = self.returns[-1] if len(self.returns) > 0 else 0
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min(self.intensity / intensity_history / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(126)
class TradeClusterDetector(BaseFormula):
    """ID 126: Trade Cluster Detection"""

    CATEGORY = "microstructure"
    NAME = "TradeClusterDetector"
    DESCRIPTION = "Detect bursts of trading activity"

    def __init__(self, lookback: int = 100, cluster_threshold: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.cluster_threshold = cluster_threshold
        self.cluster_detected = False

    def _compute(self) -> None:
        if len(self.timestamps) < 20:
            return
        times = np.array(self.timestamps)
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if len(intervals) < 10:
            return
        recent_avg = np.mean(intervals[-5:])
        historical_avg = np.mean(intervals)
        self.cluster_detected = recent_avg < historical_avg * self.cluster_threshold
        if self.cluster_detected:
            recent_return = np.mean(list(self.returns)[-5:]) if len(self.returns) >= 5 else 0
            if recent_return > 0:
                self.signal = 1
            else:
                self.signal = -1
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(127)
class VolumeClockOFI(BaseFormula):
    """ID 127: Volume-Clock OFI"""

    CATEGORY = "microstructure"
    NAME = "VolumeClockOFI"
    DESCRIPTION = "OFI sampled by volume buckets"

    def __init__(self, lookback: int = 100, bucket_volume: float = 1e6, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bucket_volume = bucket_volume
        self.current_bucket_vol = 0.0
        self.bucket_ofi = 0.0
        self.bucket_ofis = deque(maxlen=50)

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        sign = 1 if self.returns[-1] >= 0 else -1
        vol = self.volumes[-1]
        self.bucket_ofi += sign * vol
        self.current_bucket_vol += vol
        if self.current_bucket_vol >= self.bucket_volume:
            self.bucket_ofis.append(self.bucket_ofi / self.bucket_volume)
            self.bucket_ofi = 0.0
            self.current_bucket_vol = 0.0
        if len(self.bucket_ofis) < 5:
            return
        recent_ofi = np.mean(list(self.bucket_ofis)[-5:])
        if recent_ofi > 0.3:
            self.signal = 1
            self.confidence = min(recent_ofi, 1.0)
        elif recent_ofi < -0.3:
            self.signal = -1
            self.confidence = min(abs(recent_ofi), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(128)
class AggressorRatio(BaseFormula):
    """ID 128: Aggressor Ratio"""

    CATEGORY = "microstructure"
    NAME = "AggressorRatio"
    DESCRIPTION = "Buy aggressor / (Buy + Sell aggressor)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.buy_aggressor = deque(maxlen=lookback)
        self.sell_aggressor = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 1 or len(self.volumes) < 1:
            return
        vol = self.volumes[-1]
        if self.returns[-1] > 0:
            self.buy_aggressor.append(vol)
            self.sell_aggressor.append(0)
        else:
            self.buy_aggressor.append(0)
            self.sell_aggressor.append(vol)
        if len(self.buy_aggressor) < 10:
            return
        total_buy = sum(self.buy_aggressor)
        total_sell = sum(self.sell_aggressor)
        total = total_buy + total_sell + 1e-10
        ratio = total_buy / total
        if ratio > 0.6:
            self.signal = 1
            self.confidence = (ratio - 0.5) * 2
        elif ratio < 0.4:
            self.signal = -1
            self.confidence = (0.5 - ratio) * 2
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(129)
class PriceReversion(BaseFormula):
    """ID 129: Post-Trade Price Reversion"""

    CATEGORY = "microstructure"
    NAME = "PriceReversion"
    DESCRIPTION = "Measure price reversion after large trades"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.reversions = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 10 or len(self.volumes) < 10:
            return
        volumes = self._volumes_array()
        prices = self._prices_array()
        avg_vol = np.mean(volumes)
        for i in range(5, len(volumes) - 3):
            if volumes[i] > avg_vol * 2:
                initial_move = prices[i] - prices[i-1]
                subsequent_move = prices[i+3] - prices[i]
                if initial_move != 0:
                    reversion = -subsequent_move / initial_move
                    self.reversions.append(reversion)
        if len(self.reversions) < 5:
            return
        avg_reversion = np.mean(self.reversions)
        if avg_reversion > 0.5:
            self.signal = -1
            self.confidence = min(avg_reversion, 1.0)
        elif avg_reversion < -0.2:
            self.signal = 1
            self.confidence = min(abs(avg_reversion), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(130)
class MarketMakerInventory(BaseFormula):
    """ID 130: Market Maker Inventory Model"""

    CATEGORY = "microstructure"
    NAME = "MarketMakerInventory"
    DESCRIPTION = "Infer MM inventory from price dynamics"

    def __init__(self, lookback: int = 100, target_inventory: float = 0.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.target_inventory = target_inventory
        self.inferred_inventory = 0.0
        self.adjustment_speed = 0.1

    def _compute(self) -> None:
        if len(self.returns) < 10 or len(self.volumes) < 10:
            return
        returns = self._returns_array()
        volumes = self._volumes_array()
        for i in range(len(returns)):
            sign = 1 if returns[i] > 0 else -1
            self.inferred_inventory -= sign * volumes[i] * self.adjustment_speed
        self.inferred_inventory *= 0.95
        inv_std = np.std(volumes) * len(volumes) + 1e-10
        normalized_inv = self.inferred_inventory / inv_std
        if normalized_inv > 0.5:
            self.signal = -1
            self.confidence = min(normalized_inv, 1.0)
        elif normalized_inv < -0.5:
            self.signal = 1
            self.confidence = min(abs(normalized_inv), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


__all__ = [
    'KylesLambda', 'KyleObizhaeva', 'AlmgrenChriss', 'LinearPriceImpact',
    'SquareRootImpact', 'PermanentImpact', 'TemporaryImpact', 'PropagatorModel',
    'TransientImpact', 'MetaOrderImpact',
    'VPIN', 'BulkVPIN', 'ToxicityIndex', 'InformedTraderProbability',
    'AdverseSelection', 'InventoryComponent', 'OrderProcessingCost',
    'RealizedSpread', 'EffectiveSpread', 'QuotedSpread',
    'OrderFlowImbalance', 'CumulativeOFI', 'NormalizedOFI', 'DeltaOFI',
    'TradeArrivalRate', 'TradeClusterDetector', 'VolumeClockOFI',
    'AggressorRatio', 'PriceReversion', 'MarketMakerInventory',
]
