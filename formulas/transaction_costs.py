"""
Renaissance Formula Library - Transaction Cost Formulas
=========================================================
IDs 317-319: Dynamic Adverse Selection, Price Impact, Complete Transaction Cost

These formulas were MISSING from the original system (F013, F017, F018)
and explain why strategies were losing despite 51% win rate.

CRITICAL FINDING:
- Old model: Fixed ~2 bps cost
- New model: Dynamic 5-15 bps cost depending on conditions
- With 51% WR and 5:3 TP:SL, expected profit was ~0.05%
- But TRUE costs are 0.1-0.3% round-trip (5-15 bps x 2)
- THIS IS WHY WE WERE LOSING!

Sources:
- Kyle (1985): Continuous Auctions and Insider Trading
- Baron & Brogaard (2012): NBER HFT Study
- arxiv.org/pdf/1610.00261: Limit Order Strategic Placement
"""

import numpy as np
from typing import Dict, Any
from collections import deque

from .base import BaseFormula, FormulaRegistry


@FormulaRegistry.register(317)
class DynamicAdverseSelectionFormula(BaseFormula):
    """
    ID 317: Dynamic Adverse Selection (Kyle Model) - F013

    CRITICAL MISSING FORMULA: Adverse selection is DYNAMIC based on order flow!

    Formula: AS = lambda * |OFI| * sigma

    Where:
    - AS: Adverse selection cost (dynamic, NOT static)
    - lambda: Probability of trading with informed trader (0.15 = 15% typical)
    - OFI: Order flow imbalance (-1 to +1)
    - sigma: Current volatility

    Authority Score: 9 x 3 x 10 = 270 (CRITICAL)
    """

    FORMULA_ID = 317
    CATEGORY = "transaction_costs"
    NAME = "Dynamic Adverse Selection"
    DESCRIPTION = "Kyle model for dynamic adverse selection cost based on order flow"

    def __init__(self, lookback: int = 100, lambda_informed: float = 0.15,
                 min_adverse_selection: float = 0.0001, **kwargs):
        super().__init__(lookback, **kwargs)
        self.lambda_informed = lambda_informed
        self.min_adverse_selection = min_adverse_selection
        self.adverse_selection_cost = 0.0
        self.order_flow_imbalance = 0.0
        self.volatility = 0.0
        self.buy_volume = deque(maxlen=lookback)
        self.sell_volume = deque(maxlen=lookback)
        self.volatility_window = kwargs.get('volatility_window', 20)
        self.min_samples = max(self.min_samples, self.volatility_window)

    def update_order_flow(self, buy_volume: float, sell_volume: float):
        """Update order flow data"""
        self.buy_volume.append(buy_volume)
        self.sell_volume.append(sell_volume)

    def _compute(self) -> None:
        """Compute dynamic adverse selection cost"""
        prices = self._prices_array()
        if len(prices) < self.volatility_window:
            return

        # Estimate volatility
        returns = np.diff(np.log(prices[-self.volatility_window:]))
        self.volatility = np.std(returns) if len(returns) > 1 else 0.002

        # Calculate order flow imbalance
        total_buy = sum(self.buy_volume) if self.buy_volume else 0
        total_sell = sum(self.sell_volume) if self.sell_volume else 0
        total_volume = total_buy + total_sell

        if total_volume > 0:
            self.order_flow_imbalance = (total_buy - total_sell) / total_volume
        else:
            self.order_flow_imbalance = 0.0

        # Dynamic Adverse Selection: AS = lambda * |OFI| * sigma
        self.adverse_selection_cost = (
            self.lambda_informed *
            abs(self.order_flow_imbalance) *
            self.volatility
        )

        # Apply minimum (even balanced markets have some adverse selection)
        self.adverse_selection_cost = max(
            self.adverse_selection_cost,
            self.min_adverse_selection
        )

        # Signal: High adverse selection = avoid trading
        if self.adverse_selection_cost < 0.0002:  # < 2 bps
            self.signal = 1 if self.order_flow_imbalance > 0 else -1
            self.confidence = 1.0 - (self.adverse_selection_cost / 0.001)
        elif self.adverse_selection_cost > 0.0005:  # > 5 bps
            self.signal = 0  # Don't trade when adverse selection is high
            self.confidence = 0.2
        else:
            self.signal = 1 if self.order_flow_imbalance > 0 else -1
            self.confidence = 0.5

    def get_adverse_selection(self) -> float:
        """Get current adverse selection cost"""
        return self.adverse_selection_cost

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        state = super().get_state()
        state.update({
            'adverse_selection': self.adverse_selection_cost,
            'order_flow_imbalance': self.order_flow_imbalance,
            'volatility': self.volatility,
            'lambda_informed': self.lambda_informed
        })
        return state


@FormulaRegistry.register(318)
class PriceImpactFormula(BaseFormula):
    """
    ID 318: Price Impact Model - F017

    MISSING FORMULA: Larger orders move price against you.

    Formula: PI = (order_size / market_depth) * (1 / kappa)

    Where:
    - PI: Price impact cost
    - order_size: Size in USD
    - market_depth: Estimated book depth
    - kappa: Order book resilience

    Authority Score: 9 x 2 x 8.5 = 153
    """

    FORMULA_ID = 318
    CATEGORY = "transaction_costs"
    NAME = "Price Impact"
    DESCRIPTION = "Price impact cost based on order size and market depth"

    def __init__(self, lookback: int = 100, market_depth: float = 100000.0,
                 kappa_depth: float = 100.0, max_impact: float = 0.001, **kwargs):
        super().__init__(lookback, **kwargs)
        self.market_depth = market_depth
        self.kappa_depth = kappa_depth
        self.max_impact = max_impact
        self.price_impact = 0.0
        self.current_order_size = 0.0

    def set_order_size(self, order_size_usd: float):
        """Set order size for impact calculation"""
        self.current_order_size = order_size_usd

    def update_market_depth(self, depth_usd: float):
        """Update market depth estimate"""
        if depth_usd > 0:
            self.market_depth = depth_usd

    def _compute(self) -> None:
        """Compute price impact for current order size"""
        if self.current_order_size <= 0:
            self.price_impact = 0.0
            self.signal = 0
            self.confidence = 0.5
            return

        # Guard against division by zero (use safe defaults if not yet calibrated)
        depth = self.market_depth if self.market_depth > 0 else 100000.0
        kappa = self.kappa_depth if self.kappa_depth > 0 else 100.0

        # Price Impact: PI = (size / depth) * (1 / kappa)
        size_ratio = self.current_order_size / depth
        self.price_impact = size_ratio * (1.0 / kappa)

        # Cap at maximum impact
        self.price_impact = min(self.price_impact, self.max_impact)

        # Signal: Low impact = safe to trade larger
        if self.price_impact < 0.0002:  # < 2 bps
            self.signal = 1
            self.confidence = 1.0
        elif self.price_impact > 0.0005:  # > 5 bps
            self.signal = -1  # Reduce size
            self.confidence = min(1.0, self.price_impact * 1000)
        else:
            self.signal = 0
            self.confidence = 0.5

    def get_price_impact(self, order_size_usd: float = None) -> float:
        """Get price impact for specific order size"""
        if order_size_usd is None:
            return self.price_impact
        size_ratio = order_size_usd / self.market_depth
        return min(size_ratio * (1.0 / self.kappa_depth), self.max_impact)

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        state = super().get_state()
        state.update({
            'price_impact': self.price_impact,
            'order_size': self.current_order_size,
            'market_depth': self.market_depth,
            'kappa_depth': self.kappa_depth
        })
        return state


@FormulaRegistry.register(319)
class CompleteTransactionCostFormula(BaseFormula):
    """
    ID 319: Complete Transaction Cost Model - F018

    CRITICAL: This is THE formula that explains why we were losing with 51% WR!

    Total_Cost = Spread + Fees + Slippage + Adverse_Selection + Inventory_Risk + Price_Impact

    Old model: Fixed ~2 bps cost
    New model: Dynamic 5-15 bps cost depending on conditions

    With 51% WR and 5:3 TP:SL, expected profit was ~0.05%
    But TRUE costs are 0.1-0.3% round-trip (5-15 bps x 2)
    THIS IS WHY WE WERE LOSING!
    """

    FORMULA_ID = 319
    CATEGORY = "transaction_costs"
    NAME = "Complete Transaction Cost"
    DESCRIPTION = "Total transaction cost including all dynamic components"

    def __init__(self, lookback: int = 100,
                 spread_cost: float = 0.0002,
                 fee_cost: float = -0.0002,
                 slippage_cost: float = 0.0001,
                 lambda_informed: float = 0.15,
                 gamma_risk: float = 0.1,
                 kappa_depth: float = 100.0,
                 market_depth: float = 100000.0,
                 **kwargs):
        super().__init__(lookback, **kwargs)
        self.spread_cost = spread_cost
        self.fee_cost = fee_cost
        self.slippage_cost = slippage_cost
        self.lambda_informed = lambda_informed
        self.gamma_risk = gamma_risk
        self.kappa_depth = kappa_depth
        self.market_depth = market_depth

        # Sub-formulas
        self.adverse_selection_formula = DynamicAdverseSelectionFormula(
            lookback=lookback, lambda_informed=lambda_informed
        )
        self.price_impact_formula = PriceImpactFormula(
            lookback=lookback, market_depth=market_depth, kappa_depth=kappa_depth
        )

        # State
        self.total_cost = 0.0
        self.cost_breakdown = {}
        self.inventory_position = 0.0
        self.volatility = 0.0
        self.volatility_window = kwargs.get('volatility_window', 20)
        self.min_samples = max(self.min_samples, self.volatility_window)

    def set_inventory(self, inventory: float):
        """Set current inventory position (normalized -1 to +1)"""
        self.inventory_position = np.clip(inventory, -1.0, 1.0)

    def set_order_size(self, order_size_usd: float):
        """Set order size for impact calculation"""
        self.price_impact_formula.set_order_size(order_size_usd)

    def update_order_flow(self, buy_volume: float, sell_volume: float):
        """Update order flow for adverse selection calculation"""
        self.adverse_selection_formula.update_order_flow(buy_volume, sell_volume)

    def _compute(self) -> None:
        """Compute total transaction cost with all dynamic components"""
        prices = self._prices_array()
        if len(prices) < self.volatility_window:
            return

        # Update sub-formulas
        for price in prices[-5:]:
            self.adverse_selection_formula.update(price)
            self.price_impact_formula.update(price)

        # Get volatility
        returns = np.diff(np.log(prices[-self.volatility_window:]))
        self.volatility = np.std(returns) if len(returns) > 1 else 0.002

        # Calculate each component
        # 1. Base static costs
        base_cost = self.spread_cost + self.fee_cost + self.slippage_cost

        # 2. Dynamic Adverse Selection (F013)
        adverse_selection = self.adverse_selection_formula.get_adverse_selection()

        # 3. Inventory Risk Penalty (F014 - from GLFT)
        inventory_risk = (
            self.gamma_risk *
            abs(self.inventory_position) *
            (self.volatility ** 2)
        )

        # 4. Price Impact (F017)
        price_impact = self.price_impact_formula.get_price_impact()

        # Total cost
        self.total_cost = base_cost + adverse_selection + inventory_risk + price_impact

        # Store breakdown
        self.cost_breakdown = {
            'spread': self.spread_cost,
            'fees': self.fee_cost,
            'slippage': self.slippage_cost,
            'adverse_selection': adverse_selection,
            'inventory_risk': inventory_risk,
            'price_impact': price_impact,
            'total': self.total_cost
        }

        # Signal: Only trade if expected profit > total cost
        expected_profit = 0.0005  # 5 bps baseline expectation

        if self.total_cost < expected_profit * 0.5:  # Cost < 50% of profit
            self.signal = 1  # Trade is profitable
            self.confidence = 1.0 - (self.total_cost / expected_profit)
        elif self.total_cost > expected_profit:  # Cost > profit
            self.signal = -1  # Don't trade
            self.confidence = min(1.0, self.total_cost / expected_profit - 0.5)
        else:
            self.signal = 0  # Marginal
            self.confidence = 0.5

    def get_total_cost(self) -> float:
        """Get total transaction cost"""
        return self.total_cost

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get breakdown of all cost components"""
        return self.cost_breakdown.copy()

    def is_trade_profitable(self, expected_profit_per_trade: float) -> bool:
        """Check if trade is profitable after costs"""
        return expected_profit_per_trade > self.total_cost * 2  # Round-trip

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        state = super().get_state()
        state.update({
            'total_cost': self.total_cost,
            'cost_breakdown': self.cost_breakdown,
            'inventory': self.inventory_position,
            'volatility': self.volatility
        })
        return state


class TransactionCostAggregator:
    """
    Aggregates all transaction cost formulas (F013-F018)
    for comprehensive cost analysis.
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.cost_formula = CompleteTransactionCostFormula(
            lookback=config.get('lookback', 100),
            spread_cost=config.get('spread_cost', 0.0002),
            fee_cost=config.get('fee_cost', -0.0002),
            slippage_cost=config.get('slippage_cost', 0.0001),
            lambda_informed=config.get('lambda_informed', 0.15),
            gamma_risk=config.get('gamma_risk', 0.1),
            kappa_depth=config.get('kappa_depth', 100.0),
            market_depth=config.get('market_depth', 100000.0)
        )

    def update(self, price: float, buy_volume: float = 0.0,
               sell_volume: float = 0.0, order_size_usd: float = 0.0,
               inventory: float = 0.0):
        """Update with market data"""
        self.cost_formula.update(price)
        self.cost_formula.update_order_flow(buy_volume, sell_volume)
        self.cost_formula.set_order_size(order_size_usd)
        self.cost_formula.set_inventory(inventory)

    def get_total_cost(self) -> float:
        """Get total transaction cost"""
        return self.cost_formula.get_total_cost()

    def get_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown"""
        return self.cost_formula.get_cost_breakdown()

    def should_trade(self, expected_profit: float) -> bool:
        """Check if trade is worth it"""
        return self.cost_formula.is_trade_profitable(expected_profit)

    def get_signal(self) -> int:
        """Get trading signal (-1, 0, 1)"""
        return self.cost_formula.get_signal()

    def get_confidence(self) -> float:
        """Get signal confidence"""
        return self.cost_formula.get_confidence()
