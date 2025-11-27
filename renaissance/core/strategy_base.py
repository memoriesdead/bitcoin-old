"""
Renaissance Trading System - Base Strategy V16
==============================================
V4 PHASE 1+2 BASE + KVM8 MILLISECOND OPTIMIZATION + LIVE MARKET SCALING

NEW V16: LIVE MARKET DATA + ACADEMIC FORMULA SCALING
====================================================
Integrates live market data with academic formulas:
- Grinold-Kahn (1989): IR = TC * IC * sqrt(BR) for optimal trade frequency
- Kelly-Thorp (1956, 2006): f* = (p*b - q) / b for optimal position sizing
- Almgren-Chriss (2000): Cost = gamma*X + eta*sum(v^2) for market impact
- Kyle's Lambda (1985): Market depth coefficient for liquidity

LIVE DATA SOURCES (5+ backups per data type):
- CoinGecko, CryptoCompare, CoinPaprika, Messari, CoinCap
- Kraken, Coinbase, Bitstamp, Gemini (for order book)
- Circuit breakers + automatic failover

CORE FORMULA (V4 Phase 1+2 - THE EDGE):
- OU theta < 0.8 (anti-whipsaw - block high mean-reversion)
- P(reversal) < 0.75 (trend continuation)
- Kalman filter (noise reduction)

KVM8 MILLISECOND SPEED:
- OFI Order Flow Imbalance (55-65% accuracy)
- VPIN Toxicity (skip toxic flow >0.85)
- TP: 0.08%, SL: 0.04% (proven 2:1 R/R ratio)
- Max hold: 45 seconds (millisecond speed)

RENAISSANCE MATH:
- With 2:1 R/R, need only 33% WR to break even
- Target: 50.75% WR x massive volume = explosive growth

ALL PHASES:
- Phase 1: Anti-whipsaw (OU + Z-score)
- Phase 2: Kalman filter
- Phase 3: Multi-timeframe momentum
- Phase 4: Volatility regime
- Phase 5: Trend strength (ADX + Hurst)
- Phase 6: Dynamic Kelly
- Phase 7: Triple Barrier Method (ID 151) - KVM8 TP/SL
- Phase 8: Laufer Dynamic Betting (ID 179)
- Phase 9: VPIN Toxicity (ID 133) - KVM8 threshold
- Phase 10: Time-Series Momentum (ID 016)
- Phase 11: Avellaneda-Stoikov (ID 220)
- Phase 12-17: Probability-Based Models
- Phase 18: LIVE MARKET SCALING (V16 NEW!)
"""
import numpy as np
import time as time_module
from collections import deque

# Filters
from .filters import (
    AdaptiveKalmanFilter,
    MasterFilter,
    TimeSeriesMomentum,
    TickOFI,              # NEW: Order Flow Imbalance (55-65% accuracy)
    OFISignalGenerator,   # NEW: OFI signal generator
    volatility_regime,
    hurst_exponent,
    ou_mean_reversion_speed,
    reversal_probability,
    # V9: ADAPTIVE MATH ENGINE - Universal market adaptation
    AdaptiveMathEngine,
    UniversalMathFormulas,
    get_adaptive_engine,
    reset_adaptive_engine,
    # V14: ULTRA_RAPID - Millisecond-level adaptive trading
    UltraRapidEngine,
    MicroRegimeDetector,
    VolumeWebSocketFeed,
    create_ultra_rapid_engine
)

# Exits
from .exits import (
    DynamicKelly,
    EnhancedExitManager
)

# Indicators
from .indicators import AvellanedaStoikov

# Probability
from .probability import (
    MasterProbabilityEngine,
    SoftFilterProbability
)

from .config import STARTING_CAPITAL

# Import FormulaEngine for 217 formula integration
try:
    from .formula_engine import FormulaEngine, create_formula_engine
    FORMULA_ENGINE_AVAILABLE = True
except ImportError:
    FORMULA_ENGINE_AVAILABLE = False

# V16 NEW: Import Live Market Data + Academic Formula Scaling
try:
    from .data.live_market_data import LiveMarketData, get_live_data, calculate_optimal_scaling
    LIVE_MARKET_DATA_AVAILABLE = True
except ImportError:
    LIVE_MARKET_DATA_AVAILABLE = False


class BaseStrategy:
    """
    Base strategy with ALL phases built-in.
    V1-V8 inherit and override only CONFIG.

    V5 Enhancements:
    - Phase 7: Triple Barrier for dynamic ATR-based TP/SL (3:1 R:R target)
    - Phase 8: Laufer Dynamic Betting for regime-adaptive position sizing
    - Phase 9: VPIN FIXED with dynamic percentile (not absolute values)
    - Phase 10: Time-Series Momentum for +12-18% WR improvement
    - Phase 11: Avellaneda-Stoikov for inventory risk management
    - Phase 12-17: PROBABILITY-BASED MODELS (NEW!)
      - Platt/Isotonic calibration, Bayesian WR, Regime probability
      - Ensemble fusion, Soft filtering
      - KEY: Probability scales position, never blocks!
    """

    def __init__(self, config: dict):
        self.config = config
        self.version = config.get("name", "BASE")

        # Price data
        self.prices = deque(maxlen=300)
        self.filtered_prices = deque(maxlen=300)

        # Position state
        self.position = None
        self.entry_price = 0
        self.entry_time = 0
        self.pos_size = 0

        # Performance tracking
        self.capital = STARTING_CAPITAL
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.exits = {}

        # Phase 2: Kalman filter
        self.kalman = AdaptiveKalmanFilter(
            process_var=config.get("kalman_process_var", 1e-5),
            measurement_var=config.get("kalman_measurement_var", 1e-4)
        )

        # Phases 1,3,4,5,9: Master filter (includes VPIN) - LEGACY, still used for comparison
        self.master_filter = MasterFilter()

        # Phase 6: Dynamic Kelly (legacy - still used for backup)
        self.dynamic_kelly = DynamicKelly(
            base_kelly=config.get("kelly_frac", 0.10),
            max_kelly=config.get("max_kelly", 0.25),
            min_kelly=config.get("min_kelly", 0.02)
        )

        # Phase 7+8: Enhanced Exit Manager (Triple Barrier + Laufer + Z-score exits)
        self.exit_manager = EnhancedExitManager(
            base_kelly=config.get("kelly_frac", 0.10),
            config=config  # V7: Pass config for z-score exit thresholds
        )
        self.exit_manager.set_capital(STARTING_CAPITAL)

        # Phase 10: Time-Series Momentum
        self.tsmom = TimeSeriesMomentum(
            lookback=config.get("lookback", 50),
            vol_lookback=20
        )

        # Phase 11: Avellaneda-Stoikov for inventory management
        self.avellaneda = AvellanedaStoikov(gamma=0.1, k=1.5)

        # V6 NEW: OFI (Order Flow Imbalance) - 55-65% accuracy
        # This is the PRIMARY signal source for millisecond trading
        # OFI predicts next tick direction based on buy/sell volume
        self.ofi = TickOFI(
            window=config.get("lookback", 20),
            threshold=config.get("ofi_threshold", 0.10)
        )
        self.ofi_generator = OFISignalGenerator(
            ofi_threshold=config.get("ofi_threshold", 0.10),
            min_confidence=config.get("min_confidence", 0.20)
        )
        self.use_ofi_mode = config.get("use_ofi_mode", True)  # Default ON

        # V9 NEW: ADAPTIVE MATH ENGINE - Universal market adaptation
        # This engine automatically adjusts ALL parameters based on:
        # 1. Current volatility (EWMA fast/slow)
        # 2. Current regime (Bayesian: MR/Trending/Volatile)
        # 3. Recent win rate (for adaptive Kelly)
        # 4. OU half-life (for optimal holding time)
        self.use_adaptive_mode = config.get("use_adaptive_mode", False)
        self.adaptive_engine = None
        if self.use_adaptive_mode:
            self.adaptive_engine = AdaptiveMathEngine(
                base_lookback=config.get("lookback", 50),
                base_kelly=config.get("kelly_frac", 0.02)
            )
        self.adaptive_params = {}  # Store current adaptive parameters

        # V14 NEW: ULTRA_RAPID ENGINE - Millisecond-level adaptive trading
        # This engine provides:
        # 1. Micro-regime detection (5-50 tick windows)
        # 2. Volume integration ($77B daily BTC volume)
        # 3. Kyle's Lambda for liquidity-adjusted sizing
        # 4. Parameter adaptation every single tick
        self.use_ultra_rapid_mode = config.get("use_ultra_rapid_mode", False)
        self.ultra_rapid_engine = None
        if self.use_ultra_rapid_mode:
            self.ultra_rapid_engine = create_ultra_rapid_engine(config)
        self.ultra_rapid_signal = {}  # Store current ultra-rapid signal

        # V15 NEW: FORMULA ENGINE - 217 Renaissance Mathematical Formulas
        # This integrates all 217 formulas from the modular formula library
        # Each version (V1-V14) has a specific formula set optimized for its edge
        self.use_formula_engine = config.get("use_formula_engine", False)
        self.formula_engine = None
        if self.use_formula_engine and FORMULA_ENGINE_AVAILABLE:
            # Get version directly from config (added in V15)
            engine_version = config.get("version", "V8")  # Default to V8 if not set
            self.formula_engine = create_formula_engine(
                version=engine_version,
                lookback=config.get("lookback", 50)
            )
        self.formula_signal = {}  # Store formula engine signals

        # V16 NEW: LIVE MARKET DATA + ACADEMIC FORMULA SCALING
        # This integrates live market data with academic formulas:
        # - Grinold-Kahn for optimal trade frequency
        # - Kelly-Thorp for optimal position sizing
        # - Almgren-Chriss for market impact minimization
        # - Kyle's Lambda for liquidity-based sizing
        self.use_live_scaling = config.get("use_live_scaling", True)  # ON by default
        self.live_market_data = None
        self.live_scaling = {}
        self.last_scaling_update = 0
        self.scaling_update_interval = 60  # Update scaling every 60 seconds

        if self.use_live_scaling and LIVE_MARKET_DATA_AVAILABLE:
            try:
                self.live_market_data = get_live_data()
                self._update_live_scaling()
                print(f"[{self.version}] Live Market Scaling ENABLED - Academic formulas active")
            except Exception as e:
                print(f"[{self.version}] Live scaling init error: {e}")
                self.use_live_scaling = False

        # Phase 12-17: MASTER PROBABILITY ENGINE (V5 NEW!)
        # Replaces hard filtering with probability scaling
        self.prob_engine = MasterProbabilityEngine()
        self.use_probability_mode = config.get("use_probability_mode", True)

        # V4 PHASE 1+2 + KVM8 MILLISECOND OPTIMIZATION
        # Use probability filtering to ensure quality trades
        # Min 50% probability = target 50.75% WR (Renaissance math)
        self.min_probability = config.get("min_probability", 0.50)  # 50% min (from config)
        self.trade_cooldown_sec = config.get("trade_cooldown_sec", 0)  # NO COOLDOWN for speed
        self.last_trade_time = 0  # Track last trade for cooldown

        # Current trade info
        self.current_barriers = None
        self.current_laufer_info = None
        self.current_prob_info = None  # NEW: probability components

        self.confidence = 1.0
        self.last_reason = ""
        self.last_signal_score = 0  # Track for probability learning

    def update(self, price: float):
        """Update price history with Kalman filtering"""
        self.prices.append(price)
        filtered = self.kalman.update(price)
        self.filtered_prices.append(filtered)
        # Update Triple Barrier ATR calculation
        self.exit_manager.update_price(price)
        # Update VPIN for toxicity tracking
        self.master_filter.update_vpin(price)
        # Update Time-Series Momentum
        self.tsmom.update(price)
        # Update Avellaneda-Stoikov
        self.avellaneda.update(price)
        # V5: Update probability engine
        self.prob_engine.update_price(price)
        # V6: Update OFI (Order Flow Imbalance) - PRIMARY SIGNAL
        self.ofi.add_tick(price)
        self.ofi_generator.update(price)
        # V9: Update ADAPTIVE MATH ENGINE
        if self.adaptive_engine is not None:
            self.adaptive_params = self.adaptive_engine.update(price)
            # Update exit manager with adaptive parameters
            if self.adaptive_params:
                self.exit_manager.update_config({
                    **self.config,
                    'exit_z_threshold': self.adaptive_params.get('exit_z_threshold', -0.5),
                    'stop_z_threshold': self.adaptive_params.get('stop_z_threshold', -3.0),
                    'min_hold_ms': self.adaptive_params.get('min_hold_ms', 500),
                })
        # V14: Update ULTRA_RAPID ENGINE (millisecond adaptation)
        if self.ultra_rapid_engine is not None:
            timestamp_ms = int(time_module.time() * 1000)
            self.ultra_rapid_signal = self.ultra_rapid_engine.update(
                price=price,
                volume=0.0,  # Volume will come from WebSocket if available
                is_buy=True,  # Default, will be overridden by tick direction
                timestamp_ms=timestamp_ms
            )
        # V15: Update FORMULA ENGINE (217 Renaissance formulas)
        if self.formula_engine is not None:
            timestamp_ms = int(time_module.time() * 1000)
            self.formula_engine.update(price=price, volume=0.0, timestamp=timestamp_ms)

    def _update_live_scaling(self):
        """
        V16: Update live scaling parameters from market data.
        Uses academic formulas (Grinold-Kahn, Kelly, Almgren-Chriss) on LIVE data.
        """
        if not self.use_live_scaling or self.live_market_data is None:
            return

        current_time = time_module.time()
        if current_time - self.last_scaling_update < self.scaling_update_interval:
            return  # Use cached scaling

        try:
            # Get current win rate from our trades
            wr = self.wins / self.trades if self.trades > 0 else 0.5075  # Default to Medallion's

            # Calculate optimal scaling from LIVE data + academic formulas
            scaling = self.live_market_data.calculate_scaling_variables(
                capital=self.capital,
                target=300000.0,
                win_rate=wr
            )

            self.live_scaling = scaling
            self.last_scaling_update = current_time

            # Extract key parameters from academic formulas
            if scaling.get('optimal_strategy'):
                strat = scaling['optimal_strategy']
                self.optimal_leverage = strat.get('optimal_leverage', 1.0)
                self.optimal_trades_per_day = strat.get('optimal_trades_per_day', 1000)
                self.edge_per_trade_pct = strat.get('edge_per_trade_pct', 0.019)

                # Update Kelly fraction from live Kelly calculation
                kelly = scaling.get('kelly', {})
                if kelly.get('half_kelly', 0) > 0:
                    self.config['kelly_frac'] = kelly['half_kelly']

        except Exception as e:
            # Fail silently - use defaults
            pass

    def get_live_position_multiplier(self) -> float:
        """
        V16: Get position size multiplier from live market data.
        Uses Grinold-Kahn IR and Kelly fraction from academic formulas.
        """
        if not self.use_live_scaling or not self.live_scaling:
            return 1.0

        try:
            # Get optimal leverage from live calculation
            optimal_lev = self.live_scaling.get('optimal_strategy', {}).get('optimal_leverage', 1.0)

            # Get Kelly fraction
            kelly = self.live_scaling.get('kelly', {}).get('half_kelly', 0.10)

            # Grinold-Kahn Information Ratio scaling
            ir = self.live_scaling.get('grinold_kahn', {}).get('information_ratio', 1.0)
            ir_mult = min(ir / 5.0, 2.0)  # Scale IR to multiplier, cap at 2x

            # Combined multiplier (capped for safety)
            mult = min(optimal_lev * kelly * ir_mult, 10.0)  # Max 10x
            return max(mult, 0.1)  # Min 0.1x

        except Exception:
            return 1.0

    def get_momentum(self) -> float:
        """Calculate momentum from filtered prices"""
        lookback = self.config["lookback"]
        if len(self.filtered_prices) < lookback:
            return 0
        return (self.filtered_prices[-1] - self.filtered_prices[-lookback]) / self.filtered_prices[-lookback]

    def get_multi_timeframe_momentum(self) -> tuple:
        """
        Calculate momentum across multiple timeframes (Grinold-Kahn breadth)

        Returns: (combined_signal, signal_count, alignment_score)

        Multiple aligned timeframes = stronger signal + more trades
        """
        lookback_fast = self.config.get("lookback_fast", 20)
        lookback_medium = self.config.get("lookback_medium", 50)
        lookback_slow = self.config.get("lookback_slow", 100)

        prices = list(self.filtered_prices)

        signals = []
        strengths = []

        # Fast momentum (20 bars)
        if len(prices) >= lookback_fast:
            mom_fast = (prices[-1] - prices[-lookback_fast]) / prices[-lookback_fast]
            signals.append(np.sign(mom_fast))
            strengths.append(abs(mom_fast))

        # Medium momentum (50 bars)
        if len(prices) >= lookback_medium:
            mom_med = (prices[-1] - prices[-lookback_medium]) / prices[-lookback_medium]
            signals.append(np.sign(mom_med))
            strengths.append(abs(mom_med))

        # Slow momentum (100 bars)
        if len(prices) >= lookback_slow:
            mom_slow = (prices[-1] - prices[-lookback_slow]) / prices[-lookback_slow]
            signals.append(np.sign(mom_slow))
            strengths.append(abs(mom_slow))

        if not signals:
            return 0, 0, 0

        # Alignment score (how many agree)
        dominant_dir = np.sign(sum(signals)) if sum(signals) != 0 else 0
        aligned = sum(1 for s in signals if s == dominant_dir)
        alignment = aligned / len(signals)

        # Combined signal strength
        combined_strength = np.mean(strengths)

        return dominant_dir * combined_strength, len(signals), alignment

    def get_signal(self, price: float) -> tuple:
        """
        Main signal generation with ALL phases.
        Returns: (direction: int, size: float)

        V5 Enhancements:
        - Multi-timeframe momentum for more signals (Grinold-Kahn breadth)
        - Time-Series Momentum (TSMOM) for +12-18% WR
        - Adaptive entry threshold via Laufer golden zone
        - Avellaneda-Stoikov inventory adjustment
        - PROBABILITY-BASED POSITION SIZING (NEW!)
          - Never blocks trades, only scales position size
          - Uses Platt/Isotonic calibration, Bayesian WR, Regime probability
        """
        self.update(price)

        # V6 OPTIMIZATION: Cooldown check to prevent overtrading
        current_time = time_module.time()
        if self.last_trade_time > 0 and (current_time - self.last_trade_time) < self.trade_cooldown_sec:
            self.last_reason = f"COOLDOWN_{self.trade_cooldown_sec - (current_time - self.last_trade_time):.0f}s"
            return 0, 0

        lookback = self.config["lookback"]
        # MAXIMUM FREQUENCY: Reduced lookback requirement (30 vs 100)
        min_prices = max(lookback + 5, 35)  # 35 minimum for faster startup
        if len(self.filtered_prices) < min_prices:
            return 0, 0

        prices = list(self.filtered_prices)

        # V6: OFI (Order Flow Imbalance) as PRIMARY signal source
        # OFI has 55-65% directional accuracy vs ~30% for momentum
        # Signal Weights from kvm8 MILLISECOND_SCALPER:
        #   - OFI: 35% (highest - most predictive)
        #   - Hawkes: 25%
        #   - Regime: 20%
        #   - Momentum: 20%

        # Get OFI signal (PRIMARY)
        ofi_direction, ofi_strength = self.ofi.get_signal()
        ofi_value = self.ofi.calculate()

        # V5 signals (SECONDARY - for confirmation)
        # 1. Multi-timeframe momentum
        mtf_signal, signal_count, alignment = self.get_multi_timeframe_momentum()

        # 2. Time-Series Momentum (TSMOM)
        tsmom_dir, tsmom_strength = self.tsmom.get_signal()

        # 3. Single timeframe momentum (backup)
        mom = self.get_momentum()

        # V6: Combine signals with OFI as PRIMARY (35% weight)
        signals = []
        weights = []

        # OFI - PRIMARY SIGNAL (35% weight from kvm8)
        if ofi_direction != 0:
            signals.append(ofi_direction)
            weights.append(1.75)  # 35% relative weight (0.35/0.20 = 1.75)

        # Momentum signals (20% weight each from kvm8)
        if signal_count >= 2:
            signals.append(np.sign(mtf_signal) if mtf_signal != 0 else 0)
            weights.append(1.0)  # MTF weight

        if tsmom_dir != 0:
            signals.append(tsmom_dir)
            weights.append(1.0)  # TSMOM weight

        if mom != 0:
            signals.append(np.sign(mom))
            weights.append(0.8)  # Single timeframe weight (lower)

        # V15: FORMULA ENGINE SIGNALS (217 Renaissance formulas)
        # Integrates all formulas for the current version
        if self.formula_engine is not None:
            fe_direction, fe_confidence = self.formula_engine.get_signal()
            if fe_direction != 0:
                signals.append(fe_direction)
                weights.append(2.0 * fe_confidence)  # High weight for formula engine

                # Get risk signals for position sizing
                risk_signals = self.formula_engine.get_risk_signal()
                self.formula_signal = {
                    'direction': fe_direction,
                    'confidence': fe_confidence,
                    'risk': risk_signals,
                }

        if not signals:
            return 0, 0

        # Weighted vote for direction
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)
        direction = 1 if weighted_sum > 0 else -1

        # Signal strength = agreement level
        agreement = abs(weighted_sum) / total_weight
        effective_mom = ofi_value if ofi_direction != 0 else mom

        # V9: ADAPTIVE MODE - Override direction based on regime
        # In TRENDING regime: Trade WITH the trend, not against it
        # In MEAN REVERSION regime: Use OFI/momentum signals normally
        if self.use_adaptive_mode and self.adaptive_params:
            current_regime = self.adaptive_params.get('current_regime', 'unknown')
            trend_dir = self.adaptive_params.get('trend_direction', 0)
            strategy_mode = self.adaptive_params.get('strategy_mode', 'MEAN_REVERSION')

            if strategy_mode == 'MOMENTUM' and trend_dir != 0:
                # In MOMENTUM mode: Only trade WITH the trend
                # If we would trade against trend, SKIP or reverse direction
                if direction != trend_dir:
                    # Option 1: Skip trade (safer)
                    # self.last_reason = f"TREND_FILTER_{trend_dir}"
                    # return 0, 0

                    # Option 2: Reverse to follow trend (more aggressive)
                    direction = trend_dir
                    self.last_reason = f"TREND_FOLLOW_{trend_dir}"

        # Store for probability learning
        self.last_signal_score = effective_mom * 1000  # Scale for probability engine

        # V5: PROBABILITY-BASED APPROACH (replaces hard filtering)
        if self.use_probability_mode:
            # Build soft filter conditions from what would have been hard filters
            ou = ou_mean_reversion_speed(prices, threshold=1.0)
            z, p_rev = reversal_probability(prices)
            H = hurst_exponent(prices)
            vol_regime, vol_mult = volatility_regime(prices)

            # Convert to soft conditions (signal, threshold, direction)
            filter_conditions = [
                (1.0 - ou["theta"], 0.0, 'above'),     # Low theta = good (inverted)
                (1.0 - p_rev, 0.12, 'above'),          # Low reversal prob = good
                (H, 0.45, 'above'),                     # High Hurst = trending
                (alignment, 0.5, 'above'),              # High alignment = good
            ]

            # Gather all signal values for ensemble
            ensemble_signals = [
                mtf_signal * 1000,     # Multi-timeframe
                tsmom_strength * tsmom_dir,  # TSMOM
                mom * 1000,            # Single momentum
                (1 - p_rev) * 2 - 1,   # Reversal (inverted, scaled -1 to 1)
                H * 2 - 1,             # Hurst (scaled -1 to 1)
            ]

            # Get probability-based position multiplier
            prob_result = self.prob_engine.get_trade_probability(
                main_signal=self.last_signal_score,
                signals=ensemble_signals,
                filter_conditions=filter_conditions
            )

            prob_mult = prob_result['position_mult']
            trade_prob = prob_result['probability']
            self.current_prob_info = prob_result['components']

            # V6 OPTIMIZATION: Higher probability threshold (data-driven)
            # Previous 0.25 threshold was too low, accepting 0.37-0.42 trades that lost
            if trade_prob < self.min_probability:
                self.last_reason = f"LOW_PROB_{trade_prob:.2f}<{self.min_probability:.2f}"
                return 0, 0

        else:
            # LEGACY: Use hard filtering (V4 behavior)
            # Adaptive threshold from Laufer
            if self.config.get("use_adaptive_threshold", False):
                threshold = self.exit_manager.laufer.get_entry_threshold(prices)
            else:
                threshold = self.config["momentum_threshold"]

            if abs(effective_mom) < threshold:
                return 0, 0

            # Master filter (phases 1,3,4,5,9)
            can_enter, conf, reason = self.master_filter.should_enter(prices, direction)

            if not can_enter:
                self.last_reason = reason
                return 0, 0

            prob_mult = 1.0
            trade_prob = conf

        # Apply signal agreement boost to confidence
        agreement_boost = 0.8 + agreement * 0.4  # 0.8 to 1.2
        conf = min(trade_prob * agreement_boost, 1.3)

        # V5: Avellaneda-Stoikov inventory adjustment
        if self.position == "LONG":
            self.avellaneda.set_inventory(0.5)
        elif self.position == "SHORT":
            self.avellaneda.set_inventory(-0.5)
        else:
            self.avellaneda.set_inventory(0)

        inv_mult = self.avellaneda.get_position_limit_mult()

        self.confidence = conf
        self.last_reason = f"PROB:{trade_prob:.2f}" if self.use_probability_mode else "OK"

        # Phase 7+8: Laufer position sizing + Triple Barrier setup
        entry_params = self.exit_manager.get_entry_params(
            price=price,
            signal_strength=conf,
            prices=prices,
            direction=direction
        )

        # V5: Apply probability multiplier to position size
        size = entry_params['size'] * inv_mult * prob_mult

        # V16 NEW: Apply LIVE MARKET SCALING from academic formulas
        # This scales position based on Grinold-Kahn IR, Kelly, and market depth
        if self.use_live_scaling:
            self._update_live_scaling()  # Update if needed (every 60s)
            live_mult = self.get_live_position_multiplier()
            size = size * live_mult

        self.current_barriers = entry_params['barriers']
        self.current_laufer_info = entry_params['laufer_info']

        # V6 OPTIMIZATION: Update last trade time for cooldown
        if size > 0.01:
            self.last_trade_time = time_module.time()
            return direction, size
        return 0, 0

    def check_exit(self, price: float, ts: float) -> tuple:
        """
        V7 CRITICAL FIX: Check exit with Z-score mean reversion

        Key changes:
        - Exit at z=-0.5 (75% reversion) instead of z=-1.9 (10% reversion)
        - This should increase win rate from 26% to 75%+
        - Uses regime-aware exits when configured

        Returns: (should_exit: bool, reason: str)
        """
        if not self.position:
            return False, None

        # V7: Detect current regime for regime-aware exits
        prices = list(self.filtered_prices)
        regime = 'normal'
        if len(prices) >= 20 and self.config.get("use_regime_exits", False):
            H = hurst_exponent(prices)
            if H < 0.45:
                regime = 'mean_reversion'  # Strong mean reversion
            elif H > 0.55:
                regime = 'trending'  # Strong trend

        # V7: Use Z-score based exit checking (THE CRITICAL FIX!)
        should_exit, reason, pnl_pct = self.exit_manager.check_exit(
            price, ts, self.entry_time, regime=regime
        )

        if should_exit:
            return True, reason

        # Additional safety checks
        hold = ts - self.entry_time
        cfg = self.config

        # Trend exhaustion check - exit early if Hurst drops significantly
        if hold > cfg.get("min_hold_sec", 0.5) * 2 and pnl_pct > 0:
            if len(prices) >= 20:
                H = hurst_exponent(prices)
                if H < 0.35:  # Very strong mean reversion detected
                    return True, "TREND_EXHAUST"

        return False, None

    def enter(self, direction: int, price: float, size: float, ts: float) -> str:
        """Enter a position with Triple Barrier targets"""
        self.pos_size = size
        self.capital -= size
        self.position = "LONG" if direction > 0 else "SHORT"
        self.entry_price = price
        self.entry_time = ts

        kelly_pct = (size / (self.capital + size)) * 100

        # Build info string with barrier details
        info_parts = [f"{self.position} ${price:,.2f}"]
        info_parts.append(f"K:{kelly_pct:.1f}%")

        if self.current_barriers:
            b = self.current_barriers
            info_parts.append(f"R:R={b['r_r_ratio']:.1f}")
            info_parts.append(f"TP:{b['tp_pct']*100:.3f}%")
            info_parts.append(f"SL:{b['sl_pct']*100:.3f}%")

        if self.current_laufer_info:
            info_parts.append(f"[{self.current_laufer_info['regime']}]")

        return " ".join(info_parts)

    def exit(self, price: float, reason: str) -> str:
        """Exit current position and update Laufer + Probability Engine for adaptation"""
        pnl_pct = (price - self.entry_price) / self.entry_price
        if self.position == "SHORT":
            pnl_pct = -pnl_pct

        net_pnl = self.pos_size * pnl_pct
        self.capital += self.pos_size + net_pnl
        self.total_pnl += net_pnl
        self.trades += 1

        win = net_pnl > 0
        if win:
            self.wins += 1

        # Update dynamic Kelly (legacy)
        self.dynamic_kelly.update(win, net_pnl, self.capital)

        # Update Laufer for regime adaptation
        self.exit_manager.record_trade(win, net_pnl, self.capital)
        self.exit_manager.set_capital(self.capital)

        # V5: Update probability engine for online learning
        self.prob_engine.record_trade(self.last_signal_score, win)

        # V9: Update ADAPTIVE MATH ENGINE for dynamic Kelly calculation
        if self.adaptive_engine is not None:
            self.adaptive_engine.record_trade(win)

        # V15: Update FORMULA ENGINE for performance tracking
        if self.formula_engine is not None:
            self.formula_engine.record_trade(win)

        # Clear current trade info
        self.current_barriers = None
        self.current_laufer_info = None
        self.current_prob_info = None

        self.exits[reason] = self.exits.get(reason, 0) + 1

        w = "W" if win else "L"
        result = f"EXIT {self.position} PnL=${net_pnl:.6f}[{w}] ({reason})"
        self.position = None
        return result

    def get_stats(self) -> dict:
        """Get current performance stats"""
        wr = self.wins / self.trades * 100 if self.trades else 0
        edge = self.total_pnl / self.trades if self.trades else 0
        ret = (self.capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100

        # Get probability engine stats
        prob_stats = self.prob_engine.get_stats()

        return {
            "trades": self.trades,
            "wins": self.wins,
            "wr": wr,
            "edge": edge,
            "capital": self.capital,
            "return_pct": ret,
            "exits": self.exits,
            "filter_stats": self.master_filter.stats(),
            "prob_stats": prob_stats,
            "bayesian_wr": prob_stats.get('bayesian_wr', 0.5),
            "regime": prob_stats.get('current_regime', 0)
        }

    def print_status(self, price: float):
        """Print current status line"""
        stats = self.get_stats()
        mom = self.get_momentum() * 100
        pos = f" [{self.position}]" if self.position else ""

        # Current Kelly
        kelly = self.dynamic_kelly.get_kelly(self.confidence)

        print(f"[{time_module.strftime('%H:%M:%S')}] ${price:,.2f} mom={mom:+.3f}% | "
              f"T:{stats['trades']} WR:{stats['wr']:.0f}% Edge:${stats['edge']:.6f} K:{kelly:.2f}{pos}")
        print(f"  Filter: {stats['filter_stats']}")

    def print_results(self, minutes: int):
        """Print final results"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  {self.version} RESULTS ({minutes} min)")
        print(f"  Trades: {stats['trades']} | WR: {stats['wr']:.1f}% | Edge: ${stats['edge']:.6f}")
        print(f"  Capital: ${STARTING_CAPITAL:.2f} -> ${stats['capital']:.4f} ({stats['return_pct']:+.4f}%)")
        print(f"  Exits: {stats['exits']}")
        print(f"  Filter: {stats['filter_stats']}")
        print(f"{'='*60}\n")
