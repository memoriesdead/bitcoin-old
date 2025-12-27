"""
RenTech Real-Time Feature Engine
=================================

Calculates all features needed by the 9 validated RenTech formulas in real-time.
Designed for live trading integration with Hyperliquid.

Features calculated:
- anomaly_score: Mahalanobis distance (multivariate anomaly)
- tx_price_correlation_30d: Rolling TX-price correlation
- whale_momentum_7d, tx_momentum_7d: Momentum indicators
- bb_position: Bollinger band position
- rsi_14: Relative Strength Index
- price_vs_ma30: Price deviation from 30-day MA
- tx_z30, whale_z30, value_z30: Rolling z-scores
- volatility_20d: Annualized volatility
- ret_3d, ret_7d, ret_14d: Lookback returns
- regime: K-means derived market regime

Created: 2025-12-16
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeatureSnapshot:
    """Complete feature snapshot at a point in time."""
    timestamp: float
    price: float

    # Price-derived
    ret_1d: float = 0.0
    ret_3d: float = 0.0
    ret_7d: float = 0.0
    ret_14d: float = 0.0
    ret_30d: float = 0.0

    # Moving averages & deviations
    ma30: float = 0.0
    price_vs_ma30: float = 0.0  # % deviation

    # Volatility
    volatility_20d: float = 0.0  # Annualized

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.0  # -1 to +1 (normalized position)

    # RSI
    rsi_14: float = 50.0

    # On-chain z-scores
    tx_z30: float = 0.0
    whale_z30: float = 0.0
    value_z30: float = 0.0

    # Momentum
    tx_momentum_7d: float = 0.0
    whale_momentum_7d: float = 0.0

    # Correlations
    tx_price_correlation_30d: float = 0.0

    # Anomaly detection
    anomaly_score: float = 0.0  # Mahalanobis distance

    # Regime
    regime: str = "NEUTRAL"  # CAPITULATION, BEAR, NEUTRAL, BULL, EUPHORIA


class RenTechFeatures:
    """
    Real-time feature calculation engine for RenTech formulas.

    Maintains rolling windows and calculates all features needed
    by the 9 validated trading formulas.
    """

    def __init__(self, db_path: str = "data/unified_bitcoin.db"):
        """
        Initialize feature engine.

        Args:
            db_path: Path to historical data for warm-up
        """
        self.db_path = db_path

        # Rolling windows
        self.prices = deque(maxlen=200)
        self.timestamps = deque(maxlen=200)
        self.tx_counts = deque(maxlen=200)
        self.whale_counts = deque(maxlen=200)
        self.values = deque(maxlen=200)

        # RSI tracking
        self.gains = deque(maxlen=14)
        self.losses = deque(maxlen=14)

        # Regime clustering (pre-defined centers from historical analysis)
        # These were derived from K-means on 16 years of Bitcoin data
        self.regime_centers = {
            "CAPITULATION": {"ret_7d": -25.0, "volatility": 80.0, "tx_z": -2.0},
            "BEAR": {"ret_7d": -10.0, "volatility": 50.0, "tx_z": -0.5},
            "NEUTRAL": {"ret_7d": 0.0, "volatility": 40.0, "tx_z": 0.0},
            "BULL": {"ret_7d": 10.0, "volatility": 45.0, "tx_z": 0.5},
            "EUPHORIA": {"ret_7d": 25.0, "volatility": 70.0, "tx_z": 1.5},
        }

        # Last calculated features
        self.last_features: Optional[FeatureSnapshot] = None
        self.is_warmed_up = False

        # Try to warm up from database
        self._warm_up()

    def _warm_up(self):
        """Load historical data to warm up rolling windows."""
        try:
            db_path = Path(self.db_path)
            if not db_path.exists():
                logger.warning(f"Database not found at {self.db_path}, starting cold")
                return

            conn = sqlite3.connect(str(db_path))

            # Load last 200 days of prices
            cursor = conn.execute(
                "SELECT date, close FROM prices ORDER BY date DESC LIMIT 200"
            )
            prices = list(reversed(cursor.fetchall()))

            # Load last 200 days of features
            cursor = conn.execute(
                "SELECT date, tx_count, whale_tx_count, total_value_btc "
                "FROM daily_features ORDER BY date DESC LIMIT 200"
            )
            features = list(reversed(cursor.fetchall()))
            conn.close()

            # Build lookup for features
            feature_lookup = {f[0]: f for f in features}

            for date, price in prices:
                self.prices.append(price)
                self.timestamps.append(time.time())  # Approximate

                if date in feature_lookup:
                    _, tx, whale, value = feature_lookup[date]
                    self.tx_counts.append(tx or 0)

                    # Handle whale count (may be bytes)
                    if isinstance(whale, bytes):
                        import struct
                        try:
                            whale = struct.unpack('<q', whale)[0]
                        except:
                            whale = 0
                    self.whale_counts.append(whale or 0)
                    self.values.append(value or 0)
                else:
                    self.tx_counts.append(0)
                    self.whale_counts.append(0)
                    self.values.append(0)

            self.is_warmed_up = len(self.prices) >= 30
            logger.info(f"Warmed up with {len(self.prices)} historical records")

        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
            self.is_warmed_up = False

    def update(
        self,
        price: float,
        tx_count: Optional[int] = None,
        whale_count: Optional[int] = None,
        total_value: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> FeatureSnapshot:
        """
        Update with new data and calculate features.

        Args:
            price: Current BTC price
            tx_count: Daily transaction count (optional, uses last known)
            whale_count: Daily whale transaction count (optional)
            total_value: Daily total value in BTC (optional)
            timestamp: Unix timestamp (optional, uses current time)

        Returns:
            FeatureSnapshot with all calculated features
        """
        ts = timestamp or time.time()

        # Update price window
        self.prices.append(price)
        self.timestamps.append(ts)

        # Update on-chain metrics (use last known if not provided)
        if tx_count is not None:
            self.tx_counts.append(tx_count)
        elif len(self.tx_counts) > 0:
            self.tx_counts.append(self.tx_counts[-1])
        else:
            self.tx_counts.append(0)

        if whale_count is not None:
            self.whale_counts.append(whale_count)
        elif len(self.whale_counts) > 0:
            self.whale_counts.append(self.whale_counts[-1])
        else:
            self.whale_counts.append(0)

        if total_value is not None:
            self.values.append(total_value)
        elif len(self.values) > 0:
            self.values.append(self.values[-1])
        else:
            self.values.append(0)

        # Calculate all features
        features = self._calculate_features(price, ts)
        self.last_features = features

        return features

    def _calculate_features(self, price: float, timestamp: float) -> FeatureSnapshot:
        """Calculate all features from current windows."""
        prices = list(self.prices)
        n = len(prices)

        snapshot = FeatureSnapshot(timestamp=timestamp, price=price)

        if n < 2:
            return snapshot

        # Returns
        if n >= 2:
            snapshot.ret_1d = (price / prices[-2] - 1) * 100
        if n >= 4:
            snapshot.ret_3d = (price / prices[-4] - 1) * 100
        if n >= 8:
            snapshot.ret_7d = (price / prices[-8] - 1) * 100
        if n >= 15:
            snapshot.ret_14d = (price / prices[-15] - 1) * 100
        if n >= 31:
            snapshot.ret_30d = (price / prices[-31] - 1) * 100

        # Moving average and deviation (match backtest: rolling(30).mean())
        if n >= 30:
            snapshot.ma30 = np.mean(prices[-30:])  # Include current like backtest
            snapshot.price_vs_ma30 = (price / snapshot.ma30 - 1) * 100

        # Volatility (annualized, match backtest: sqrt(252))
        if n >= 21:
            returns = np.diff(prices[-21:]) / np.array(prices[-21:-1])
            snapshot.volatility_20d = np.std(returns) * np.sqrt(252) * 100

        # Bollinger Bands
        if n >= 20:
            ma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            snapshot.bb_upper = ma20 + 2 * std20
            snapshot.bb_lower = ma20 - 2 * std20
            if std20 > 0:
                snapshot.bb_position = (price - ma20) / (2 * std20)  # Normalized

        # RSI
        if n >= 15:
            snapshot.rsi_14 = self._calculate_rsi(prices[-15:])

        # On-chain z-scores
        tx_list = list(self.tx_counts)
        whale_list = list(self.whale_counts)
        value_list = list(self.values)

        if len(tx_list) >= 30:
            snapshot.tx_z30 = self._z_score(tx_list[-1], tx_list[-30:])
        if len(whale_list) >= 30:
            snapshot.whale_z30 = self._z_score(whale_list[-1], whale_list[-30:])
        if len(value_list) >= 30:
            snapshot.value_z30 = self._z_score(value_list[-1], value_list[-30:])

        # Momentum
        if len(tx_list) >= 14:
            tx_7d = np.mean(tx_list[-7:])
            tx_14d = np.mean(tx_list[-14:])
            if tx_14d > 0:
                snapshot.tx_momentum_7d = (tx_7d / tx_14d - 1) * 100

        if len(whale_list) >= 14:
            whale_7d = np.mean(whale_list[-7:])
            whale_14d = np.mean(whale_list[-14:])
            if whale_14d > 0:
                snapshot.whale_momentum_7d = (whale_7d / whale_14d - 1) * 100

        # TX-Price Correlation
        if n >= 30 and len(tx_list) >= 30:
            snapshot.tx_price_correlation_30d = self._correlation(
                prices[-30:], tx_list[-30:]
            )

        # Mahalanobis distance (anomaly score)
        snapshot.anomaly_score = self._mahalanobis_distance(snapshot)

        # Regime detection
        snapshot.regime = self._detect_regime(snapshot)

        return snapshot

    def _z_score(self, value: float, window: List[float]) -> float:
        """Calculate z-score of value relative to window."""
        if len(window) < 2:
            return 0.0
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return 0.0
        return (value - mean) / std

    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI from price series."""
        if len(prices) < 2:
            return 50.0

        changes = np.diff(prices)
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-10:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation."""
        if len(x) < 2 or len(y) < 2:
            return 0.0

        x = np.array(x)
        y = np.array(y)

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _mahalanobis_distance(self, snapshot: FeatureSnapshot) -> float:
        """
        Calculate Mahalanobis distance for anomaly detection.

        Uses: tx_z30, whale_z30, value_z30, ret_7d, volatility_20d
        """
        # Feature vector
        features = np.array([
            snapshot.tx_z30,
            snapshot.whale_z30,
            snapshot.value_z30,
            snapshot.ret_7d / 10,  # Scale to similar magnitude
            snapshot.volatility_20d / 50
        ])

        # Assume independent features (diagonal covariance)
        # This is a simplification - could use historical covariance
        means = np.zeros(5)  # Centered features
        stds = np.ones(5)    # Unit variance (already z-scored mostly)

        # Mahalanobis: sqrt(sum((x - mu)^2 / var))
        z_squared = ((features - means) / stds) ** 2
        return np.sqrt(np.sum(z_squared))

    def _detect_regime(self, snapshot: FeatureSnapshot) -> str:
        """
        Detect market regime using distance to pre-computed cluster centers.
        """
        feature_vec = {
            "ret_7d": snapshot.ret_7d,
            "volatility": snapshot.volatility_20d,
            "tx_z": snapshot.tx_z30
        }

        min_distance = float('inf')
        closest_regime = "NEUTRAL"

        for regime, center in self.regime_centers.items():
            distance = 0
            for key, value in center.items():
                distance += ((feature_vec.get(key, 0) - value) ** 2)
            distance = np.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                closest_regime = regime

        return closest_regime

    def get_features(self) -> Optional[FeatureSnapshot]:
        """Get last calculated features."""
        return self.last_features

    def get_feature_dict(self) -> Dict[str, float]:
        """Get features as dictionary for formula evaluation."""
        if self.last_features is None:
            return {}

        f = self.last_features
        return {
            "price": f.price,
            "ret_1d": f.ret_1d,
            "ret_3d": f.ret_3d,
            "ret_7d": f.ret_7d,
            "ret_14d": f.ret_14d,
            "ret_30d": f.ret_30d,
            "ma30": f.ma30,
            "price_vs_ma30": f.price_vs_ma30,
            "volatility_20d": f.volatility_20d,
            "bb_upper": f.bb_upper,
            "bb_lower": f.bb_lower,
            "bb_position": f.bb_position,
            "rsi_14": f.rsi_14,
            "tx_z30": f.tx_z30,
            "whale_z30": f.whale_z30,
            "value_z30": f.value_z30,
            "tx_momentum_7d": f.tx_momentum_7d,
            "whale_momentum_7d": f.whale_momentum_7d,
            "tx_price_correlation_30d": f.tx_price_correlation_30d,
            "anomaly_score": f.anomaly_score,
            "regime": f.regime,
        }

    def is_ready(self) -> bool:
        """Check if enough data for reliable features."""
        return len(self.prices) >= 30


if __name__ == "__main__":
    # Test the feature engine
    print("Testing RenTech Feature Engine")
    print("=" * 60)

    engine = RenTechFeatures()

    # Simulate some price updates
    import random
    price = 100000.0

    for i in range(50):
        price *= (1 + random.gauss(0, 0.02))
        tx = int(300000 + random.gauss(0, 50000))
        whale = int(5000 + random.gauss(0, 1000))
        value = 1000000 + random.gauss(0, 100000)

        features = engine.update(price, tx, whale, value)

        if i % 10 == 0:
            print(f"\nTick {i}: Price=${price:,.0f}")
            print(f"  ret_7d: {features.ret_7d:+.2f}%")
            print(f"  price_vs_ma30: {features.price_vs_ma30:+.2f}%")
            print(f"  rsi_14: {features.rsi_14:.1f}")
            print(f"  tx_z30: {features.tx_z30:.2f}")
            print(f"  anomaly_score: {features.anomaly_score:.2f}")
            print(f"  regime: {features.regime}")

    print("\n" + "=" * 60)
    print("Feature engine test complete")
