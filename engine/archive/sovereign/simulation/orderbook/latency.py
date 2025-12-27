"""
Latency Models
==============

Exchange-specific latency models for realistic simulation.
Ported from hftbacktest latency modeling.

Different exchanges have different latency characteristics:
- Binance: 50-150ms typical
- Bybit: 30-100ms typical
- Coinbase: 100-300ms typical
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LatencyComponents:
    """Breakdown of latency components."""
    network_ms: float      # Network round-trip
    processing_ms: float   # Exchange processing
    queue_ms: float        # Order queue delay
    total_ms: float        # Total latency


class LatencyModel(ABC):
    """
    Base class for latency models.

    Models the delay between:
    1. Order submission and acknowledgement
    2. Order acknowledgement and fill
    """

    @abstractmethod
    def get_submission_latency(self) -> float:
        """
        Get latency for order submission (ms).

        Time from client send to exchange acknowledgement.
        """
        pass

    @abstractmethod
    def get_fill_latency(self) -> float:
        """
        Get latency for fill notification (ms).

        Time from fill to client notification.
        """
        pass

    @abstractmethod
    def get_market_data_latency(self) -> float:
        """
        Get latency for market data (ms).

        Time from exchange event to client receipt.
        """
        pass

    def get_total_round_trip(self) -> float:
        """Get total round-trip latency."""
        return self.get_submission_latency() + self.get_fill_latency()

    def get_components(self) -> LatencyComponents:
        """Get detailed latency breakdown."""
        submission = self.get_submission_latency()
        fill = self.get_fill_latency()

        return LatencyComponents(
            network_ms=submission * 0.7,  # Approximate split
            processing_ms=submission * 0.3,
            queue_ms=fill * 0.5,
            total_ms=submission + fill,
        )


class ConstantLatencyModel(LatencyModel):
    """
    Simple constant latency model.

    Useful for deterministic testing.
    """

    def __init__(self, submission_ms: float = 50.0,
                 fill_ms: float = 10.0,
                 market_data_ms: float = 20.0):
        self.submission_ms = submission_ms
        self.fill_ms = fill_ms
        self.market_data_ms = market_data_ms

    def get_submission_latency(self) -> float:
        return self.submission_ms

    def get_fill_latency(self) -> float:
        return self.fill_ms

    def get_market_data_latency(self) -> float:
        return self.market_data_ms


class GaussianLatencyModel(LatencyModel):
    """
    Gaussian latency model with mean and std.

    More realistic than constant but still simple.
    """

    def __init__(self,
                 mean_submission_ms: float = 50.0,
                 std_submission_ms: float = 15.0,
                 mean_fill_ms: float = 10.0,
                 std_fill_ms: float = 5.0,
                 mean_market_data_ms: float = 20.0,
                 std_market_data_ms: float = 8.0):
        self.mean_submission = mean_submission_ms
        self.std_submission = std_submission_ms
        self.mean_fill = mean_fill_ms
        self.std_fill = std_fill_ms
        self.mean_market_data = mean_market_data_ms
        self.std_market_data = std_market_data_ms

    def get_submission_latency(self) -> float:
        return max(1.0, np.random.normal(self.mean_submission, self.std_submission))

    def get_fill_latency(self) -> float:
        return max(1.0, np.random.normal(self.mean_fill, self.std_fill))

    def get_market_data_latency(self) -> float:
        return max(1.0, np.random.normal(self.mean_market_data, self.std_market_data))


class BinanceLatencyModel(LatencyModel):
    """
    Binance-specific latency model.

    Based on empirical measurements:
    - REST API: 50-150ms
    - WebSocket: 10-50ms
    - Fills: 5-20ms additional
    """

    def __init__(self, use_websocket: bool = True,
                 region: str = "us"):
        self.use_websocket = use_websocket
        self.region = region

        # Region-specific base latencies
        self.region_latency = {
            "us": 30.0,
            "eu": 40.0,
            "asia": 20.0,
        }.get(region, 35.0)

    def get_submission_latency(self) -> float:
        """
        Binance order submission latency.

        WebSocket is faster than REST.
        """
        base = self.region_latency

        if self.use_websocket:
            # WebSocket: faster but more variable
            return max(5.0, np.random.gamma(2.0, base / 2))
        else:
            # REST: slower but more consistent
            return max(20.0, np.random.gamma(3.0, base))

    def get_fill_latency(self) -> float:
        """Binance fill notification latency."""
        # Fills are very fast on Binance
        return max(2.0, np.random.exponential(10.0))

    def get_market_data_latency(self) -> float:
        """Binance market data latency."""
        if self.use_websocket:
            return max(5.0, np.random.gamma(2.0, 10.0))
        else:
            return max(30.0, np.random.gamma(3.0, 25.0))


class BybitLatencyModel(LatencyModel):
    """
    Bybit-specific latency model.

    Generally faster than Binance:
    - WebSocket: 5-30ms
    - REST: 20-80ms
    """

    def __init__(self, use_websocket: bool = True):
        self.use_websocket = use_websocket

    def get_submission_latency(self) -> float:
        if self.use_websocket:
            return max(3.0, np.random.gamma(2.0, 8.0))
        else:
            return max(15.0, np.random.gamma(2.5, 20.0))

    def get_fill_latency(self) -> float:
        return max(1.0, np.random.exponential(5.0))

    def get_market_data_latency(self) -> float:
        if self.use_websocket:
            return max(3.0, np.random.gamma(2.0, 5.0))
        else:
            return max(20.0, np.random.gamma(2.5, 15.0))


class CoinbaseLatencyModel(LatencyModel):
    """
    Coinbase-specific latency model.

    Generally slower:
    - WebSocket: 50-150ms
    - REST: 100-300ms
    """

    def __init__(self, use_websocket: bool = True):
        self.use_websocket = use_websocket

    def get_submission_latency(self) -> float:
        if self.use_websocket:
            return max(30.0, np.random.gamma(3.0, 30.0))
        else:
            return max(80.0, np.random.gamma(4.0, 50.0))

    def get_fill_latency(self) -> float:
        return max(10.0, np.random.exponential(20.0))

    def get_market_data_latency(self) -> float:
        if self.use_websocket:
            return max(20.0, np.random.gamma(3.0, 20.0))
        else:
            return max(60.0, np.random.gamma(3.5, 40.0))


class KrakenLatencyModel(LatencyModel):
    """
    Kraken-specific latency model.

    Moderate latency:
    - WebSocket: 30-100ms
    - REST: 80-200ms
    """

    def __init__(self, use_websocket: bool = True):
        self.use_websocket = use_websocket

    def get_submission_latency(self) -> float:
        if self.use_websocket:
            return max(20.0, np.random.gamma(2.5, 25.0))
        else:
            return max(60.0, np.random.gamma(3.0, 40.0))

    def get_fill_latency(self) -> float:
        return max(5.0, np.random.exponential(15.0))

    def get_market_data_latency(self) -> float:
        if self.use_websocket:
            return max(15.0, np.random.gamma(2.5, 15.0))
        else:
            return max(50.0, np.random.gamma(3.0, 30.0))


class AdaptiveLatencyModel(LatencyModel):
    """
    Adaptive latency model that learns from observations.

    Tracks actual latencies and adjusts model parameters.
    """

    def __init__(self, base_model: Optional[LatencyModel] = None):
        self.base_model = base_model or GaussianLatencyModel()

        # Observation buffers
        self.submission_observations: list = []
        self.fill_observations: list = []
        self.market_data_observations: list = []

        # Learned parameters
        self.learned_submission_mean: Optional[float] = None
        self.learned_submission_std: Optional[float] = None

        self.max_observations = 1000

    def record_submission_latency(self, latency_ms: float):
        """Record observed submission latency."""
        self.submission_observations.append(latency_ms)
        if len(self.submission_observations) > self.max_observations:
            self.submission_observations = self.submission_observations[-self.max_observations:]
        self._update_learned_params()

    def record_fill_latency(self, latency_ms: float):
        """Record observed fill latency."""
        self.fill_observations.append(latency_ms)
        if len(self.fill_observations) > self.max_observations:
            self.fill_observations = self.fill_observations[-self.max_observations:]

    def record_market_data_latency(self, latency_ms: float):
        """Record observed market data latency."""
        self.market_data_observations.append(latency_ms)
        if len(self.market_data_observations) > self.max_observations:
            self.market_data_observations = self.market_data_observations[-self.max_observations:]

    def _update_learned_params(self):
        """Update learned parameters from observations."""
        if len(self.submission_observations) >= 30:
            self.learned_submission_mean = np.mean(self.submission_observations)
            self.learned_submission_std = np.std(self.submission_observations)

    def get_submission_latency(self) -> float:
        if self.learned_submission_mean is not None:
            return max(1.0, np.random.normal(
                self.learned_submission_mean,
                self.learned_submission_std or 10.0
            ))
        return self.base_model.get_submission_latency()

    def get_fill_latency(self) -> float:
        if len(self.fill_observations) >= 30:
            return max(1.0, np.random.normal(
                np.mean(self.fill_observations),
                np.std(self.fill_observations)
            ))
        return self.base_model.get_fill_latency()

    def get_market_data_latency(self) -> float:
        if len(self.market_data_observations) >= 30:
            return max(1.0, np.random.normal(
                np.mean(self.market_data_observations),
                np.std(self.market_data_observations)
            ))
        return self.base_model.get_market_data_latency()

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'submission_observations': len(self.submission_observations),
            'fill_observations': len(self.fill_observations),
            'market_data_observations': len(self.market_data_observations),
            'learned_submission_mean': self.learned_submission_mean,
            'learned_submission_std': self.learned_submission_std,
        }


def get_latency_model(exchange: str, use_websocket: bool = True) -> LatencyModel:
    """
    Get appropriate latency model for exchange.

    Args:
        exchange: Exchange name
        use_websocket: Whether using WebSocket connection

    Returns:
        Appropriate LatencyModel instance
    """
    exchange_lower = exchange.lower()

    if "binance" in exchange_lower:
        return BinanceLatencyModel(use_websocket=use_websocket)
    elif "bybit" in exchange_lower:
        return BybitLatencyModel(use_websocket=use_websocket)
    elif "coinbase" in exchange_lower:
        return CoinbaseLatencyModel(use_websocket=use_websocket)
    elif "kraken" in exchange_lower:
        return KrakenLatencyModel(use_websocket=use_websocket)
    else:
        # Default: moderate latency
        return GaussianLatencyModel()


# Alias for backwards compatibility
create_latency_model = get_latency_model


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Latency Model Demo")
    print("=" * 50)

    exchanges = ["binance", "bybit", "coinbase", "kraken"]

    for exchange in exchanges:
        model = get_latency_model(exchange)
        print(f"\n{exchange.upper()} Latency Model:")

        # Sample latencies
        submission_samples = [model.get_submission_latency() for _ in range(100)]
        fill_samples = [model.get_fill_latency() for _ in range(100)]

        print(f"  Submission: {np.mean(submission_samples):.1f}ms "
              f"(std: {np.std(submission_samples):.1f}ms)")
        print(f"  Fill:       {np.mean(fill_samples):.1f}ms "
              f"(std: {np.std(fill_samples):.1f}ms)")
        print(f"  Components: {model.get_components()}")

    # Test adaptive model
    print("\n" + "=" * 50)
    print("Adaptive Model Test:")

    adaptive = AdaptiveLatencyModel()

    # Feed some observations
    for _ in range(50):
        adaptive.record_submission_latency(np.random.normal(75, 20))

    print(f"Stats: {adaptive.get_stats()}")
    print(f"Sample submission: {adaptive.get_submission_latency():.1f}ms")
