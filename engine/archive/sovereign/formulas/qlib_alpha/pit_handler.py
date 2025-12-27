"""
Point-in-Time (PIT) Handler
============================

Ported from Microsoft QLib concepts.

CRITICAL: Prevents lookahead bias by ensuring all features
are computed using only data available at decision time.

The #1 cause of backtest overfitting is lookahead bias:
- Using future data to make past decisions
- Using data that wasn't available at the time
- Incorrectly aligning timestamps

This module enforces strict point-in-time semantics.
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import threading


@dataclass
class PITRecord:
    """A single point-in-time record."""
    timestamp: float  # Unix timestamp when data was KNOWN
    data: Dict[str, Any]
    source: str = "unknown"

    def __post_init__(self):
        # Validate timestamp is reasonable
        if self.timestamp < 1230940800:  # Before Bitcoin genesis
            raise ValueError(f"Timestamp {self.timestamp} is before Bitcoin")
        if self.timestamp > datetime.now().timestamp() + 86400:
            raise ValueError(f"Timestamp {self.timestamp} is in the future")


class PointInTimeHandler:
    """
    Manages point-in-time data access to prevent lookahead bias.

    QLIB CONCEPT: All data must be accessed with a "as_of" timestamp.
    We can only use data that was KNOWN at that timestamp.

    Example:
        handler = PointInTimeHandler()

        # Add data with its knowledge timestamp
        handler.add_record(
            timestamp=1702500000,  # When we KNEW this data
            data={'flow': 10.5, 'price': 42000}
        )

        # Query data as of a specific time
        features = handler.get_features(
            as_of=1702500100,  # Current decision time
            lookback=100       # Only use last 100 seconds
        )
    """

    def __init__(self, max_records: int = 10000, min_delay_seconds: float = 0.0, live_mode: bool = False):
        self.max_records = max_records
        self.min_delay_seconds = min_delay_seconds  # Minimum delay before data is "valid"
        self.live_mode = live_mode  # Skip strict PIT validation for live trading
        self.records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._last_query_time: Optional[float] = None

        # Stats for debugging
        self.stats = {
            'records_added': 0,
            'queries': 0,
            'lookahead_prevented': 0,
        }

    def add_record(self, timestamp: float, data: Dict[str, Any],
                   source: str = "live") -> None:
        """
        Add a record with its point-in-time timestamp.

        Args:
            timestamp: Unix timestamp when this data became KNOWN
            data: Dictionary of feature values
            source: Data source identifier
        """
        record = PITRecord(timestamp=timestamp, data=data, source=source)

        with self._lock:
            # Enforce monotonic timestamps for live data
            if self.records and source == "live":
                last_ts = self.records[-1].timestamp
                if timestamp < last_ts:
                    self.stats['lookahead_prevented'] += 1
                    raise ValueError(
                        f"Non-monotonic timestamp: {timestamp} < {last_ts}. "
                        "This would cause lookahead bias."
                    )

            self.records.append(record)
            self.stats['records_added'] += 1

    def get_features(self, as_of: float, lookback: int = 100,
                     feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get features using only data available at as_of timestamp.

        Args:
            as_of: Decision timestamp - only use data BEFORE this
            lookback: Number of records to include
            feature_names: Specific features to extract (None = all)

        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        with self._lock:
            self.stats['queries'] += 1

            # Filter records: only those with timestamp < as_of
            valid_records = [
                r for r in self.records
                if r.timestamp < as_of
            ]

            if not valid_records:
                return {}

            # Take last N records
            recent = valid_records[-lookback:]

            # Extract features
            if feature_names is None:
                # Get all feature names from first record
                feature_names = list(recent[0].data.keys())

            features = {}
            for name in feature_names:
                values = []
                for r in recent:
                    if name in r.data:
                        values.append(r.data[name])
                if values:
                    features[name] = np.array(values)

            self._last_query_time = as_of
            return features

    def get_latest(self, as_of: float) -> Optional[Dict[str, Any]]:
        """Get the most recent record available at as_of time."""
        with self._lock:
            for r in reversed(self.records):
                if r.timestamp < as_of:
                    return r.data
            return None

    def validate_no_lookahead(self, feature_time: float,
                               decision_time: float) -> bool:
        """
        Validate that a feature timestamp doesn't cause lookahead.

        Args:
            feature_time: When the feature data was generated
            decision_time: When the trading decision is made

        Returns:
            True if valid (no lookahead), raises ValueError if invalid
        """
        if feature_time >= decision_time:
            self.stats['lookahead_prevented'] += 1
            raise ValueError(
                f"LOOKAHEAD DETECTED: Feature time {feature_time} >= "
                f"decision time {decision_time}"
            )
        return True

    def is_valid(self, data_timestamp: float, decision_time: float) -> bool:
        """
        Check if data is valid for use at decision_time (no lookahead).

        This is a non-throwing version of validate_no_lookahead.
        Also enforces minimum delay if configured.

        Args:
            data_timestamp: When the data was generated
            decision_time: When the trading decision is made

        Returns:
            True if valid (no lookahead and meets min delay), False otherwise
        """
        # In live mode, skip strict PIT validation (data is always "now")
        if self.live_mode:
            return True

        # Data must be from before decision time
        if data_timestamp >= decision_time:
            self.stats['lookahead_prevented'] += 1
            return False

        # If min_delay is configured, data must be old enough
        if self.min_delay_seconds > 0:
            age = decision_time - data_timestamp
            if age < self.min_delay_seconds:
                return False

        return True

    def get_stats(self) -> Dict[str, int]:
        """Get handler statistics."""
        with self._lock:
            return {
                **self.stats,
                'current_records': len(self.records),
            }


class PITFlowDatabase:
    """
    Point-in-time wrapper for HistoricalFlowDatabase.

    Ensures all queries respect point-in-time semantics.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()

        # Cache table schema
        self._columns = self._get_columns()

        # Stats
        self.stats = {
            'queries': 0,
            'rows_returned': 0,
            'lookahead_prevented': 0,
        }

    def _get_columns(self) -> List[str]:
        """Get column names from the flows table."""
        cursor = self.conn.execute("PRAGMA table_info(flows)")
        return [row[1] for row in cursor.fetchall()]

    def query_as_of(self, as_of: float, lookback_seconds: float = 3600,
                    columns: Optional[List[str]] = None) -> List[Dict]:
        """
        Query flows available at as_of timestamp.

        Args:
            as_of: Decision timestamp (Unix)
            lookback_seconds: How far back to query
            columns: Specific columns (None = all)

        Returns:
            List of flow records
        """
        with self._lock:
            self.stats['queries'] += 1

            cols = columns if columns else self._columns
            col_str = ", ".join(cols)

            # Critical: timestamp < as_of (strict less than)
            start_time = as_of - lookback_seconds

            query = f"""
                SELECT {col_str} FROM flows
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
            """

            cursor = self.conn.execute(query, (start_time, as_of))
            rows = cursor.fetchall()

            self.stats['rows_returned'] += len(rows)

            return [dict(zip(cols, row)) for row in rows]

    def get_flow_at(self, timestamp: float) -> Optional[Dict]:
        """Get the flow record closest to (but before) timestamp."""
        with self._lock:
            query = """
                SELECT * FROM flows
                WHERE timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            cursor = self.conn.execute(query, (timestamp,))
            row = cursor.fetchone()

            if row:
                return dict(zip(self._columns, row))
            return None

    def validate_training_data(self, features_df, labels_df,
                                feature_lag: float = 0.0) -> bool:
        """
        Validate that training data has no lookahead bias.

        Args:
            features_df: DataFrame with 'timestamp' column
            labels_df: DataFrame with 'timestamp' column
            feature_lag: Minimum time between feature and label

        Returns:
            True if valid, raises ValueError if lookahead detected
        """
        if 'timestamp' not in features_df.columns:
            raise ValueError("features_df must have 'timestamp' column")
        if 'timestamp' not in labels_df.columns:
            raise ValueError("labels_df must have 'timestamp' column")

        # Check that all labels are AFTER their corresponding features
        for idx in range(len(features_df)):
            feat_time = features_df.iloc[idx]['timestamp']
            label_time = labels_df.iloc[idx]['timestamp']

            if feat_time + feature_lag >= label_time:
                self.stats['lookahead_prevented'] += 1
                raise ValueError(
                    f"LOOKAHEAD at index {idx}: feature_time={feat_time}, "
                    f"label_time={label_time}, required_lag={feature_lag}"
                )

        return True

    def get_stats(self) -> Dict[str, int]:
        """Get database query statistics."""
        return dict(self.stats)

    def close(self):
        """Close database connection."""
        self.conn.close()


def validate_no_lookahead(feature_timestamps: np.ndarray,
                          label_timestamps: np.ndarray,
                          min_lag: float = 0.0) -> Tuple[bool, Optional[int]]:
    """
    Standalone function to validate no lookahead in arrays.

    Args:
        feature_timestamps: Array of feature generation times
        label_timestamps: Array of label/target times
        min_lag: Minimum required lag between feature and label

    Returns:
        (is_valid, first_violation_index)
    """
    for i in range(len(feature_timestamps)):
        if feature_timestamps[i] + min_lag >= label_timestamps[i]:
            return False, i
    return True, None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Demo point-in-time handling
    handler = PointInTimeHandler()

    # Simulate adding flow data
    import time
    base_time = time.time() - 100

    for i in range(50):
        handler.add_record(
            timestamp=base_time + i * 2,
            data={
                'flow': np.random.randn() * 10,
                'price': 42000 + np.random.randn() * 100,
                'volume': np.random.randint(100, 1000),
            },
            source="demo"
        )

    # Query features as of "now"
    current_time = time.time()
    features = handler.get_features(
        as_of=current_time,
        lookback=20,
        feature_names=['flow', 'price']
    )

    print("Point-in-Time Handler Demo")
    print("=" * 40)
    print(f"Records added: {handler.stats['records_added']}")
    print(f"Flow values (last 5): {features.get('flow', [])[-5:]}")
    print(f"Stats: {handler.get_stats()}")

    # Try to cause lookahead (should fail)
    try:
        handler.validate_no_lookahead(
            feature_time=current_time + 10,  # Future!
            decision_time=current_time
        )
    except ValueError as e:
        print(f"\nLookahead prevented: {e}")
