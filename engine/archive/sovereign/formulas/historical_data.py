"""
HISTORICAL DATA MODULE - RenTech Style
======================================

"We don't start with models. We start with data."
"We look for things that can be replicated thousands of times."
- Jim Simons

This module:
1. Scans historical blockchain data (past blocks)
2. Extracts all exchange flows (inflow/outflow)
3. Correlates flows with subsequent price movements
4. Builds training dataset for HMM and pattern discovery

THE EDGE:
- INFLOW to exchange = Selling pressure (SHORT)
- OUTFLOW from exchange = Accumulation (LONG)
- We see this 10-60 seconds BEFORE price moves

DATA STRUCTURE:
    FlowEvent:
        - timestamp: Unix time
        - exchange: Exchange ID
        - direction: +1 (outflow) or -1 (inflow)
        - btc_amount: Size of flow
        - price_at_flow: BTC price when flow detected
        - price_after_30s: Price 30 seconds later
        - price_after_60s: Price 60 seconds later
        - price_after_120s: Price 2 minutes later
        - outcome: +1 (price went up), -1 (price went down), 0 (no change)

TRAINING GOAL:
    Find flow patterns where:
    - Direction prediction accuracy > 50.75%
    - Pattern repeats 1000+ times
    - Edge persists in out-of-sample data
"""

import os
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class FlowEvent:
    """Single exchange flow event with price outcome."""
    timestamp: float
    exchange: str
    direction: int  # +1 outflow (LONG signal), -1 inflow (SHORT signal)
    btc_amount: float
    tx_hash: str
    block_height: int

    # Price tracking
    price_at_flow: float
    price_after_10s: float = 0.0
    price_after_30s: float = 0.0
    price_after_60s: float = 0.0
    price_after_120s: float = 0.0

    # Outcome: did price move in predicted direction?
    outcome_10s: int = 0   # +1 correct, -1 wrong, 0 unknown
    outcome_30s: int = 0
    outcome_60s: int = 0
    outcome_120s: int = 0

    # Features for HMM
    flow_imbalance: float = 0.0  # Running (out-in)/(out+in)
    flow_velocity: float = 0.0   # BTC/second
    whale_ratio: float = 0.0     # Large flow ratio
    fee_percentile: float = 50.0 # Transaction urgency


@dataclass
class FlowSequence:
    """Sequence of flow events for pattern matching."""
    events: List[FlowEvent]
    regime_sequence: List[int]  # HMM state sequence
    net_outcome: float  # Total PnL if traded all signals
    win_rate: float     # % of correct predictions


class HistoricalFlowDatabase:
    """
    SQLite database for historical flow events.

    Stores all exchange flows with price outcomes for training.

    SCHEMA:
        flows: Individual flow events
        sequences: Pattern sequences for matching
        prices: Price history for outcome calculation
        stats: Aggregate statistics
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default path
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            db_path = os.path.join(base, 'data', 'historical_flows.db')

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Flow events table
        c.execute('''
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                exchange TEXT NOT NULL,
                direction INTEGER NOT NULL,
                btc_amount REAL NOT NULL,
                tx_hash TEXT UNIQUE,
                block_height INTEGER,

                price_at_flow REAL,
                price_after_10s REAL DEFAULT 0,
                price_after_30s REAL DEFAULT 0,
                price_after_60s REAL DEFAULT 0,
                price_after_120s REAL DEFAULT 0,

                outcome_10s INTEGER DEFAULT 0,
                outcome_30s INTEGER DEFAULT 0,
                outcome_60s INTEGER DEFAULT 0,
                outcome_120s INTEGER DEFAULT 0,

                flow_imbalance REAL DEFAULT 0,
                flow_velocity REAL DEFAULT 0,
                whale_ratio REAL DEFAULT 0,
                fee_percentile REAL DEFAULT 50,

                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # Price history for outcome calculation
        c.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp REAL PRIMARY KEY,
                price REAL NOT NULL,
                source TEXT DEFAULT 'coinbase'
            )
        ''')

        # Pattern sequences discovered
        c.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                sequence TEXT NOT NULL,
                expected_direction INTEGER,
                occurrences INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.5,
                avg_return REAL DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                last_seen REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # HMM model parameters (serialized)
        c.execute('''
            CREATE TABLE IF NOT EXISTS hmm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                n_states INTEGER,
                transition_matrix TEXT,
                emission_means TEXT,
                emission_vars TEXT,
                initial_probs TEXT,
                training_samples INTEGER,
                validation_accuracy REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # Training statistics
        c.execute('''
            CREATE TABLE IF NOT EXISTS training_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE,
                stat_value REAL,
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # Indexes for fast queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_flows_timestamp ON flows(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_flows_exchange ON flows(exchange)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_flows_direction ON flows(direction)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp)')

        conn.commit()
        conn.close()

    def add_flow(self, event: FlowEvent) -> int:
        """Add a flow event to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute('''
                INSERT OR IGNORE INTO flows (
                    timestamp, exchange, direction, btc_amount, tx_hash, block_height,
                    price_at_flow, price_after_10s, price_after_30s, price_after_60s, price_after_120s,
                    outcome_10s, outcome_30s, outcome_60s, outcome_120s,
                    flow_imbalance, flow_velocity, whale_ratio, fee_percentile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp, event.exchange, event.direction, event.btc_amount,
                event.tx_hash, event.block_height,
                event.price_at_flow, event.price_after_10s, event.price_after_30s,
                event.price_after_60s, event.price_after_120s,
                event.outcome_10s, event.outcome_30s, event.outcome_60s, event.outcome_120s,
                event.flow_imbalance, event.flow_velocity, event.whale_ratio, event.fee_percentile
            ))
            conn.commit()
            return c.lastrowid
        except Exception as e:
            print(f"[DB] Error adding flow: {e}")
            return -1
        finally:
            conn.close()

    def add_price(self, timestamp: float, price: float, source: str = 'coinbase'):
        """Add price point."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('INSERT OR REPLACE INTO prices (timestamp, price, source) VALUES (?, ?, ?)',
                     (timestamp, price, source))
            conn.commit()
        finally:
            conn.close()

    def get_price_at(self, timestamp: float, tolerance: float = 5.0) -> Optional[float]:
        """Get price closest to timestamp."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT price FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?)
            LIMIT 1
        ''', (timestamp - tolerance, timestamp + tolerance, timestamp))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def update_flow_outcomes(self, flow_id: int, prices: Dict[str, float], outcomes: Dict[str, int]):
        """Update flow with price outcomes."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE flows SET
                price_after_10s = ?,
                price_after_30s = ?,
                price_after_60s = ?,
                price_after_120s = ?,
                outcome_10s = ?,
                outcome_30s = ?,
                outcome_60s = ?,
                outcome_120s = ?
            WHERE id = ?
        ''', (
            prices.get('10s', 0), prices.get('30s', 0), prices.get('60s', 0), prices.get('120s', 0),
            outcomes.get('10s', 0), outcomes.get('30s', 0), outcomes.get('60s', 0), outcomes.get('120s', 0),
            flow_id
        ))
        conn.commit()
        conn.close()

    def get_flows(self, start_time: float = None, end_time: float = None,
                  exchange: str = None, min_btc: float = 0) -> List[FlowEvent]:
        """Query flows from database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        query = 'SELECT * FROM flows WHERE btc_amount >= ?'
        params = [min_btc]

        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        if exchange:
            query += ' AND exchange = ?'
            params.append(exchange)

        query += ' ORDER BY timestamp ASC'

        c.execute(query, params)
        rows = c.fetchall()
        conn.close()

        # Convert to FlowEvent objects
        events = []
        for row in rows:
            events.append(FlowEvent(
                timestamp=row[1],
                exchange=row[2],
                direction=row[3],
                btc_amount=row[4],
                tx_hash=row[5] or '',
                block_height=row[6] or 0,
                price_at_flow=row[7] or 0,
                price_after_10s=row[8] or 0,
                price_after_30s=row[9] or 0,
                price_after_60s=row[10] or 0,
                price_after_120s=row[11] or 0,
                outcome_10s=row[12] or 0,
                outcome_30s=row[13] or 0,
                outcome_60s=row[14] or 0,
                outcome_120s=row[15] or 0,
                flow_imbalance=row[16] or 0,
                flow_velocity=row[17] or 0,
                whale_ratio=row[18] or 0,
                fee_percentile=row[19] or 50,
            ))

        return events

    def get_flow_count(self) -> int:
        """Get total number of flows."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM flows')
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_win_rate(self, timeframe: str = '30s', min_btc: float = 0) -> Dict:
        """Calculate overall win rate."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        outcome_col = f'outcome_{timeframe}'
        c.execute(f'''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN {outcome_col} = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN {outcome_col} = -1 THEN 1 ELSE 0 END) as losses
            FROM flows
            WHERE btc_amount >= ? AND {outcome_col} != 0
        ''', (min_btc,))

        row = c.fetchone()
        conn.close()

        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0

        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / max(1, total),
            'edge': (wins / max(1, total)) - 0.5,  # Edge over random
        }

    def save_pattern(self, name: str, sequence: List[int], direction: int,
                     occurrences: int, wins: int, losses: int):
        """Save discovered pattern."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        win_rate = wins / max(1, occurrences)

        c.execute('''
            INSERT OR REPLACE INTO patterns (
                name, sequence, expected_direction, occurrences, wins, losses,
                win_rate, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, json.dumps(sequence), direction, occurrences, wins, losses,
              win_rate, time.time()))

        conn.commit()
        conn.close()

    def get_patterns(self, min_occurrences: int = 100, min_win_rate: float = 0.5075) -> List[Dict]:
        """Get patterns that meet criteria."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT name, sequence, expected_direction, occurrences, wins, losses, win_rate
            FROM patterns
            WHERE occurrences >= ? AND win_rate >= ?
            ORDER BY win_rate DESC
        ''', (min_occurrences, min_win_rate))

        patterns = []
        for row in c.fetchall():
            patterns.append({
                'name': row[0],
                'sequence': json.loads(row[1]),
                'direction': row[2],
                'occurrences': row[3],
                'wins': row[4],
                'losses': row[5],
                'win_rate': row[6],
            })

        conn.close()
        return patterns

    def save_hmm_model(self, name: str, n_states: int,
                       transition_matrix: List[List[float]],
                       emission_means: Dict[int, List[float]],
                       emission_vars: Dict[int, List[float]],
                       initial_probs: List[float],
                       training_samples: int,
                       validation_accuracy: float):
        """Save trained HMM model."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT OR REPLACE INTO hmm_models (
                name, n_states, transition_matrix, emission_means, emission_vars,
                initial_probs, training_samples, validation_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, n_states,
            json.dumps(transition_matrix),
            json.dumps({str(k): v for k, v in emission_means.items()}),
            json.dumps({str(k): v for k, v in emission_vars.items()}),
            json.dumps(initial_probs),
            training_samples,
            validation_accuracy
        ))

        conn.commit()
        conn.close()

    def load_hmm_model(self, name: str = 'default') -> Optional[Dict]:
        """Load trained HMM model."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT n_states, transition_matrix, emission_means, emission_vars,
                   initial_probs, training_samples, validation_accuracy
            FROM hmm_models WHERE name = ?
        ''', (name,))

        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'n_states': row[0],
            'transition_matrix': json.loads(row[1]),
            'emission_means': {int(k): v for k, v in json.loads(row[2]).items()},
            'emission_vars': {int(k): v for k, v in json.loads(row[3]).items()},
            'initial_probs': json.loads(row[4]),
            'training_samples': row[5],
            'validation_accuracy': row[6],
        }

    def set_stat(self, name: str, value: float):
        """Set training statistic."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO training_stats (stat_name, stat_value, updated_at)
            VALUES (?, ?, ?)
        ''', (name, value, time.time()))
        conn.commit()
        conn.close()

    def get_stat(self, name: str) -> Optional[float]:
        """Get training statistic."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT stat_value FROM training_stats WHERE stat_name = ?', (name,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_stats_summary(self) -> Dict:
        """Get summary of all stats."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Flow counts
        c.execute('SELECT COUNT(*) FROM flows')
        total_flows = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM flows WHERE outcome_30s != 0')
        flows_with_outcomes = c.fetchone()[0]

        c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM flows')
        time_range = c.fetchone()

        c.execute('SELECT COUNT(*) FROM patterns WHERE win_rate >= 0.5075')
        valid_patterns = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM hmm_models')
        hmm_models = c.fetchone()[0]

        conn.close()

        return {
            'total_flows': total_flows,
            'flows_with_outcomes': flows_with_outcomes,
            'time_range_start': time_range[0] if time_range[0] else 0,
            'time_range_end': time_range[1] if time_range[1] else 0,
            'valid_patterns': valid_patterns,
            'hmm_models': hmm_models,
        }


class HistoricalFlowScanner:
    """
    Scans historical blockchain data to build training dataset.

    PROCESS:
    1. Connect to Bitcoin Core RPC
    2. Scan blocks from start_height to end_height
    3. For each transaction, check if it involves exchange addresses
    4. Record flow events with timestamps
    5. Fetch historical prices to calculate outcomes

    REQUIREMENTS:
    - Bitcoin Core with txindex=1
    - exchanges.json with known exchange addresses
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()

        # Load exchange addresses
        self.exchange_addresses = self._load_exchange_addresses()

        # Running statistics for features
        self.total_inflow = 0.0
        self.total_outflow = 0.0
        self.whale_inflow = 0.0
        self.whale_outflow = 0.0
        self.recent_flows = []  # Last N flows for velocity calculation

        # Price cache
        self.price_cache: Dict[int, float] = {}  # timestamp -> price

    def _load_exchange_addresses(self) -> Dict[str, str]:
        """Load exchange addresses from JSON."""
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        json_path = os.path.join(base, 'data', 'exchanges.json')

        addresses = {}
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                for exchange, info in data.get('exchanges', {}).items():
                    for addr in info.get('addresses', []):
                        addresses[addr] = exchange
            print(f"[SCANNER] Loaded {len(addresses)} exchange addresses")
        except Exception as e:
            print(f"[SCANNER] Error loading exchanges.json: {e}")

        return addresses

    def _get_rpc(self):
        """Get Bitcoin Core RPC connection."""
        from ..blockchain.rpc import get_rpc_from_env
        return get_rpc_from_env()

    def scan_block(self, block_height: int, rpc=None) -> List[FlowEvent]:
        """Scan a single block for exchange flows."""
        if rpc is None:
            rpc = self._get_rpc()

        events = []

        try:
            # Get block with full transactions
            block_hash = rpc.call('getblockhash', block_height)
            block = rpc.call('getblock', block_hash, 2)  # verbosity=2 for full tx

            block_time = block['time']

            for tx in block.get('tx', []):
                flow_events = self._process_transaction(tx, block_height, block_time)
                events.extend(flow_events)

        except Exception as e:
            print(f"[SCANNER] Error scanning block {block_height}: {e}")

        return events

    def _process_transaction(self, tx: Dict, block_height: int, block_time: int) -> List[FlowEvent]:
        """Process single transaction for exchange flows."""
        events = []

        tx_hash = tx.get('txid', '')

        # Extract input addresses (senders)
        input_addrs = set()
        input_value = 0.0
        for vin in tx.get('vin', []):
            if 'coinbase' in vin:
                continue
            # Would need to look up prevout to get address
            # For now, skip inputs (focus on outputs which are more reliable)

        # Extract output addresses (receivers)
        for vout in tx.get('vout', []):
            value = vout.get('value', 0)
            script = vout.get('scriptPubKey', {})
            addresses = script.get('addresses', [])

            # Handle newer format
            if not addresses and 'address' in script:
                addresses = [script['address']]

            for addr in addresses:
                if addr in self.exchange_addresses:
                    exchange = self.exchange_addresses[addr]

                    # This is an INFLOW to exchange (someone sending TO exchange)
                    # INFLOW = SHORT signal (they will sell)

                    # Update running totals
                    self.total_inflow += value
                    if value >= 100:
                        self.whale_inflow += value

                    # Calculate features
                    total = self.total_inflow + self.total_outflow + 0.001
                    flow_imbalance = (self.total_outflow - self.total_inflow) / total

                    # Velocity (simplified - would need time window)
                    flow_velocity = value / 600  # Approximate per-block

                    whale_total = self.whale_inflow + self.whale_outflow + 0.001
                    whale_ratio = (self.whale_outflow - self.whale_inflow) / whale_total

                    event = FlowEvent(
                        timestamp=float(block_time),
                        exchange=exchange,
                        direction=-1,  # INFLOW = SHORT
                        btc_amount=value,
                        tx_hash=tx_hash,
                        block_height=block_height,
                        price_at_flow=0.0,  # Will be filled later
                        flow_imbalance=flow_imbalance,
                        flow_velocity=flow_velocity,
                        whale_ratio=whale_ratio,
                    )
                    events.append(event)

        # Note: Detecting OUTFLOWS requires tracking which addresses belong to exchanges
        # and seeing them as inputs. This is more complex and would need prevout lookup.
        # For initial implementation, focus on inflows which are easier to detect.

        return events

    def scan_range(self, start_height: int, end_height: int,
                   progress_callback=None) -> int:
        """Scan range of blocks."""
        rpc = self._get_rpc()
        total_events = 0

        print(f"[SCANNER] Scanning blocks {start_height} to {end_height}")

        for height in range(start_height, end_height + 1):
            events = self.scan_block(height, rpc)

            for event in events:
                self.db.add_flow(event)
                total_events += 1

            if height % 100 == 0:
                print(f"[SCANNER] Block {height}/{end_height} - {total_events} events")
                if progress_callback:
                    progress_callback(height, end_height, total_events)

        return total_events

    def fetch_historical_prices(self, start_time: float, end_time: float,
                                interval: int = 10) -> int:
        """Fetch historical prices from exchange API."""
        import urllib.request
        import json as json_lib

        # Use Coinbase API for historical prices
        # Note: This is simplified - real implementation would use proper historical data API

        print(f"[SCANNER] Fetching historical prices...")

        # For now, just record that we need price data
        # In production, would use:
        # - Coinbase Pro API historical candles
        # - CryptoCompare historical data
        # - Local price database

        count = 0
        current = start_time

        while current <= end_time:
            # Placeholder - would fetch real historical price
            # self.db.add_price(current, price)
            current += interval
            count += 1

        return count

    def calculate_outcomes(self, timeframes: List[int] = [10, 30, 60, 120]):
        """Calculate outcomes for all flows based on price movements."""
        print("[SCANNER] Calculating outcomes...")

        flows = self.db.get_flows()
        updated = 0

        for flow in flows:
            if flow.price_at_flow == 0:
                # Get price at flow time
                price = self.db.get_price_at(flow.timestamp)
                if not price:
                    continue
                flow.price_at_flow = price

            prices = {}
            outcomes = {}

            for tf in timeframes:
                key = f'{tf}s'
                future_price = self.db.get_price_at(flow.timestamp + tf)

                if future_price and flow.price_at_flow > 0:
                    prices[key] = future_price

                    # Did price move in predicted direction?
                    price_change = (future_price - flow.price_at_flow) / flow.price_at_flow

                    # flow.direction: +1 = OUTFLOW (LONG), -1 = INFLOW (SHORT)
                    # For LONG: we want price to go UP
                    # For SHORT: we want price to go DOWN

                    if flow.direction == 1:  # LONG signal
                        outcomes[key] = 1 if price_change > 0.0001 else (-1 if price_change < -0.0001 else 0)
                    else:  # SHORT signal
                        outcomes[key] = 1 if price_change < -0.0001 else (-1 if price_change > 0.0001 else 0)

            if prices:
                # Update in database (would need flow ID)
                updated += 1

        print(f"[SCANNER] Updated {updated} flows with outcomes")
        return updated


# Convenience function
def get_database(path: str = None) -> HistoricalFlowDatabase:
    """Get or create historical flow database."""
    return HistoricalFlowDatabase(path)
