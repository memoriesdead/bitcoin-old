"""
Address Database - SQLite persistence for discovered addresses.
Survives restarts. Grows with every block scanned.
"""
import sqlite3
import json
import os
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from contextlib import contextmanager


class AddressDatabase:
    """
    Persistent storage for blockchain-discovered addresses.

    Stores:
    - Address profiles (tx counts, balances, classification)
    - Entity clusters (groups of addresses = one actor)
    - Consolidation history (50+ input transactions)
    - Exchange labels (propagated from seeds and patterns)
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default path - create data directory if needed
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "addresses.db")

        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            # Main address table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS addresses (
                    address TEXT PRIMARY KEY,
                    entity_id TEXT,
                    classification TEXT,
                    exchange_id TEXT,
                    confidence REAL DEFAULT 0.0,
                    first_seen INTEGER DEFAULT 0,
                    last_seen INTEGER DEFAULT 0,
                    tx_count INTEGER DEFAULT 0,
                    receive_count INTEGER DEFAULT 0,
                    send_count INTEGER DEFAULT 0,
                    total_received REAL DEFAULT 0.0,
                    total_sent REAL DEFAULT 0.0,
                    consolidation_count INTEGER DEFAULT 0,
                    is_hot_wallet INTEGER DEFAULT 0,
                    active_hours INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Entity clusters table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    label TEXT,
                    exchange_id TEXT,
                    address_count INTEGER DEFAULT 0,
                    total_received REAL DEFAULT 0.0,
                    total_sent REAL DEFAULT 0.0,
                    classification TEXT,
                    confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Consolidation transactions (definitive exchange markers)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS consolidations (
                    txid TEXT PRIMARY KEY,
                    block_height INTEGER,
                    input_count INTEGER,
                    output_count INTEGER,
                    total_btc REAL,
                    input_addresses TEXT,
                    output_addresses TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Entity links (address pairs in same entity)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS entity_links (
                    addr1 TEXT,
                    addr2 TEXT,
                    link_type TEXT,
                    txid TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (addr1, addr2)
                )
            ''')

            # Known exchange seeds (bootstrapping)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS exchange_seeds (
                    address TEXT PRIMARY KEY,
                    exchange_id TEXT NOT NULL,
                    label TEXT,
                    source TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Scan progress tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scan_progress (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_block_scanned INTEGER DEFAULT 0,
                    total_addresses INTEGER DEFAULT 0,
                    total_consolidations INTEGER DEFAULT 0,
                    last_scan_time TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for fast lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addr_entity ON addresses(entity_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addr_exchange ON addresses(exchange_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addr_class ON addresses(classification)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addr_hot ON addresses(is_hot_wallet)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_exchange ON entities(exchange_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cons_block ON consolidations(block_height)')

            # Initialize scan progress if empty
            conn.execute('''
                INSERT OR IGNORE INTO scan_progress (id, last_block_scanned)
                VALUES (1, 0)
            ''')

    # ==========================================================================
    # ADDRESS OPERATIONS
    # ==========================================================================

    def upsert_address(self, address: str, **kwargs):
        """Insert or update an address."""
        with self._get_conn() as conn:
            # Build update clause dynamically
            fields = ['address']
            values = [address]
            updates = []

            for key, value in kwargs.items():
                fields.append(key)
                values.append(value)
                updates.append(f"{key} = excluded.{key}")

            placeholders = ','.join(['?' for _ in values])
            field_str = ','.join(fields)
            update_str = ','.join(updates) if updates else 'address = address'

            conn.execute(f'''
                INSERT INTO addresses ({field_str})
                VALUES ({placeholders})
                ON CONFLICT(address) DO UPDATE SET
                {update_str},
                updated_at = CURRENT_TIMESTAMP
            ''', values)

    def upsert_addresses_batch(self, addresses: List[Dict]):
        """Batch insert/update addresses."""
        with self._get_conn() as conn:
            for addr_data in addresses:
                address = addr_data.pop('address')
                fields = ['address'] + list(addr_data.keys())
                values = [address] + list(addr_data.values())
                updates = [f"{k} = excluded.{k}" for k in addr_data.keys()]

                placeholders = ','.join(['?' for _ in values])
                field_str = ','.join(fields)
                update_str = ','.join(updates) if updates else 'address = address'

                conn.execute(f'''
                    INSERT INTO addresses ({field_str})
                    VALUES ({placeholders})
                    ON CONFLICT(address) DO UPDATE SET
                    {update_str},
                    updated_at = CURRENT_TIMESTAMP
                ''', values)

    def get_address(self, address: str) -> Optional[Dict]:
        """Get address profile."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM addresses WHERE address = ?',
                [address]
            ).fetchone()
            return dict(row) if row else None

    def get_addresses_by_exchange(self, exchange_id: str) -> List[str]:
        """Get all addresses for an exchange."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT address FROM addresses WHERE exchange_id = ?',
                [exchange_id]
            ).fetchall()
            return [row['address'] for row in rows]

    def get_all_exchange_addresses(self) -> Dict[str, List[str]]:
        """Get all addresses grouped by exchange."""
        with self._get_conn() as conn:
            rows = conn.execute('''
                SELECT address, exchange_id
                FROM addresses
                WHERE exchange_id IS NOT NULL AND exchange_id != ''
            ''').fetchall()

            result = defaultdict(list)
            for row in rows:
                result[row['exchange_id']].append(row['address'])
            return dict(result)

    def get_exchange_addresses_set(self) -> Set[str]:
        """Get set of all exchange addresses (for fast lookup)."""
        with self._get_conn() as conn:
            rows = conn.execute('''
                SELECT address FROM addresses
                WHERE classification LIKE 'exchange%'
                   OR exchange_id IS NOT NULL
                   OR is_hot_wallet = 1
                   OR consolidation_count > 0
            ''').fetchall()
            return {row['address'] for row in rows}

    def get_hot_wallets(self) -> List[str]:
        """Get all hot wallet addresses."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT address FROM addresses WHERE is_hot_wallet = 1'
            ).fetchall()
            return [row['address'] for row in rows]

    def get_whale_addresses(self) -> List[str]:
        """Get whale addresses (large balance, not exchange)."""
        with self._get_conn() as conn:
            rows = conn.execute('''
                SELECT address FROM addresses
                WHERE classification = 'whale'
                  AND (exchange_id IS NULL OR exchange_id = '')
            ''').fetchall()
            return [row['address'] for row in rows]

    # ==========================================================================
    # ENTITY OPERATIONS
    # ==========================================================================

    def upsert_entity(self, entity_id: str, **kwargs):
        """Insert or update an entity."""
        with self._get_conn() as conn:
            fields = ['entity_id']
            values = [entity_id]
            updates = []

            for key, value in kwargs.items():
                fields.append(key)
                values.append(value)
                updates.append(f"{key} = excluded.{key}")

            placeholders = ','.join(['?' for _ in values])
            field_str = ','.join(fields)
            update_str = ','.join(updates) if updates else 'entity_id = entity_id'

            conn.execute(f'''
                INSERT INTO entities ({field_str})
                VALUES ({placeholders})
                ON CONFLICT(entity_id) DO UPDATE SET
                {update_str},
                updated_at = CURRENT_TIMESTAMP
            ''', values)

    def assign_addresses_to_entity(self, entity_id: str, addresses: List[str]):
        """Assign addresses to an entity."""
        with self._get_conn() as conn:
            for addr in addresses:
                conn.execute(
                    'UPDATE addresses SET entity_id = ? WHERE address = ?',
                    [entity_id, addr]
                )

    def label_entity(self, entity_id: str, exchange_id: str):
        """Label an entity with exchange ID and propagate to all addresses."""
        with self._get_conn() as conn:
            # Update entity
            conn.execute(
                'UPDATE entities SET exchange_id = ? WHERE entity_id = ?',
                [exchange_id, entity_id]
            )
            # Propagate to all addresses in entity
            conn.execute(
                'UPDATE addresses SET exchange_id = ? WHERE entity_id = ?',
                [exchange_id, entity_id]
            )

    # ==========================================================================
    # CONSOLIDATION OPERATIONS
    # ==========================================================================

    def add_consolidation(self, txid: str, block_height: int, input_count: int,
                         output_count: int, total_btc: float,
                         input_addresses: List[str], output_addresses: List[str]):
        """Record a consolidation transaction."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO consolidations
                (txid, block_height, input_count, output_count, total_btc,
                 input_addresses, output_addresses)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [
                txid, block_height, input_count, output_count, total_btc,
                json.dumps(input_addresses), json.dumps(output_addresses)
            ])

    def get_consolidation_addresses(self) -> Tuple[Set[str], Set[str]]:
        """Get all addresses from consolidations (inputs=deposits, outputs=hot wallets)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT input_addresses, output_addresses FROM consolidations'
            ).fetchall()

            inputs = set()
            outputs = set()

            for row in rows:
                inputs.update(json.loads(row['input_addresses']))
                outputs.update(json.loads(row['output_addresses']))

            return inputs, outputs

    # ==========================================================================
    # ENTITY LINKS
    # ==========================================================================

    def add_entity_link(self, addr1: str, addr2: str, link_type: str = 'common_input',
                       txid: str = None):
        """Add a link between two addresses (same entity)."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO entity_links (addr1, addr2, link_type, txid)
                VALUES (?, ?, ?, ?)
            ''', [addr1, addr2, link_type, txid])

    def add_entity_links_batch(self, links: List[Tuple[str, str, str]]):
        """Batch add entity links."""
        with self._get_conn() as conn:
            conn.executemany('''
                INSERT OR IGNORE INTO entity_links (addr1, addr2, link_type)
                VALUES (?, ?, ?)
            ''', links)

    # ==========================================================================
    # EXCHANGE SEEDS
    # ==========================================================================

    def add_exchange_seed(self, address: str, exchange_id: str, label: str = None,
                         source: str = 'manual'):
        """Add a known exchange seed address."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO exchange_seeds (address, exchange_id, label, source)
                VALUES (?, ?, ?, ?)
            ''', [address, exchange_id, label, source])

            # Also update addresses table
            conn.execute('''
                INSERT INTO addresses (address, exchange_id, classification)
                VALUES (?, ?, 'exchange_hot')
                ON CONFLICT(address) DO UPDATE SET
                exchange_id = excluded.exchange_id,
                updated_at = CURRENT_TIMESTAMP
            ''', [address, exchange_id])

    def add_exchange_seeds_batch(self, seeds: Dict[str, List[str]]):
        """Batch add exchange seeds."""
        with self._get_conn() as conn:
            for exchange_id, addresses in seeds.items():
                for addr in addresses:
                    conn.execute('''
                        INSERT OR REPLACE INTO exchange_seeds
                        (address, exchange_id, source)
                        VALUES (?, ?, 'batch')
                    ''', [addr, exchange_id])

                    conn.execute('''
                        INSERT INTO addresses (address, exchange_id, classification)
                        VALUES (?, ?, 'exchange_hot')
                        ON CONFLICT(address) DO UPDATE SET
                        exchange_id = excluded.exchange_id,
                        updated_at = CURRENT_TIMESTAMP
                    ''', [addr, exchange_id])

    def get_exchange_seeds(self) -> Dict[str, List[str]]:
        """Get all exchange seed addresses."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT address, exchange_id FROM exchange_seeds'
            ).fetchall()

            result = defaultdict(list)
            for row in rows:
                result[row['exchange_id']].append(row['address'])
            return dict(result)

    # ==========================================================================
    # SCAN PROGRESS
    # ==========================================================================

    def get_scan_progress(self) -> Dict:
        """Get current scan progress."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT * FROM scan_progress WHERE id = 1').fetchone()
            return dict(row) if row else {'last_block_scanned': 0}

    def update_scan_progress(self, last_block: int, total_addresses: int = None,
                            total_consolidations: int = None):
        """Update scan progress."""
        with self._get_conn() as conn:
            updates = ['last_block_scanned = ?', 'last_scan_time = CURRENT_TIMESTAMP']
            values = [last_block]

            if total_addresses is not None:
                updates.append('total_addresses = ?')
                values.append(total_addresses)

            if total_consolidations is not None:
                updates.append('total_consolidations = ?')
                values.append(total_consolidations)

            conn.execute(f'''
                UPDATE scan_progress
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            ''', values)

    # ==========================================================================
    # STATISTICS
    # ==========================================================================

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._get_conn() as conn:
            stats = {}

            # Address counts
            stats['total_addresses'] = conn.execute(
                'SELECT COUNT(*) FROM addresses'
            ).fetchone()[0]

            stats['exchange_addresses'] = conn.execute(
                "SELECT COUNT(*) FROM addresses WHERE exchange_id IS NOT NULL AND exchange_id != ''"
            ).fetchone()[0]

            stats['hot_wallets'] = conn.execute(
                'SELECT COUNT(*) FROM addresses WHERE is_hot_wallet = 1'
            ).fetchone()[0]

            stats['consolidation_participants'] = conn.execute(
                'SELECT COUNT(*) FROM addresses WHERE consolidation_count > 0'
            ).fetchone()[0]

            # Entity counts
            stats['total_entities'] = conn.execute(
                'SELECT COUNT(*) FROM entities'
            ).fetchone()[0]

            stats['labeled_entities'] = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE exchange_id IS NOT NULL AND exchange_id != ''"
            ).fetchone()[0]

            # Consolidation counts
            stats['total_consolidations'] = conn.execute(
                'SELECT COUNT(*) FROM consolidations'
            ).fetchone()[0]

            # Scan progress
            progress = self.get_scan_progress()
            stats['last_block_scanned'] = progress.get('last_block_scanned', 0)

            # Exchange breakdown
            exchange_counts = conn.execute('''
                SELECT exchange_id, COUNT(*) as count
                FROM addresses
                WHERE exchange_id IS NOT NULL AND exchange_id != ''
                GROUP BY exchange_id
                ORDER BY count DESC
            ''').fetchall()

            stats['exchanges'] = {row['exchange_id']: row['count'] for row in exchange_counts}

            return stats

    def print_stats(self):
        """Print database statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("ADDRESS DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total Addresses:          {stats['total_addresses']:,}")
        print(f"Exchange Addresses:       {stats['exchange_addresses']:,}")
        print(f"Hot Wallets:              {stats['hot_wallets']:,}")
        print(f"Consolidation Addresses:  {stats['consolidation_participants']:,}")
        print(f"Total Entities:           {stats['total_entities']:,}")
        print(f"Labeled Entities:         {stats['labeled_entities']:,}")
        print(f"Consolidations Found:     {stats['total_consolidations']:,}")
        print(f"Last Block Scanned:       {stats['last_block_scanned']:,}")
        print("-" * 60)
        print("EXCHANGE BREAKDOWN:")
        for ex_id, count in sorted(stats['exchanges'].items(), key=lambda x: -x[1]):
            print(f"  {ex_id}: {count:,}")
        print("=" * 60)


def get_database() -> AddressDatabase:
    """Get default database instance."""
    return AddressDatabase()
