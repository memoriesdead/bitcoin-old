"""
Historical Blockchain Scanner - RenTech Style

Scans the full blockchain to extract ALL exchange flows.
This is our data goldmine for finding statistical edges.
"""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from engine.sovereign.blockchain.rpc import BitcoinRPC, RPCBatchProcessor


@dataclass
class HistoricalFlow:
    """Single exchange flow from history."""
    block_height: int
    block_time: int  # Unix timestamp
    txid: str
    exchange: str
    direction: int  # 1 = outflow (LONG), -1 = inflow (SHORT)
    amount_btc: float
    address: str


class HistoricalBlockchainScanner:
    """
    Scan historical blocks to extract all exchange flows.

    This is the foundation of RenTech-style backtesting:
    - Scan millions of transactions
    - Identify exchange flows
    - Build database for hypothesis testing
    """

    def __init__(
        self,
        rpc_host: str = "127.0.0.1",
        rpc_port: int = 8332,
        rpc_user: str = "bitcoin",
        rpc_password: str = "bitcoin",
        exchanges_file: str = "data/exchanges.json",
        db_path: str = "data/historical_flows.db"
    ):
        self.rpc = BitcoinRPC(
            host=rpc_host,
            port=rpc_port,
            user=rpc_user,
            password=rpc_password,
            timeout=120
        )
        self.db_path = Path(db_path)
        self.exchanges_file = Path(exchanges_file)

        # Load exchange addresses
        self.exchange_addresses: Dict[str, str] = {}  # address -> exchange name
        self._load_exchanges()

        # Stats
        self.blocks_scanned = 0
        self.flows_found = 0
        self.txs_processed = 0

    def _load_exchanges(self):
        """Load exchange address database."""
        if not self.exchanges_file.exists():
            print(f"[!] Exchange file not found: {self.exchanges_file}")
            return

        with open(self.exchanges_file) as f:
            data = json.load(f)

        # Flatten all exchanges into address -> name mapping
        for exchange, addresses in data.items():
            if isinstance(addresses, list):
                for addr in addresses:
                    self.exchange_addresses[addr] = exchange
            elif isinstance(addresses, dict):
                for addr in addresses.get('addresses', []):
                    self.exchange_addresses[addr] = exchange

        print(f"[+] Loaded {len(self.exchange_addresses):,} exchange addresses")

    def _init_db(self):
        """Initialize SQLite database for flows."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_height INTEGER,
                block_time INTEGER,
                txid TEXT,
                exchange TEXT,
                direction INTEGER,
                amount_btc REAL,
                address TEXT,
                UNIQUE(txid, address)
            )
        ''')

        c.execute('CREATE INDEX IF NOT EXISTS idx_block_time ON flows(block_time)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_exchange ON flows(exchange)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_direction ON flows(direction)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_amount ON flows(amount_btc)')

        # Progress tracking table
        c.execute('''
            CREATE TABLE IF NOT EXISTS scan_progress (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                updated_at TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print(f"[+] Database initialized: {self.db_path}")

    def _get_last_scanned_block(self) -> int:
        """Get the last block we scanned."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT last_block FROM scan_progress WHERE id = 1')
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0

    def _save_progress(self, block_height: int):
        """Save scan progress."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO scan_progress (id, last_block, updated_at)
            VALUES (1, ?, ?)
        ''', (block_height, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def _save_flows(self, flows: List[HistoricalFlow]):
        """Batch save flows to database."""
        if not flows:
            return

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for flow in flows:
            try:
                c.execute('''
                    INSERT OR IGNORE INTO flows
                    (block_height, block_time, txid, exchange, direction, amount_btc, address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    flow.block_height,
                    flow.block_time,
                    flow.txid,
                    flow.exchange,
                    flow.direction,
                    flow.amount_btc,
                    flow.address
                ))
            except Exception as e:
                pass  # Skip duplicates

        conn.commit()
        conn.close()
        self.flows_found += len(flows)

    def _extract_addresses_from_tx(self, tx: Dict) -> tuple:
        """Extract input and output addresses from transaction."""
        inputs = set()
        outputs = set()
        output_values = {}  # address -> BTC amount

        # Extract inputs (where coins come FROM)
        for vin in tx.get('vin', []):
            if 'prevout' in vin:
                script = vin['prevout'].get('scriptPubKey', {})
                addr = script.get('address')
                if addr:
                    inputs.add(addr)

        # Extract outputs (where coins go TO)
        for vout in tx.get('vout', []):
            script = vout.get('scriptPubKey', {})
            addr = script.get('address')
            value = vout.get('value', 0)
            if addr:
                outputs.add(addr)
                output_values[addr] = output_values.get(addr, 0) + value

        return inputs, outputs, output_values

    def _process_block(self, block: Dict) -> List[HistoricalFlow]:
        """Process a single block and extract exchange flows."""
        flows = []
        block_height = block['height']
        block_time = block['time']

        for tx in block.get('tx', []):
            self.txs_processed += 1
            txid = tx.get('txid', '')

            inputs, outputs, output_values = self._extract_addresses_from_tx(tx)

            # Check for INFLOWS (deposits to exchange)
            # Pattern: non-exchange inputs -> exchange outputs
            for addr in outputs:
                if addr in self.exchange_addresses:
                    exchange = self.exchange_addresses[addr]
                    # Check if NOT from same exchange (internal transfer)
                    input_exchanges = {self.exchange_addresses.get(a) for a in inputs}
                    if exchange not in input_exchanges:
                        amount = output_values.get(addr, 0)
                        if amount >= 0.01:  # Min 0.01 BTC
                            flows.append(HistoricalFlow(
                                block_height=block_height,
                                block_time=block_time,
                                txid=txid,
                                exchange=exchange,
                                direction=-1,  # INFLOW = SHORT signal
                                amount_btc=amount,
                                address=addr
                            ))

            # Check for OUTFLOWS (withdrawals from exchange)
            # Pattern: exchange inputs -> non-exchange outputs
            for addr in inputs:
                if addr in self.exchange_addresses:
                    exchange = self.exchange_addresses[addr]
                    # Check if NOT to same exchange (internal transfer)
                    output_exchanges = {self.exchange_addresses.get(a) for a in outputs}
                    if exchange not in output_exchanges:
                        # Sum all non-exchange outputs as withdrawal amount
                        amount = sum(
                            output_values.get(a, 0)
                            for a in outputs
                            if a not in self.exchange_addresses
                        )
                        if amount >= 0.01:  # Min 0.01 BTC
                            flows.append(HistoricalFlow(
                                block_height=block_height,
                                block_time=block_time,
                                txid=txid,
                                exchange=exchange,
                                direction=1,  # OUTFLOW = LONG signal
                                amount_btc=amount,
                                address=addr
                            ))

        return flows

    def scan(
        self,
        start_block: int = None,
        end_block: int = None,
        batch_size: int = 10,
        progress_callback: Callable = None
    ):
        """
        Scan blockchain for exchange flows.

        Args:
            start_block: Starting block (default: resume from last)
            end_block: Ending block (default: current tip)
            batch_size: Blocks to process before saving
            progress_callback: Called with (current, total, flows_found)
        """
        self._init_db()

        # Get current chain tip
        chain_info = self.rpc.getblockchaininfo()
        current_tip = chain_info['blocks']

        # Determine range
        if start_block is None:
            start_block = self._get_last_scanned_block()
            if start_block > 0:
                start_block += 1  # Resume from next block

        if end_block is None:
            end_block = current_tip

        total_blocks = end_block - start_block + 1

        print(f"\n{'='*60}")
        print(f"HISTORICAL BLOCKCHAIN SCANNER")
        print(f"{'='*60}")
        print(f"Blocks to scan: {start_block:,} to {end_block:,} ({total_blocks:,} blocks)")
        print(f"Exchange addresses loaded: {len(self.exchange_addresses):,}")
        print(f"{'='*60}\n")

        batch_flows = []
        start_time = time.time()

        for height in range(start_block, end_block + 1):
            try:
                # Get block with full tx data
                blockhash = self.rpc.getblockhash(height)
                block = self.rpc.getblock(blockhash, 2)  # verbosity=2 for full tx

                # Process block
                flows = self._process_block(block)
                batch_flows.extend(flows)
                self.blocks_scanned += 1

                # Progress
                if self.blocks_scanned % 100 == 0:
                    elapsed = time.time() - start_time
                    blocks_per_sec = self.blocks_scanned / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_blocks - self.blocks_scanned) / blocks_per_sec if blocks_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600

                    print(f"[{height:,}] Scanned {self.blocks_scanned:,}/{total_blocks:,} blocks | "
                          f"Flows: {self.flows_found:,} | "
                          f"Speed: {blocks_per_sec:.1f} blk/s | "
                          f"ETA: {eta_hours:.1f}h")

                    if progress_callback:
                        progress_callback(self.blocks_scanned, total_blocks, self.flows_found)

                # Save batch
                if len(batch_flows) >= batch_size * 100:  # Save every ~1000 flows
                    self._save_flows(batch_flows)
                    self._save_progress(height)
                    batch_flows = []

            except Exception as e:
                print(f"[!] Error at block {height}: {e}")
                # Save progress and continue
                if batch_flows:
                    self._save_flows(batch_flows)
                    batch_flows = []
                self._save_progress(height - 1)
                time.sleep(1)
                continue

        # Final save
        if batch_flows:
            self._save_flows(batch_flows)
        self._save_progress(end_block)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Blocks scanned: {self.blocks_scanned:,}")
        print(f"Transactions processed: {self.txs_processed:,}")
        print(f"Flows found: {self.flows_found:,}")
        print(f"Time elapsed: {elapsed/3600:.2f} hours")
        print(f"Database: {self.db_path}")
        print(f"{'='*60}\n")

    def get_flow_stats(self) -> Dict:
        """Get statistics from the flow database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        stats = {}

        # Total flows
        c.execute('SELECT COUNT(*) FROM flows')
        stats['total_flows'] = c.fetchone()[0]

        # By direction
        c.execute('SELECT direction, COUNT(*), SUM(amount_btc) FROM flows GROUP BY direction')
        for row in c.fetchall():
            direction = 'outflow' if row[0] == 1 else 'inflow'
            stats[f'{direction}_count'] = row[1]
            stats[f'{direction}_btc'] = row[2]

        # By exchange
        c.execute('SELECT exchange, COUNT(*), SUM(amount_btc) FROM flows GROUP BY exchange ORDER BY COUNT(*) DESC LIMIT 10')
        stats['top_exchanges'] = [(row[0], row[1], row[2]) for row in c.fetchall()]

        # Time range
        c.execute('SELECT MIN(block_time), MAX(block_time) FROM flows')
        row = c.fetchone()
        if row[0]:
            stats['start_date'] = datetime.fromtimestamp(row[0]).isoformat()
            stats['end_date'] = datetime.fromtimestamp(row[1]).isoformat()

        conn.close()
        return stats


def main():
    """Run historical scan."""
    import argparse

    parser = argparse.ArgumentParser(description='Scan blockchain for exchange flows')
    parser.add_argument('--start', type=int, default=None, help='Start block')
    parser.add_argument('--end', type=int, default=None, help='End block')
    parser.add_argument('--host', default='127.0.0.1', help='Bitcoin RPC host')
    parser.add_argument('--port', type=int, default=8332, help='Bitcoin RPC port')
    parser.add_argument('--user', default='bitcoin', help='RPC user')
    parser.add_argument('--password', default='bitcoin', help='RPC password')
    args = parser.parse_args()

    scanner = HistoricalBlockchainScanner(
        rpc_host=args.host,
        rpc_port=args.port,
        rpc_user=args.user,
        rpc_password=args.password
    )

    scanner.scan(start_block=args.start, end_block=args.end)

    # Print stats
    stats = scanner.get_flow_stats()
    print("\nFLOW STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
