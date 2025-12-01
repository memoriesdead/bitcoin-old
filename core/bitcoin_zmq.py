#!/usr/bin/env python3
"""
BITCOIN CORE ZMQ MEMPOOL INTEGRATION
====================================
Real-time mempool data directly from Bitcoin Core via ZeroMQ.

Why This Matters:
- See transactions 10-60 MINUTES before they confirm
- Detect whale movements in mempool
- Front-run large orders (MEV extraction)
- TRUE 100% coverage of all Bitcoin transactions

Setup Required:
1. Install Bitcoin Core
2. Add to bitcoin.conf:
   zmqpubrawtx=tcp://127.0.0.1:28332
   zmqpubrawblock=tcp://127.0.0.1:28333
   zmqpubhashtx=tcp://127.0.0.1:28334
   zmqpubhashblock=tcp://127.0.0.1:28335

3. Install pyzmq: pip install pyzmq

Reference:
- https://github.com/bitcoin/bitcoin/blob/master/doc/zmq.md
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Deque
from collections import deque
from hashlib import sha256

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('bitcoin_zmq')

# Try to import zmq
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("[WARNING] pyzmq not installed - run: pip install pyzmq")


@dataclass
class MempoolTx:
    """Mempool transaction from Bitcoin Core."""
    txid: str
    raw_size: int
    timestamp: float
    fee_estimate: int = 0  # Estimated from raw tx if possible
    value_estimate: float = 0.0


@dataclass
class NewBlock:
    """New block notification."""
    block_hash: str
    height: int  # Will be 0 if unknown
    timestamp: float


@dataclass
class MempoolStats:
    """Mempool statistics."""
    tx_count: int
    tx_rate: float  # txs per second
    large_tx_count: int  # txs > 10KB
    whale_activity: float  # 0-1 indicator
    timestamp: float


class BitcoinZMQ:
    """
    Bitcoin Core ZMQ mempool integration.

    Provides real-time access to:
    - All transactions entering mempool (rawtx)
    - All new blocks (rawblock)
    - Transaction and block hashes (for lightweight monitoring)

    Usage:
        zmq_feed = BitcoinZMQ(endpoint="tcp://127.0.0.1:28332")

        async def on_tx(tx: MempoolTx):
            print(f"New TX: {tx.txid[:16]}... size={tx.raw_size}")

        zmq_feed.on_tx = on_tx
        await zmq_feed.start()
    """

    def __init__(
        self,
        rawtx_endpoint: str = "tcp://127.0.0.1:28332",
        rawblock_endpoint: str = "tcp://127.0.0.1:28333",
        hashtx_endpoint: str = "tcp://127.0.0.1:28334",
        hashblock_endpoint: str = "tcp://127.0.0.1:28335",
        buffer_size: int = 100_000,
    ):
        self.rawtx_endpoint = rawtx_endpoint
        self.rawblock_endpoint = rawblock_endpoint
        self.hashtx_endpoint = hashtx_endpoint
        self.hashblock_endpoint = hashblock_endpoint

        # Callbacks
        self.on_tx: Optional[Callable[[MempoolTx], None]] = None
        self.on_block: Optional[Callable[[NewBlock], None]] = None
        self.on_stats: Optional[Callable[[MempoolStats], None]] = None

        # State
        self.running = False
        self.connected = False

        # Data buffers
        self.transactions: Deque[MempoolTx] = deque(maxlen=buffer_size)
        self.blocks: Deque[NewBlock] = deque(maxlen=1000)

        # Statistics
        self._start_time = 0.0
        self._tx_count = 0
        self._block_count = 0
        self._bytes_received = 0
        self._large_tx_count = 0
        self._recent_tx_timestamps: Deque[float] = deque(maxlen=1000)

        # ZMQ context
        self._context = None
        self._sockets = {}

    async def start(self):
        """Start ZMQ feed."""
        if not ZMQ_AVAILABLE:
            raise RuntimeError("pyzmq not installed - run: pip install pyzmq")

        self.running = True
        self._start_time = time.time()

        print("=" * 70)
        print("BITCOIN CORE ZMQ MEMPOOL FEED - STARTING")
        print("=" * 70)
        print(f"Raw TX endpoint:    {self.rawtx_endpoint}")
        print(f"Raw Block endpoint: {self.rawblock_endpoint}")
        print()

        # Create ZMQ context
        self._context = zmq.asyncio.Context()

        # Start all subscribers
        tasks = [
            self._subscribe_rawtx(),
            self._subscribe_rawblock(),
            self._stats_reporter(),
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"ZMQ error: {e}")
        finally:
            self.stop()

    async def _subscribe_rawtx(self):
        """Subscribe to raw transactions."""
        socket = self._context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 100000)  # High water mark
        socket.setsockopt_string(zmq.SUBSCRIBE, 'rawtx')

        try:
            socket.connect(self.rawtx_endpoint)
            self.connected = True
            print(f"[ZMQ] Connected to rawtx at {self.rawtx_endpoint}")

            while self.running:
                try:
                    msg = await asyncio.wait_for(
                        socket.recv_multipart(),
                        timeout=30
                    )

                    if len(msg) >= 2:
                        topic = msg[0].decode('utf-8')
                        raw_tx = msg[1]

                        if topic == 'rawtx':
                            await self._process_rawtx(raw_tx)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"[ZMQ] rawtx error: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"[ZMQ] rawtx connection error: {e}")
        finally:
            socket.close()

    async def _subscribe_rawblock(self):
        """Subscribe to raw blocks."""
        socket = self._context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 1000)
        socket.setsockopt_string(zmq.SUBSCRIBE, 'rawblock')

        try:
            socket.connect(self.rawblock_endpoint)
            print(f"[ZMQ] Connected to rawblock at {self.rawblock_endpoint}")

            while self.running:
                try:
                    msg = await asyncio.wait_for(
                        socket.recv_multipart(),
                        timeout=60
                    )

                    if len(msg) >= 2:
                        topic = msg[0].decode('utf-8')
                        raw_block = msg[1]

                        if topic == 'rawblock':
                            await self._process_rawblock(raw_block)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"[ZMQ] rawblock error: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"[ZMQ] rawblock connection error: {e}")
        finally:
            socket.close()

    async def _process_rawtx(self, raw_tx: bytes):
        """Process raw transaction."""
        now = time.time()
        self._bytes_received += len(raw_tx)

        # Calculate txid (double SHA256, reversed)
        txid = sha256(sha256(raw_tx).digest()).digest()[::-1].hex()

        # Create transaction record
        tx = MempoolTx(
            txid=txid,
            raw_size=len(raw_tx),
            timestamp=now,
        )

        # Track large transactions
        if tx.raw_size > 10000:  # > 10KB
            self._large_tx_count += 1

        # Store
        self.transactions.append(tx)
        self._tx_count += 1
        self._recent_tx_timestamps.append(now)

        # Callback
        if self.on_tx:
            self.on_tx(tx)

    async def _process_rawblock(self, raw_block: bytes):
        """Process raw block."""
        now = time.time()
        self._bytes_received += len(raw_block)

        # Calculate block hash (double SHA256 of header, reversed)
        # Header is first 80 bytes
        header = raw_block[:80]
        block_hash = sha256(sha256(header).digest()).digest()[::-1].hex()

        block = NewBlock(
            block_hash=block_hash,
            height=0,  # Would need to parse or query for height
            timestamp=now,
        )

        self.blocks.append(block)
        self._block_count += 1

        print(f"[ZMQ] NEW BLOCK: {block_hash[:16]}...")

        if self.on_block:
            self.on_block(block)

    async def _stats_reporter(self):
        """Periodic stats reporting."""
        while self.running:
            await asyncio.sleep(30)

            if self._tx_count > 0:
                stats = self.get_stats()

                print(f"[ZMQ] TX: {stats['tx_count']:,} ({stats['tx_rate']:.2f}/sec) | "
                      f"Blocks: {stats['block_count']} | "
                      f"Data: {stats['mb_received']:.2f} MB")

                if self.on_stats:
                    mempool_stats = MempoolStats(
                        tx_count=stats['tx_count'],
                        tx_rate=stats['tx_rate'],
                        large_tx_count=stats['large_tx_count'],
                        whale_activity=self._calculate_whale_activity(),
                        timestamp=time.time(),
                    )
                    self.on_stats(mempool_stats)

    def _calculate_whale_activity(self) -> float:
        """
        Calculate whale activity indicator (0-1).

        Based on:
        - Large transaction frequency
        - Transaction size distribution
        """
        if self._tx_count == 0:
            return 0.0

        # Ratio of large transactions
        large_ratio = self._large_tx_count / self._tx_count

        # Scale to 0-1 (typically <1% are large)
        whale_activity = min(1.0, large_ratio * 100)

        return whale_activity

    def stop(self):
        """Stop ZMQ feed."""
        self.running = False
        self.connected = False

        if self._context:
            self._context.term()

    def get_stats(self) -> dict:
        """Get feed statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 1

        # Calculate recent tx rate
        now = time.time()
        recent = [t for t in self._recent_tx_timestamps if now - t < 60]
        tx_rate = len(recent) / 60 if recent else self._tx_count / elapsed

        return {
            'connected': self.connected,
            'elapsed_sec': elapsed,
            'tx_count': self._tx_count,
            'tx_rate': tx_rate,
            'block_count': self._block_count,
            'bytes_received': self._bytes_received,
            'mb_received': self._bytes_received / 1e6,
            'large_tx_count': self._large_tx_count,
            'whale_activity': self._calculate_whale_activity(),
        }

    def get_recent_transactions(self, count: int = 100) -> list:
        """Get most recent transactions."""
        return list(self.transactions)[-count:]

    def get_large_transactions(self, min_size: int = 10000) -> list:
        """Get large transactions (potential whale activity)."""
        return [tx for tx in self.transactions if tx.raw_size >= min_size]


async def test_zmq_feed(duration: int = 60):
    """Test Bitcoin Core ZMQ feed."""
    print("=" * 70)
    print("BITCOIN CORE ZMQ TEST")
    print("=" * 70)
    print()
    print("NOTE: This requires Bitcoin Core running with ZMQ enabled.")
    print("Add to bitcoin.conf:")
    print("  zmqpubrawtx=tcp://127.0.0.1:28332")
    print("  zmqpubrawblock=tcp://127.0.0.1:28333")
    print()

    feed = BitcoinZMQ()

    tx_count = 0

    def on_tx(tx: MempoolTx):
        nonlocal tx_count
        tx_count += 1

        if tx_count % 10 == 0:
            print(f"[TX #{tx_count}] {tx.txid[:16]}... | {tx.raw_size} bytes")

    def on_block(block: NewBlock):
        print(f"\n*** NEW BLOCK: {block.block_hash[:32]}... ***\n")

    feed.on_tx = on_tx
    feed.on_block = on_block

    async def monitor():
        await asyncio.sleep(5)
        start = time.time()

        while time.time() - start < duration:
            await asyncio.sleep(10)
            stats = feed.get_stats()

            if stats['connected']:
                print(f"\n--- Stats after {int(time.time() - start)}s ---")
                print(f"Transactions: {stats['tx_count']:,}")
                print(f"TX Rate: {stats['tx_rate']:.2f}/sec")
                print(f"Blocks: {stats['block_count']}")
                print(f"Data: {stats['mb_received']:.2f} MB")
                print(f"Large TXs: {stats['large_tx_count']}")
                print(f"Whale Activity: {stats['whale_activity']:.2%}")
            else:
                print("[ZMQ] Not connected - check Bitcoin Core configuration")

        feed.stop()

    try:
        await asyncio.gather(
            feed.start(),
            monitor(),
            return_exceptions=True
        )
    except KeyboardInterrupt:
        feed.stop()

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    stats = feed.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    asyncio.run(test_zmq_feed(duration))
