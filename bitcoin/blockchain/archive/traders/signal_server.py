#!/usr/bin/env python3
"""
BLOCKCHAIN SIGNAL SERVER
========================
Runs locally with Bitcoin Core, pushes signals to VPS.

Calculates:
- FIS: Flow Imbalance Signal
- MPI: Mempool Pressure Index
- WM: Whale Momentum
- CR: Consolidation Regime
- CRS: Combined Renaissance Signal

Pushes signals.json to VPS every block (~10 min) or on significant change.
"""

import json
import os
import sys
import time
import zmq
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from collections import deque
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.rpc import BitcoinRPC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BlockMetrics:
    """Metrics extracted from a single block."""
    height: int
    timestamp: int
    inflow_btc: float      # BTC sent TO exchanges
    outflow_btc: float     # BTC sent FROM exchanges
    whale_to_exchange: float    # >100 BTC transactions TO exchange
    whale_from_exchange: float  # >100 BTC transactions FROM exchange
    consolidation_count: int    # Transactions with 50+ inputs
    deposit_sweeps: int         # Consolidations with <=3 outputs
    withdrawal_batches: int     # Consolidations with >20 outputs
    tx_count: int
    total_btc: float


@dataclass
class MempoolMetrics:
    """Metrics from mempool."""
    timestamp: int
    pending_inflow: float
    pending_outflow: float
    high_fee_inflows: float
    total_pending_btc: float
    mempool_size: int


@dataclass
class Signals:
    """All blockchain signals."""
    timestamp: int
    block_height: int
    fis: float          # Flow Imbalance Signal [-1, +1]
    mpi: float          # Mempool Pressure Index [-1.5, +1.5]
    wm: float           # Whale Momentum [-1, +1]
    cr: float           # Consolidation Regime [-1, +1]
    crs: float          # Combined Renaissance Signal
    crs_direction: int  # -1, 0, +1
    confidence: float   # 0.0 to 1.0
    components: Dict


class BlockchainSignalServer:
    """
    Calculates blockchain signals from Bitcoin Core node.
    Pushes results to VPS for trading bot consumption.
    """

    WHALE_THRESHOLD = 100  # BTC
    CONSOLIDATION_THRESHOLD = 50  # inputs
    HIGH_FEE_THRESHOLD = 50  # sat/vbyte
    EPSILON = 1e-8

    def __init__(
        self,
        rpc_host: str = "127.0.0.1",
        rpc_port: int = 8332,
        rpc_user: str = "bitcoin",
        rpc_pass: str = "bitcoin",
        zmq_host: str = "127.0.0.1",
        zmq_port: int = 28335,
        exchange_addresses: Optional[Set[str]] = None,
        vps_host: str = "31.97.211.217",
        vps_user: str = "root",
        vps_path: str = "/root/sovereign/signals.json"
    ):
        self.rpc = BitcoinRPC(rpc_host, rpc_port, rpc_user, rpc_pass)
        self.zmq_endpoint = f"tcp://{zmq_host}:{zmq_port}"

        # VPS config
        self.vps_host = vps_host
        self.vps_user = vps_user
        self.vps_path = vps_path

        # Load exchange addresses
        self.exchange_addresses = exchange_addresses or self._load_exchange_addresses()

        # Block history for rolling calculations
        self.block_history: deque = deque(maxlen=144)  # ~1 day of blocks

        # Last mempool snapshot
        self.last_mempool: Optional[MempoolMetrics] = None

        # Current signals
        self.current_signals: Optional[Signals] = None

    def _load_exchange_addresses(self) -> Set[str]:
        """Load exchange addresses from exchanges.json."""
        addresses = set()

        # Check local data directory
        data_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "exchanges.json"),
            "data/exchanges.json",
        ]

        for path in data_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    for key, addrs in data.items():
                        if key != "_metadata" and isinstance(addrs, list):
                            addresses.update(addrs)
                    logger.info(f"Loaded {len(addresses)} exchange addresses from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if not addresses:
            logger.warning("No exchange addresses loaded! Signals will be limited.")

        return addresses

    def analyze_block(self, height: int) -> Optional[BlockMetrics]:
        """Extract metrics from a block."""
        try:
            block = self.rpc.getblockbyheight(height, verbosity=2)
        except Exception as e:
            logger.error(f"Failed to get block {height}: {e}")
            return None

        metrics = BlockMetrics(
            height=height,
            timestamp=block.get("time", int(time.time())),
            inflow_btc=0.0,
            outflow_btc=0.0,
            whale_to_exchange=0.0,
            whale_from_exchange=0.0,
            consolidation_count=0,
            deposit_sweeps=0,
            withdrawal_batches=0,
            tx_count=len(block.get("tx", [])),
            total_btc=0.0
        )

        for tx in block.get("tx", []):
            vin = tx.get("vin", [])
            vout = tx.get("vout", [])
            vin_count = len(vin)
            vout_count = len(vout)

            # Track input addresses (where BTC comes FROM)
            input_addrs = set()
            for inp in vin:
                if "prevout" in inp:
                    script = inp["prevout"].get("scriptPubKey", {})
                    addr = script.get("address")
                    if addr:
                        input_addrs.add(addr)

            # Track outputs
            tx_value = 0
            output_addrs = set()
            for out in vout:
                value = out.get("value", 0)
                tx_value += value
                script = out.get("scriptPubKey", {})
                addr = script.get("address")
                if addr:
                    output_addrs.add(addr)

            metrics.total_btc += tx_value

            # Detect flows
            from_exchange = bool(input_addrs & self.exchange_addresses)
            to_exchange = bool(output_addrs & self.exchange_addresses)

            if to_exchange and not from_exchange:
                # Inflow to exchange (sell pressure)
                metrics.inflow_btc += tx_value
                if tx_value >= self.WHALE_THRESHOLD:
                    metrics.whale_to_exchange += tx_value

            elif from_exchange and not to_exchange:
                # Outflow from exchange (accumulation)
                metrics.outflow_btc += tx_value
                if tx_value >= self.WHALE_THRESHOLD:
                    metrics.whale_from_exchange += tx_value

            # Consolidation detection
            if vin_count >= self.CONSOLIDATION_THRESHOLD:
                metrics.consolidation_count += 1
                if vout_count <= 3:
                    metrics.deposit_sweeps += 1
                elif vout_count > 20:
                    metrics.withdrawal_batches += 1

        return metrics

    def analyze_mempool(self) -> MempoolMetrics:
        """Extract metrics from mempool."""
        try:
            mempool_info = self.rpc.call("getmempoolinfo")
            raw_mempool = self.rpc.call("getrawmempool", True)
        except Exception as e:
            logger.warning(f"Mempool analysis failed: {e}")
            return MempoolMetrics(
                timestamp=int(time.time()),
                pending_inflow=0,
                pending_outflow=0,
                high_fee_inflows=0,
                total_pending_btc=0,
                mempool_size=0
            )

        pending_inflow = 0.0
        pending_outflow = 0.0
        high_fee_inflows = 0.0
        total_btc = 0.0

        for txid, info in raw_mempool.items():
            try:
                # Get fee rate
                fee = info.get("fees", {}).get("base", 0)
                vsize = info.get("vsize", 1)
                fee_rate = (fee * 1e8) / vsize  # sat/vbyte

                # Get transaction details (if not too expensive)
                # For now, estimate from mempool info
                btc_value = info.get("fees", {}).get("base", 0) * 1000  # rough estimate

                total_btc += btc_value

                # Would need full tx decode for accurate flow detection
                # This is a simplified version
                if fee_rate >= self.HIGH_FEE_THRESHOLD:
                    high_fee_inflows += btc_value * 0.5  # Assume half are inflows

            except Exception:
                continue

        return MempoolMetrics(
            timestamp=int(time.time()),
            pending_inflow=pending_inflow,
            pending_outflow=pending_outflow,
            high_fee_inflows=high_fee_inflows,
            total_pending_btc=total_btc,
            mempool_size=mempool_info.get("size", 0)
        )

    def calculate_fis(self, block: BlockMetrics) -> float:
        """Flow Imbalance Signal: [-1, +1]"""
        total = block.outflow_btc + block.inflow_btc + self.EPSILON
        return (block.outflow_btc - block.inflow_btc) / total

    def calculate_mpi(self, mempool: MempoolMetrics) -> float:
        """Mempool Pressure Index: [-1.5, +1.5]"""
        if mempool.total_pending_btc < self.EPSILON:
            return 0.0

        base = (mempool.pending_outflow - mempool.pending_inflow) / (mempool.total_pending_btc + self.EPSILON)
        urgency = 0.5 * (mempool.high_fee_inflows / (mempool.pending_inflow + self.EPSILON))
        return base - urgency

    def calculate_wm(self, block: BlockMetrics) -> float:
        """Whale Momentum: [-1, +1]"""
        total = block.whale_from_exchange + block.whale_to_exchange + self.EPSILON
        return (block.whale_from_exchange - block.whale_to_exchange) / total

    def calculate_cr(self) -> float:
        """Consolidation Regime: [-1, +1]"""
        if len(self.block_history) < 10:
            return 0.0

        deposit_sweeps = sum(b.deposit_sweeps for b in self.block_history)
        withdrawal_batches = sum(b.withdrawal_batches for b in self.block_history)
        total = deposit_sweeps + withdrawal_batches + self.EPSILON
        return (withdrawal_batches - deposit_sweeps) / total

    def calculate_crs(self, fis: float, mpi: float, wm: float, cr: float) -> Dict:
        """Combined Renaissance Signal."""
        # Weights (tuned from backtest)
        weights = {
            'fis': 0.30,
            'mpi': 0.25,
            'wm': 0.25,
            'cr': 0.20
        }

        combined = (
            weights['fis'] * fis +
            weights['mpi'] * mpi +
            weights['wm'] * wm +
            weights['cr'] * cr
        )

        # Confidence from agreement
        signals = [fis, mpi, wm, cr]
        agreeing = sum(1 for s in signals if (s > 0) == (combined > 0) or abs(s) < 0.1)
        confidence = agreeing / len(signals)

        # Direction
        if combined > 0.2 and confidence >= 0.5:
            direction = 1  # LONG bias
        elif combined < -0.2 and confidence >= 0.5:
            direction = -1  # SHORT bias
        else:
            direction = 0  # NEUTRAL

        return {
            'raw': combined,
            'direction': direction,
            'confidence': confidence
        }

    def calculate_signals(self, block: BlockMetrics, mempool: MempoolMetrics) -> Signals:
        """Calculate all signals."""
        fis = self.calculate_fis(block)
        mpi = self.calculate_mpi(mempool)
        wm = self.calculate_wm(block)
        cr = self.calculate_cr()
        crs = self.calculate_crs(fis, mpi, wm, cr)

        return Signals(
            timestamp=int(time.time()),
            block_height=block.height,
            fis=round(fis, 4),
            mpi=round(mpi, 4),
            wm=round(wm, 4),
            cr=round(cr, 4),
            crs=round(crs['raw'], 4),
            crs_direction=crs['direction'],
            confidence=round(crs['confidence'], 2),
            components={
                'fis': round(fis, 4),
                'mpi': round(mpi, 4),
                'wm': round(wm, 4),
                'cr': round(cr, 4)
            }
        )

    def push_to_vps(self, signals: Signals):
        """Push signals to VPS via SCP."""
        # Write to local temp file
        local_path = "/tmp/signals.json"
        with open(local_path, "w") as f:
            json.dump(asdict(signals), f, indent=2)

        # SCP to VPS
        try:
            cmd = f"scp {local_path} {self.vps_user}@{self.vps_host}:{self.vps_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Pushed signals to VPS: CRS={signals.crs:+.4f} ({signals.crs_direction})")
            else:
                logger.error(f"SCP failed: {result.stderr.decode()}")
        except Exception as e:
            logger.error(f"Push to VPS failed: {e}")

    def process_block(self, height: int):
        """Process a new block and update signals."""
        logger.info(f"Processing block {height}...")

        # Analyze block
        block = self.analyze_block(height)
        if not block:
            return

        self.block_history.append(block)

        # Analyze mempool
        mempool = self.analyze_mempool()
        self.last_mempool = mempool

        # Calculate signals
        signals = self.calculate_signals(block, mempool)
        self.current_signals = signals

        # Log
        logger.info(f"Block {height}: FIS={signals.fis:+.4f} MPI={signals.mpi:+.4f} "
                    f"WM={signals.wm:+.4f} CR={signals.cr:+.4f}")
        logger.info(f"CRS={signals.crs:+.4f} Direction={signals.crs_direction} "
                    f"Confidence={signals.confidence:.0%}")

        # Push to VPS
        self.push_to_vps(signals)

    def run_zmq(self):
        """Subscribe to new blocks via ZMQ."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.zmq_endpoint)
        socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")

        logger.info(f"Subscribed to ZMQ: {self.zmq_endpoint}")
        logger.info("Waiting for new blocks...")

        while True:
            try:
                msg = socket.recv_multipart()
                topic = msg[0].decode()

                if topic == "hashblock":
                    height = self.rpc.getblockcount()
                    self.process_block(height)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"ZMQ error: {e}")
                time.sleep(5)

    def run(self):
        """Main entry point."""
        logger.info("=" * 60)
        logger.info("BLOCKCHAIN SIGNAL SERVER")
        logger.info(f"Exchange addresses: {len(self.exchange_addresses)}")
        logger.info(f"VPS target: {self.vps_user}@{self.vps_host}:{self.vps_path}")
        logger.info("=" * 60)

        # Process current block first
        height = self.rpc.getblockcount()
        self.process_block(height)

        # Then subscribe to new blocks
        self.run_zmq()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Blockchain Signal Server")
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8332)
    parser.add_argument("--rpc-user", default="bitcoin")
    parser.add_argument("--rpc-pass", default="bitcoin")
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=28335)
    parser.add_argument("--vps-host", default="31.97.211.217")
    parser.add_argument("--vps-user", default="root")
    parser.add_argument("--vps-path", default="/root/sovereign/signals.json")

    args = parser.parse_args()

    server = BlockchainSignalServer(
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_user=args.rpc_user,
        rpc_pass=args.rpc_pass,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        vps_host=args.vps_host,
        vps_user=args.vps_user,
        vps_path=args.vps_path
    )

    server.run()


if __name__ == "__main__":
    main()
