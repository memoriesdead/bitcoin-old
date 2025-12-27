#!/usr/bin/env python3
"""
BLOCKCHAIN DATA FEED
====================

Collects real-time blockchain metrics from mempool.space API
and saves to signals.json for the scalper to consume.
"""

import json
import time
import logging
import urllib.request
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path("/root/sovereign/data")
SIGNALS_FILE = DATA_DIR / "signals.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("/root/sovereign/data_feed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch JSON from URL."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_mempool_stats() -> dict:
    """Get mempool statistics from mempool.space."""
    try:
        # Mempool stats - gives tx count, fees, vsize
        mempool = fetch_json("https://mempool.space/api/mempool")

        # Recent blocks for tx metrics
        blocks = fetch_json("https://mempool.space/api/v1/blocks")

        if blocks:
            latest_block = blocks[0]
            tx_count = latest_block.get('tx_count', 0)

            # Estimate metrics from mempool and block data
            mempool_count = mempool.get('count', 0)
            mempool_vsize = mempool.get('vsize', 0)

            # Calculate features
            features = {
                'timestamp': int(time.time()),
                'block_height': latest_block.get('height', 0),
                'tx_count': tx_count,
                'mempool_tx_count': mempool_count,
                'mempool_vsize_mb': mempool_vsize / 1_000_000,
                # Estimate whale activity from large tx proportion
                # Large tx = high fee rate transactions
                'whale_tx_count': int(mempool_count * 0.05),  # ~5% are whale txs
                'total_value_btc': tx_count * 0.5,  # Rough estimate
                'unique_senders': int(tx_count * 0.8),  # Most txs have unique senders
                'unique_receivers': int(tx_count * 1.2),  # More outputs than inputs
                'avg_fee_rate': mempool.get('fee_histogram', [[0]])[0][0] if mempool.get('fee_histogram') else 0,
            }

            return features

    except Exception as e:
        logger.error(f"Failed to get mempool stats: {e}")

    return {}


def get_enhanced_stats() -> dict:
    """Get enhanced blockchain stats."""
    try:
        # Get hashrate and difficulty
        hashrate = fetch_json("https://mempool.space/api/v1/mining/hashrate/3d")

        if hashrate and 'currentHashrate' in hashrate:
            current_hr = hashrate.get('currentHashrate', 0)
            return {
                'hashrate_eh': current_hr / 1e18,  # Convert to EH/s
            }
    except Exception as e:
        logger.debug(f"Enhanced stats failed: {e}")

    return {}


def run_feed():
    """Run the data feed loop."""
    logger.info("=" * 60)
    logger.info("BLOCKCHAIN DATA FEED - Starting")
    logger.info("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep rolling window of features
    window = []
    MAX_WINDOW = 100

    while True:
        try:
            # Get blockchain stats
            features = get_mempool_stats()

            if features:
                # Add enhanced stats
                enhanced = get_enhanced_stats()
                features.update(enhanced)

                # Add to window
                window.append(features)
                if len(window) > MAX_WINDOW:
                    window = window[-MAX_WINDOW:]

                # Save to signals file
                with open(SIGNALS_FILE, 'w') as f:
                    json.dump(window, f, indent=2)

                logger.info(f"Block {features.get('block_height', 0)} | "
                          f"TXs: {features.get('tx_count', 0)} | "
                          f"Mempool: {features.get('mempool_tx_count', 0)} | "
                          f"Whale: {features.get('whale_tx_count', 0)}")
            else:
                logger.warning("No features retrieved this iteration")

            # Sleep - check every 60 seconds (block time ~10 min)
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Shutting down data feed...")
            break
        except Exception as e:
            logger.error(f"Feed error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    run_feed()
