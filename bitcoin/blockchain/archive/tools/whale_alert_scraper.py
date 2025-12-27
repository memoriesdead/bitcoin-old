"""
Whale Alert Twitter Scraper - Learn exchange addresses from Whale Alert tweets.

Uses twscrape (free, no API key) to scrape @whale_alert tweets.
Extracts exchange names and learns addresses for future matching.

USAGE:
    python -m engine.sovereign.blockchain.whale_alert_scraper

TWEET FORMAT:
    "1,000 #BTC (95,234,567 USD) transferred from Coinbase to unknown wallet"
    "500 #BTC transferred from unknown wallet to Binance"

We extract:
    - Exchange name (Coinbase, Binance, etc.)
    - Direction (to/from)
    - Amount (for filtering)
"""
import re
import json
import time
import asyncio
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class WhaleTransaction:
    """Parsed Whale Alert transaction."""
    timestamp: datetime
    amount_btc: float
    amount_usd: float
    from_entity: str       # "unknown wallet" or exchange name
    to_entity: str         # "unknown wallet" or exchange name
    direction: int         # +1 = outflow (from exchange), -1 = inflow (to exchange)
    exchange: Optional[str]  # Identified exchange if any
    tweet_text: str
    tweet_id: str


class WhaleAlertScraper:
    """
    Scrape Whale Alert Twitter for exchange flow data.

    MODES:
    1. Live scraping - Get real-time tweets
    2. Historical - Get past tweets for learning

    LEARNING:
    When we see "from Coinbase" or "to Binance", we can correlate
    with recent blockchain txs to learn addresses.
    """

    # Known exchange names (various spellings)
    EXCHANGE_NAMES = {
        'coinbase': ['coinbase', 'coinbase pro'],
        'binance': ['binance', 'binance.us', 'binance us'],
        'kraken': ['kraken'],
        'gemini': ['gemini'],
        'bitstamp': ['bitstamp'],
        'bitfinex': ['bitfinex'],
        'okx': ['okx', 'okex'],
        'huobi': ['huobi', 'htx'],
        'bybit': ['bybit'],
        'kucoin': ['kucoin'],
        'gate.io': ['gate.io', 'gate'],
        'crypto.com': ['crypto.com'],
        'ftx': ['ftx'],  # Historical
    }

    # Flatten for quick lookup
    NAME_TO_EXCHANGE = {}
    for ex_id, names in EXCHANGE_NAMES.items():
        for name in names:
            NAME_TO_EXCHANGE[name.lower()] = ex_id

    # Regex patterns for parsing tweets
    AMOUNT_PATTERN = re.compile(
        r'([\d,]+)\s*#?BTC\s*\(([\d,]+)\s*USD\)',
        re.IGNORECASE
    )
    TRANSFER_PATTERN = re.compile(
        r'transferred\s+from\s+(.+?)\s+to\s+(.+?)(?:\.|$)',
        re.IGNORECASE
    )

    def __init__(self,
                 output_file: str = None,
                 min_btc: float = 100.0,
                 accounts_file: str = None):
        """
        Initialize scraper.

        Args:
            output_file: Path to save learned data (default: data/whale_alerts.json)
            min_btc: Minimum BTC to track (default: 100)
            accounts_file: Path to twscrape accounts file
        """
        if output_file is None:
            data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            output_file = os.path.join(data_dir, "data", "whale_alerts.json")

        self.output_file = output_file
        self.min_btc = min_btc
        self.accounts_file = accounts_file

        # Transaction history
        self.transactions: List[WhaleTransaction] = []

        # Stats
        self.tweets_scraped = 0
        self.exchanges_found = 0
        self.inflows = 0
        self.outflows = 0

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing transaction data."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file) as f:
                    data = json.load(f)
                    self.transactions = [
                        WhaleTransaction(**{**t, 'timestamp': datetime.fromisoformat(t['timestamp'])})
                        for t in data.get('transactions', [])
                    ]
                    print(f"[WHALE] Loaded {len(self.transactions)} historical transactions")
            except Exception as e:
                print(f"[WHALE] Failed to load data: {e}")

    def _save_data(self):
        """Save transaction data."""
        try:
            data = {
                'updated': datetime.now().isoformat(),
                'transactions': [
                    {**t.__dict__, 'timestamp': t.timestamp.isoformat()}
                    for t in self.transactions[-10000:]  # Keep last 10k
                ],
                'stats': {
                    'total_scraped': self.tweets_scraped,
                    'exchanges_found': self.exchanges_found,
                    'inflows': self.inflows,
                    'outflows': self.outflows,
                }
            }
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WHALE] Failed to save: {e}")

    def parse_tweet(self, text: str, tweet_id: str = "",
                    timestamp: datetime = None) -> Optional[WhaleTransaction]:
        """
        Parse a Whale Alert tweet.

        Returns:
            WhaleTransaction if parsed successfully, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Must contain BTC
        if '#btc' not in text.lower() and 'btc' not in text.lower():
            return None

        # Extract amount
        amount_match = self.AMOUNT_PATTERN.search(text)
        if not amount_match:
            return None

        amount_btc = float(amount_match.group(1).replace(',', ''))
        amount_usd = float(amount_match.group(2).replace(',', ''))

        # Skip if below threshold
        if amount_btc < self.min_btc:
            return None

        # Extract from/to
        transfer_match = self.TRANSFER_PATTERN.search(text)
        if not transfer_match:
            return None

        from_entity = transfer_match.group(1).strip().lower()
        to_entity = transfer_match.group(2).strip().lower()

        # Identify exchanges
        from_exchange = self._identify_exchange(from_entity)
        to_exchange = self._identify_exchange(to_entity)

        # Determine direction and exchange
        exchange = None
        direction = 0

        if from_exchange and not to_exchange:
            # FROM exchange TO unknown = OUTFLOW = LONG
            exchange = from_exchange
            direction = +1
            self.outflows += 1
        elif to_exchange and not from_exchange:
            # FROM unknown TO exchange = INFLOW = SHORT
            exchange = to_exchange
            direction = -1
            self.inflows += 1
        elif from_exchange and to_exchange:
            # Exchange to exchange - use destination
            exchange = to_exchange
            direction = -1  # Net inflow to destination
            self.inflows += 1

        if exchange:
            self.exchanges_found += 1

        tx = WhaleTransaction(
            timestamp=timestamp,
            amount_btc=amount_btc,
            amount_usd=amount_usd,
            from_entity=from_entity,
            to_entity=to_entity,
            direction=direction,
            exchange=exchange,
            tweet_text=text,
            tweet_id=tweet_id,
        )

        self.transactions.append(tx)
        return tx

    def _identify_exchange(self, entity: str) -> Optional[str]:
        """Identify if entity is an exchange."""
        entity_lower = entity.lower().strip()

        # Check direct match
        if entity_lower in self.NAME_TO_EXCHANGE:
            return self.NAME_TO_EXCHANGE[entity_lower]

        # Check if exchange name is contained
        for name, ex_id in self.NAME_TO_EXCHANGE.items():
            if name in entity_lower:
                return ex_id

        return None

    async def scrape_live(self, duration_hours: float = 24):
        """
        Scrape Whale Alert tweets in real-time.

        Requires twscrape with authenticated accounts.
        """
        try:
            from twscrape import API, gather
        except ImportError:
            print("[WHALE] twscrape not installed. Run: pip install twscrape")
            return

        print(f"[WHALE] Starting live scrape for {duration_hours} hours...")

        api = API()

        # Add accounts if file provided
        if self.accounts_file and os.path.exists(self.accounts_file):
            await api.pool.load_from_file(self.accounts_file)

        end_time = time.time() + (duration_hours * 3600)
        last_tweet_id = None

        while time.time() < end_time:
            try:
                # Search for whale_alert tweets
                query = "from:whale_alert #BTC"
                tweets = await gather(api.search(query, limit=20))

                for tweet in tweets:
                    if last_tweet_id and int(tweet.id) <= int(last_tweet_id):
                        continue

                    self.tweets_scraped += 1
                    tx = self.parse_tweet(
                        tweet.rawContent,
                        tweet_id=str(tweet.id),
                        timestamp=tweet.date
                    )

                    if tx and tx.direction != 0:
                        dir_str = "LONG" if tx.direction > 0 else "SHORT"
                        print(f"[WHALE] {tx.amount_btc:.0f} BTC | {tx.exchange} | {dir_str}")

                    last_tweet_id = tweet.id

                # Save periodically
                if self.tweets_scraped % 10 == 0:
                    self._save_data()

            except Exception as e:
                print(f"[WHALE] Scrape error: {e}")

            # Wait between requests
            await asyncio.sleep(30)

        self._save_data()
        print(f"[WHALE] Scrape complete. {self.tweets_scraped} tweets, "
              f"{self.exchanges_found} exchanges found")

    def scrape_historical_mock(self, tweets: List[str]):
        """
        Process historical tweets (for testing without API).

        Args:
            tweets: List of tweet texts
        """
        for i, text in enumerate(tweets):
            tx = self.parse_tweet(text, tweet_id=str(i))
            if tx:
                dir_str = "LONG" if tx.direction > 0 else "SHORT"
                print(f"[{tx.exchange or 'unknown'}] {tx.amount_btc:.0f} BTC | {dir_str}")

        self._save_data()

    def get_recent_flows(self, hours: float = 1) -> Dict:
        """Get recent flow summary."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [t for t in self.transactions if t.timestamp >= cutoff]

        inflow_btc = sum(t.amount_btc for t in recent if t.direction < 0)
        outflow_btc = sum(t.amount_btc for t in recent if t.direction > 0)
        net = outflow_btc - inflow_btc

        # Direction based on net flow
        if net > 100:
            direction = +1  # LONG
        elif net < -100:
            direction = -1  # SHORT
        else:
            direction = 0

        return {
            'inflow_btc': inflow_btc,
            'outflow_btc': outflow_btc,
            'net_flow': net,
            'direction': direction,
            'signal': 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'NEUTRAL',
            'count': len(recent),
            'window_hours': hours,
        }

    def get_exchange_flows(self, hours: float = 24) -> Dict[str, Dict]:
        """Get per-exchange flow breakdown."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [t for t in self.transactions if t.timestamp >= cutoff and t.exchange]

        flows = {}
        for tx in recent:
            if tx.exchange not in flows:
                flows[tx.exchange] = {'inflow': 0, 'outflow': 0, 'count': 0}

            flows[tx.exchange]['count'] += 1
            if tx.direction > 0:
                flows[tx.exchange]['outflow'] += tx.amount_btc
            else:
                flows[tx.exchange]['inflow'] += tx.amount_btc

        # Add net flow
        for ex_id in flows:
            flows[ex_id]['net'] = flows[ex_id]['outflow'] - flows[ex_id]['inflow']

        return flows

    def get_stats(self) -> Dict:
        """Get scraper statistics."""
        return {
            'tweets_scraped': self.tweets_scraped,
            'transactions_stored': len(self.transactions),
            'exchanges_found': self.exchanges_found,
            'inflows': self.inflows,
            'outflows': self.outflows,
            'recent_1h': self.get_recent_flows(1),
            'recent_24h': self.get_recent_flows(24),
        }


# Test with sample tweets
if __name__ == '__main__':
    scraper = WhaleAlertScraper(min_btc=100)

    # Sample Whale Alert tweets
    sample_tweets = [
        "1,500 #BTC (142,500,000 USD) transferred from Coinbase to unknown wallet",
        "2,000 #BTC (190,000,000 USD) transferred from unknown wallet to Binance",
        "500 #BTC (47,500,000 USD) transferred from Kraken to unknown wallet",
        "3,000 #BTC (285,000,000 USD) transferred from unknown wallet to Coinbase",
        "750 #BTC (71,250,000 USD) transferred from Gemini to unknown wallet",
        "1,200 #BTC (114,000,000 USD) transferred from unknown wallet to Kraken",
        "800 #BTC (76,000,000 USD) transferred from Bitfinex to Binance",
    ]

    print("=" * 60)
    print("WHALE ALERT SCRAPER TEST")
    print("=" * 60)

    scraper.scrape_historical_mock(sample_tweets)

    print()
    print("STATS:", scraper.get_stats())
    print()
    print("EXCHANGE FLOWS:", scraper.get_exchange_flows(24))
