"""
Exchange Wallet Tracker - Known exchange addresses.
INFLOW (TO exchange) = SHORT, OUTFLOW (FROM exchange) = LONG.

RENAISSANCE MODE: Loads from exchanges.json for 1M+ addresses.
O(1) lookup. Seeds are fallback only.
"""
from typing import Dict, Optional, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import os
import json


class ExchangeType(Enum):
    USA_LEGAL = "usa_legal"
    MARKET_MOVER = "market_mover"
    OTHER = "other"


@dataclass
class ExchangeInfo:
    name: str
    exchange_type: ExchangeType
    addresses: List[str]
    notes: str = ""


# Seed addresses - used for initial labeling and fallback
EXCHANGE_SEEDS: Dict[str, ExchangeInfo] = {
    # USA-LEGAL (trading venues)
    "coinbase": ExchangeInfo("Coinbase", ExchangeType.USA_LEGAL, [
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo", "3KZ526NxCVXbKwwP66RgM3pte6zW4gY1tD",
        "3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B", "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
        "3CgKHXR17eh2xCj2RGnHJHTDjPpqaNDgyT",
    ]),
    "kraken": ExchangeInfo("Kraken", ExchangeType.USA_LEGAL, [
        "1AnwDVbwsLBVwRfqN2x9Eo4YEJSPXo2cwG", "14eQD1QQb8QFVG8YFwGz7skyzsvBLWLwJS",
        "1A7znRYE24Z6K8MCAKXLmEvuS5ixzvUrjH", "3AfP3p9UJJfKzxiSLYPNrMpHj5dTSbBph2",
        "3H5JTt42K7RmZtromfTSefcMEFMMe18pMD", "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
    ]),
    "bitstamp": ExchangeInfo("Bitstamp", ExchangeType.USA_LEGAL, [
        "3Nxwenay9Z8Lc9JBiywExpnEFiLp6Afp8v", "3P3n73hhKhoxQAEpCLrFuQpzWXpQp5VXCM",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j", "1PU2rQrtkjvuxk6RZu9XtGbpD7vd5BXMH2",
    ]),
    "gemini": ExchangeInfo("Gemini", ExchangeType.USA_LEGAL, [
        "1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd", "3LQUu4v9z6KNch71j7kbj8GPeAGUo1FW6a",
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
    ]),
    "binance": ExchangeInfo("Binance US", ExchangeType.USA_LEGAL, [
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo", "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
        "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6", "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "bc1ql49ydapnjafl5t2cp9zqpjwe6pdgmxy98859v2",
    ]),
    "hyperliquid": ExchangeInfo("Hyperliquid", ExchangeType.USA_LEGAL, [], "DEX - API only"),

    # MARKET MOVERS
    "bitfinex": ExchangeInfo("Bitfinex", ExchangeType.MARKET_MOVER, [
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g", "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
    ]),
    "okx": ExchangeInfo("OKX", ExchangeType.MARKET_MOVER, [
        "bc1q2s3rjwvam9dt2ftt4sqxqjf3twav0gdx0k0q2etxflx38c3x7jqq8zw398",
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
    ]),
    "bybit": ExchangeInfo("Bybit", ExchangeType.MARKET_MOVER, [
        "bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0eh5tx44reyn2k7sfc27a4",
    ]),
    "huobi": ExchangeInfo("Huobi", ExchangeType.MARKET_MOVER, [
        "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ", "14qk5e9sJFBxgH6vwxJoLqnQLxChLb3rHq",
    ]),
}

# Legacy alias
EXCHANGE_DATABASE = EXCHANGE_SEEDS


class ExchangeWalletTracker:
    """
    Track exchange wallet activity.

    RENAISSANCE MODE: Loads from database for 1M+ addresses.
    Falls back to seeds if database unavailable/empty.
    """

    def __init__(self, json_path: str = None):
        """
        Initialize wallet tracker.

        Args:
            json_path: Path to exchanges.json (default: data/exchanges.json)
        """
        self.exchanges = dict(EXCHANGE_SEEDS)
        self.address_to_exchange: Dict[str, str] = {}
        self.exchange_addresses_set: Set[str] = set()  # Fast lookup set
        self.inflow_count = 0
        self.outflow_count = 0
        self.json_path = json_path
        self.json_loaded = False

        # Try JSON first (Renaissance mode)
        self._load_from_json()

        # Fall back to seeds if JSON empty/unavailable
        if not self.json_loaded or len(self.address_to_exchange) < 100:
            self._load_from_seeds()

        print(f"[WALLET] {len(self.address_to_exchange):,} addresses "
              f"from {len(self.exchanges)} exchanges "
              f"(json={self.json_loaded})")

    def _load_from_json(self):
        """Load addresses from exchanges.json (Renaissance mode)."""
        try:
            # Find JSON file
            if self.json_path is None:
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
                self.json_path = os.path.join(data_dir, "exchanges.json")

            if not os.path.exists(self.json_path):
                print(f"[JSON] File not found: {self.json_path}")
                return

            with open(self.json_path) as f:
                data = json.load(f)

            # Skip metadata
            metadata = data.pop("_metadata", {})

            # Load all exchanges
            for ex_id, addresses in data.items():
                if ex_id not in self.exchanges:
                    self.exchanges[ex_id] = ExchangeInfo(
                        name=ex_id.replace("_", " ").title(),
                        exchange_type=ExchangeType.OTHER,
                        addresses=[]
                    )
                for addr in addresses:
                    self.address_to_exchange[addr] = ex_id
                    self.exchange_addresses_set.add(addr)

            self.json_loaded = True
            print(f"[JSON] Loaded {len(self.address_to_exchange):,} addresses from {self.json_path}")

            if metadata:
                print(f"[JSON] Scan date: {metadata.get('generated', 'unknown')}")
                print(f"[JSON] Consolidations: {metadata.get('consolidations_found', 0):,}")

        except Exception as e:
            print(f"[JSON] Failed to load: {e}")
            self.json_loaded = False

    def _load_from_seeds(self):
        """Load addresses from hardcoded seeds."""
        for ex_id, info in EXCHANGE_SEEDS.items():
            for addr in info.addresses:
                if addr not in self.address_to_exchange:
                    self.address_to_exchange[addr] = ex_id
                    self.exchange_addresses_set.add(addr)

        print(f"[SEED] Loaded {len(self.address_to_exchange)} addresses from seeds")

    def reload(self):
        """Reload addresses from JSON (call after scan completes)."""
        self.address_to_exchange.clear()
        self.exchange_addresses_set.clear()
        self._load_from_json()
        if not self.json_loaded:
            self._load_from_seeds()

    def identify_address(self, address: str) -> Optional[Tuple[str, ExchangeInfo]]:
        """Identify which exchange an address belongs to."""
        if address in self.address_to_exchange:
            ex_id = self.address_to_exchange[address]
            if ex_id in self.exchanges:
                return ex_id, self.exchanges[ex_id]
            # Unknown exchange from database
            return ex_id, ExchangeInfo(ex_id.title(), ExchangeType.OTHER, [])
        return None

    def is_exchange_address(self, address: str) -> bool:
        """Fast check if address is exchange-related."""
        return address in self.exchange_addresses_set

    def is_usa_legal_exchange(self, address: str) -> bool:
        """Check if address belongs to USA-legal exchange."""
        result = self.identify_address(address)
        return result[1].exchange_type == ExchangeType.USA_LEGAL if result else False

    def analyze_transaction(self, inputs: List[str], outputs: List[Dict]) -> Dict:
        """Analyze transaction for exchange flows."""
        result = {
            'is_exchange_related': False,
            'inflows': [],
            'outflows': [],
            'total_inflow_btc': 0.0,
            'total_outflow_btc': 0.0,
            'exchanges_involved': set()
        }

        # Check outputs for INFLOWS (to exchange)
        for out in outputs:
            addr, btc = out.get('address'), out.get('btc', 0)
            if addr and self.is_exchange_address(addr):
                ex_id = self.address_to_exchange.get(addr, 'unknown')
                result['is_exchange_related'] = True
                result['inflows'].append({'exchange': ex_id, 'btc': btc, 'address': addr})
                result['total_inflow_btc'] += btc
                result['exchanges_involved'].add(ex_id)
                self.inflow_count += 1

        # Check inputs for OUTFLOWS (from exchange)
        for inp in inputs:
            addr = inp if isinstance(inp, str) else inp.get('address')
            btc = 0 if isinstance(inp, str) else inp.get('btc', 0)
            if addr and self.is_exchange_address(addr):
                ex_id = self.address_to_exchange.get(addr, 'unknown')
                result['is_exchange_related'] = True
                result['outflows'].append({'exchange': ex_id, 'btc': btc, 'address': addr})
                result['total_outflow_btc'] += btc
                result['exchanges_involved'].add(ex_id)
                self.outflow_count += 1

        result['exchanges_involved'] = list(result['exchanges_involved'])
        return result

    def add_address(self, exchange_id: str, address: str) -> bool:
        """Add an address to tracking."""
        if address in self.address_to_exchange:
            return False

        if exchange_id not in self.exchanges:
            self.exchanges[exchange_id] = ExchangeInfo(
                exchange_id.title(), ExchangeType.OTHER, []
            )

        self.exchanges[exchange_id].addresses.append(address)
        self.address_to_exchange[address] = exchange_id
        self.exchange_addresses_set.add(address)
        return True

    def add_addresses_batch(self, exchange_id: str, addresses: List[str]) -> int:
        """Add multiple addresses at once."""
        added = 0
        for addr in addresses:
            if self.add_address(exchange_id, addr):
                added += 1
        return added

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        stats = {
            'total_addresses': len(self.address_to_exchange),
            'total_exchanges': len(self.exchanges),
            'inflow_count': self.inflow_count,
            'outflow_count': self.outflow_count,
            'json_loaded': self.json_loaded,
        }

        # Breakdown by exchange
        exchange_counts = {}
        for addr, ex_id in self.address_to_exchange.items():
            exchange_counts[ex_id] = exchange_counts.get(ex_id, 0) + 1
        stats['by_exchange'] = exchange_counts

        return stats

    def get_all_addresses(self) -> Set[str]:
        """Get all tracked exchange addresses."""
        return self.exchange_addresses_set.copy()
