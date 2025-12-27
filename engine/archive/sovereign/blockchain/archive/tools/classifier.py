"""
Address Classifier - Pattern-based classification.
Identify exchanges, whales, users by behavior.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AddressType(Enum):
    """Address classification types."""
    EXCHANGE_HOT = "exchange_hot"       # High-frequency exchange wallet
    EXCHANGE_COLD = "exchange_cold"     # Large, infrequent exchange storage
    EXCHANGE_DEPOSIT = "exchange_deposit"  # User deposit address
    WHALE = "whale"                     # Large holder, not exchange
    MINING_POOL = "mining_pool"         # Mining pool payout address
    SERVICE = "service"                 # Payment processor, etc
    USER = "user"                       # Regular user


@dataclass
class ClassificationResult:
    """Result of address classification."""
    address: str
    classification: AddressType
    confidence: float  # 0.0 - 1.0
    reasons: List[str]
    exchange_id: Optional[str] = None


class AddressClassifier:
    """
    Classify addresses by behavior patterns.

    PATTERNS:
    1. EXCHANGE_HOT: High tx frequency, 24/7 activity, consolidation participant
    2. EXCHANGE_COLD: Large balance, rare movement, receives from hot wallets
    3. EXCHANGE_DEPOSIT: 1-5 receives, sends to consolidation, short lifespan
    4. WHALE: Large balance, low frequency, not consolidation related
    5. MINING_POOL: Receives coinbase, many outputs
    6. USER: Everything else
    """

    # Thresholds for classification
    THRESHOLDS = {
        "exchange_hot": {
            "tx_per_day": 50,
            "active_hours": 18,
            "consolidation_min": 1
        },
        "exchange_deposit": {
            "receive_count_max": 5,
            "send_count_max": 2,
            "lifespan_blocks_max": 4320  # ~30 days
        },
        "whale": {
            "balance_btc_min": 1000,
            "tx_per_day_max": 5
        },
        "mining_pool": {
            "coinbase_receives_min": 1,
            "output_count_avg_min": 50
        }
    }

    def __init__(self, seed_addresses: Dict[str, List[str]] = None):
        """
        Initialize classifier.

        Args:
            seed_addresses: Known exchange addresses {exchange_id: [addresses]}
        """
        self.seed_addresses = seed_addresses or {}
        self.known_exchange_addrs: Dict[str, str] = {}  # addr -> exchange_id

        for ex_id, addrs in self.seed_addresses.items():
            for addr in addrs:
                self.known_exchange_addrs[addr] = ex_id

    def classify(self, profile: Dict) -> ClassificationResult:
        """
        Classify an address based on its profile.

        Args:
            profile: Address profile dict with metrics

        Returns:
            ClassificationResult
        """
        addr = profile.get("address", "")
        reasons = []

        # Check if known address
        if addr in self.known_exchange_addrs:
            return ClassificationResult(
                address=addr,
                classification=AddressType.EXCHANGE_HOT,
                confidence=1.0,
                reasons=["Known exchange address"],
                exchange_id=self.known_exchange_addrs[addr]
            )

        # Extract metrics
        tx_count = profile.get("tx_count", 0)
        receive_count = profile.get("receive_count", 0)
        send_count = profile.get("send_count", 0)
        consolidation_count = profile.get("consolidation_count", 0)
        is_hot_wallet = profile.get("is_hot_wallet", False)
        active_hours = profile.get("active_hours", 0)
        total_received = profile.get("total_received", 0)
        total_sent = profile.get("total_sent", 0)
        first_seen = profile.get("first_seen", 0)
        last_seen = profile.get("last_seen", 0)
        lifespan = last_seen - first_seen

        # Calculate derived metrics
        balance = total_received - total_sent

        # PATTERN 1: Exchange Hot Wallet
        if self._is_exchange_hot(tx_count, active_hours, consolidation_count, is_hot_wallet):
            reasons.append(f"High tx count: {tx_count}")
            if active_hours >= 18:
                reasons.append(f"24/7 activity: {active_hours} hours")
            if consolidation_count > 0:
                reasons.append(f"Consolidation participant: {consolidation_count}x")
            if is_hot_wallet:
                reasons.append("Consolidation output (hot wallet)")

            return ClassificationResult(
                address=addr,
                classification=AddressType.EXCHANGE_HOT,
                confidence=min(0.95, 0.6 + consolidation_count * 0.1 + (active_hours / 24) * 0.2),
                reasons=reasons
            )

        # PATTERN 2: Exchange Deposit Address
        if self._is_exchange_deposit(receive_count, send_count, lifespan, consolidation_count):
            reasons.append(f"Few receives: {receive_count}")
            reasons.append(f"Sends to consolidation: {send_count}")
            reasons.append(f"Short lifespan: {lifespan} blocks")

            return ClassificationResult(
                address=addr,
                classification=AddressType.EXCHANGE_DEPOSIT,
                confidence=0.85 if consolidation_count > 0 else 0.6,
                reasons=reasons
            )

        # PATTERN 3: Exchange Cold Storage
        if self._is_exchange_cold(balance, tx_count, is_hot_wallet):
            reasons.append(f"Large balance: {balance:.2f} BTC")
            reasons.append(f"Low activity: {tx_count} txs")

            return ClassificationResult(
                address=addr,
                classification=AddressType.EXCHANGE_COLD,
                confidence=0.7,
                reasons=reasons
            )

        # PATTERN 4: Whale
        if self._is_whale(balance, tx_count, consolidation_count):
            reasons.append(f"Large balance: {balance:.2f} BTC")
            reasons.append(f"Not exchange pattern")

            return ClassificationResult(
                address=addr,
                classification=AddressType.WHALE,
                confidence=0.75,
                reasons=reasons
            )

        # Default: User
        return ClassificationResult(
            address=addr,
            classification=AddressType.USER,
            confidence=0.5,
            reasons=["No special pattern detected"]
        )

    def _is_exchange_hot(self, tx_count: int, active_hours: int,
                         consolidation_count: int, is_hot_wallet: bool) -> bool:
        """Check if address is exchange hot wallet."""
        t = self.THRESHOLDS["exchange_hot"]

        # Direct hot wallet indicator
        if is_hot_wallet:
            return True

        # High activity + consolidation = definite exchange
        if consolidation_count >= t["consolidation_min"]:
            if tx_count >= t["tx_per_day"] or active_hours >= t["active_hours"]:
                return True

        # Very high activity alone
        if tx_count >= t["tx_per_day"] * 2 and active_hours >= t["active_hours"]:
            return True

        return False

    def _is_exchange_deposit(self, receive_count: int, send_count: int,
                             lifespan: int, consolidation_count: int) -> bool:
        """Check if address is exchange deposit address."""
        t = self.THRESHOLDS["exchange_deposit"]

        # Must have sent to consolidation at some point
        if consolidation_count == 0 and send_count > 0:
            # Could still be deposit if pattern matches
            pass

        return (
            receive_count <= t["receive_count_max"] and
            send_count <= t["send_count_max"] and
            lifespan <= t["lifespan_blocks_max"] and
            receive_count > 0  # Must have received something
        )

    def _is_exchange_cold(self, balance: float, tx_count: int, is_hot_wallet: bool) -> bool:
        """Check if address is exchange cold storage."""
        # Cold storage: large balance, very few transactions
        return (
            balance >= 1000 and
            tx_count <= 20 and
            not is_hot_wallet
        )

    def _is_whale(self, balance: float, tx_count: int, consolidation_count: int) -> bool:
        """Check if address is whale."""
        t = self.THRESHOLDS["whale"]
        return (
            balance >= t["balance_btc_min"] and
            consolidation_count == 0  # Not exchange-related
        )

    def classify_batch(self, profiles: List[Dict]) -> List[ClassificationResult]:
        """Classify multiple addresses."""
        return [self.classify(p) for p in profiles]

    def get_exchanges(self, results: List[ClassificationResult]) -> Dict[str, List[str]]:
        """Extract exchange addresses from classification results."""
        exchanges = {
            "unknown": []  # Exchanges we detected but can't identify
        }

        for r in results:
            if r.classification in (AddressType.EXCHANGE_HOT,
                                    AddressType.EXCHANGE_DEPOSIT,
                                    AddressType.EXCHANGE_COLD):
                ex_id = r.exchange_id or "unknown"
                if ex_id not in exchanges:
                    exchanges[ex_id] = []
                exchanges[ex_id].append(r.address)

        return exchanges

    def get_whales(self, results: List[ClassificationResult]) -> List[str]:
        """Extract whale addresses."""
        return [r.address for r in results if r.classification == AddressType.WHALE]


class EntityLabeler:
    """
    Label entity clusters with exchange IDs.

    If ANY address in an entity is a known exchange address,
    the ENTIRE entity is that exchange.
    """

    def __init__(self, seed_addresses: Dict[str, List[str]]):
        self.seed_addresses = seed_addresses
        self.addr_to_exchange: Dict[str, str] = {}

        for ex_id, addrs in seed_addresses.items():
            for addr in addrs:
                self.addr_to_exchange[addr] = ex_id

    def label_entity(self, entity_addresses: List[str]) -> Optional[str]:
        """
        Label an entity cluster.

        Args:
            entity_addresses: All addresses in the entity

        Returns:
            exchange_id if any address is known, else None
        """
        for addr in entity_addresses:
            if addr in self.addr_to_exchange:
                return self.addr_to_exchange[addr]
        return None

    def propagate_labels(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Label all entities and propagate to all addresses.

        Args:
            entities: {entity_id: [addresses]}

        Returns:
            {address: exchange_id} for all labeled addresses
        """
        labeled = {}

        for entity_id, addresses in entities.items():
            exchange_id = self.label_entity(addresses)
            if exchange_id:
                for addr in addresses:
                    labeled[addr] = exchange_id

        return labeled
