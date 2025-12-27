"""
Address Cluster Discovery - Learn exchange addresses from blockchain.
Common-input-ownership: If A and B are INPUTS in same tx, same entity controls both.
"""
import time
import json
import os
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field


@dataclass
class AddressCluster:
    exchange_id: str
    seed_address: str
    discovered_addresses: Set[str] = field(default_factory=set)
    last_updated: float = 0.0
    transaction_count: int = 0


class ExchangeAddressDiscovery:
    """Auto-discover exchange addresses via common-input-ownership heuristic."""

    def __init__(self, seed_addresses: Dict[str, List[str]]):
        self.address_to_exchange: Dict[str, str] = {}
        self.clusters: Dict[str, AddressCluster] = {}
        self.all_addresses: Set[str] = set()
        self.txs_processed = 0
        self.addresses_discovered = 0

        for ex_id, addresses in seed_addresses.items():
            self.clusters[ex_id] = AddressCluster(ex_id, addresses[0] if addresses else "")
            for addr in addresses:
                self.address_to_exchange[addr] = ex_id
                self.all_addresses.add(addr)

        self.seed_count = len(self.all_addresses)
        print(f"[CLUSTER] {self.seed_count} seed addresses")

    def process_transaction(self, txid: str, input_addresses: List[str], output_addresses: List[str]) -> List[str]:
        """Process tx - if any input is known exchange, all inputs belong to that exchange."""
        self.txs_processed += 1
        discovered = []

        known_exchange = None
        for addr in input_addresses:
            if addr and addr in self.address_to_exchange:
                known_exchange = self.address_to_exchange[addr]
                break

        if not known_exchange:
            return []

        cluster = self.clusters[known_exchange]
        for addr in input_addresses:
            if not addr or addr in self.all_addresses:
                continue
            self.address_to_exchange[addr] = known_exchange
            self.all_addresses.add(addr)
            cluster.discovered_addresses.add(addr)
            cluster.last_updated = time.time()
            cluster.transaction_count += 1
            self.addresses_discovered += 1
            discovered.append(addr)

            if self.addresses_discovered <= 10 or self.addresses_discovered % 100 == 0:
                print(f"[CLUSTER] {known_exchange} #{self.addresses_discovered}: {addr[:16]}...")

        return discovered

    def is_exchange_address(self, address: str) -> bool:
        return address in self.all_addresses

    def get_exchange(self, address: str) -> Optional[str]:
        return self.address_to_exchange.get(address)

    def get_stats(self) -> Dict:
        return {
            'seed_addresses': self.seed_count,
            'discovered_addresses': self.addresses_discovered,
            'total_addresses': len(self.all_addresses),
            'txs_processed': self.txs_processed,
        }

    def export_addresses(self) -> Dict[str, List[str]]:
        result = {}
        for ex_id in self.clusters:
            result[ex_id] = [addr for addr, ex in self.address_to_exchange.items() if ex == ex_id]
        return result


class PersistentAddressCluster(ExchangeAddressDiscovery):
    """Address discovery with file persistence."""

    def __init__(self, seed_addresses: Dict[str, List[str]], persist_file: str = "/tmp/exchange_addresses.json"):
        super().__init__(seed_addresses)
        self.persist_file = persist_file
        self._load()

    def _load(self):
        if not os.path.exists(self.persist_file):
            return
        try:
            with open(self.persist_file, 'r') as f:
                data = json.load(f)
            loaded = 0
            for ex_id, addresses in data.items():
                if ex_id not in self.clusters:
                    continue
                for addr in addresses:
                    if addr not in self.all_addresses:
                        self.address_to_exchange[addr] = ex_id
                        self.all_addresses.add(addr)
                        self.clusters[ex_id].discovered_addresses.add(addr)
                        loaded += 1
            if loaded:
                print(f"[CLUSTER] Loaded {loaded} addresses")
        except Exception as e:
            print(f"[CLUSTER] Load failed: {e}")

    def save(self):
        try:
            with open(self.persist_file, 'w') as f:
                json.dump(self.export_addresses(), f, indent=2)
            print(f"[CLUSTER] Saved {len(self.all_addresses)} addresses")
        except Exception as e:
            print(f"[CLUSTER] Save failed: {e}")

    def process_transaction(self, txid: str, input_addresses: List[str], output_addresses: List[str]) -> List[str]:
        discovered = super().process_transaction(txid, input_addresses, output_addresses)
        if self.addresses_discovered > 0 and self.addresses_discovered % 100 == 0:
            self.save()
        return discovered
