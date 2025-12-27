"""
Entity Clustering - Group addresses by ownership.
Uses Union-Find for efficient clustering.
"""
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Efficient for clustering addresses by common ownership.
    Operations are nearly O(1) with path compression and union by rank.
    """

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.size: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find root of element with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """
        Unite two elements. Returns True if they were in different sets.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True

    def connected(self, x: str, y: str) -> bool:
        """Check if two elements are in the same set."""
        return self.find(x) == self.find(y)

    def get_size(self, x: str) -> int:
        """Get size of the set containing x."""
        return self.size[self.find(x)]

    def get_all_roots(self) -> Set[str]:
        """Get all unique root elements (entity IDs)."""
        roots = set()
        for x in self.parent:
            roots.add(self.find(x))
        return roots

    def get_members(self, x: str) -> List[str]:
        """Get all members of the set containing x."""
        root = self.find(x)
        return [addr for addr in self.parent if self.find(addr) == root]

    def get_all_clusters(self) -> Dict[str, List[str]]:
        """Get all clusters as {root: [members]}."""
        clusters = defaultdict(list)
        for addr in self.parent:
            root = self.find(addr)
            clusters[root].append(addr)
        return dict(clusters)


class EntityClusterer:
    """
    Cluster addresses into entities.

    An entity = one real-world actor (exchange, whale, user).

    CLUSTERING RULES:
    1. Common-input-ownership: Inputs in same tx = same entity
    2. Change address detection: Change output likely same entity
    3. Temporal clustering: Same timing patterns = likely same entity
    """

    def __init__(self):
        self.uf = UnionFind()
        self.entity_labels: Dict[str, str] = {}  # root -> label
        self.entity_metadata: Dict[str, Dict] = {}  # root -> metadata

    def process_transaction(self, inputs: List[str], outputs: List[str] = None):
        """
        Process a transaction for clustering.

        All inputs belong to the same entity (common-input-ownership).
        """
        if len(inputs) < 2:
            return

        # Unite all inputs
        first = inputs[0]
        for addr in inputs[1:]:
            self.uf.union(first, addr)

    def process_consolidation(self, inputs: List[str], outputs: List[str]):
        """
        Process a consolidation transaction.

        All inputs AND outputs belong to same entity (exchange).
        """
        all_addrs = inputs + outputs
        if len(all_addrs) < 2:
            return

        first = all_addrs[0]
        for addr in all_addrs[1:]:
            self.uf.union(first, addr)

    def add_link(self, addr1: str, addr2: str):
        """Explicitly link two addresses as same entity."""
        self.uf.union(addr1, addr2)

    def add_links_batch(self, links: List[Tuple[str, str]]):
        """Add multiple links efficiently."""
        for addr1, addr2 in links:
            self.uf.union(addr1, addr2)

    def label_entity(self, any_member: str, label: str):
        """Label an entity (by any member address)."""
        root = self.uf.find(any_member)
        self.entity_labels[root] = label

    def get_entity_label(self, addr: str) -> Optional[str]:
        """Get label for address's entity."""
        root = self.uf.find(addr)
        return self.entity_labels.get(root)

    def get_entity_id(self, addr: str) -> str:
        """Get entity ID (root address) for an address."""
        return self.uf.find(addr)

    def get_entity_members(self, addr: str) -> List[str]:
        """Get all addresses in same entity."""
        return self.uf.get_members(addr)

    def get_entity_size(self, addr: str) -> int:
        """Get number of addresses in entity."""
        return self.uf.get_size(addr)

    def get_all_entities(self) -> Dict[str, List[str]]:
        """Get all entities as {entity_id: [addresses]}."""
        return self.uf.get_all_clusters()

    def get_labeled_entities(self) -> Dict[str, Dict]:
        """Get all labeled entities with their addresses."""
        result = {}
        for root, label in self.entity_labels.items():
            result[label] = {
                "entity_id": root,
                "addresses": self.uf.get_members(root),
                "size": self.uf.get_size(root)
            }
        return result

    def propagate_labels_from_seeds(self, seed_addresses: Dict[str, List[str]]) -> int:
        """
        Propagate labels from known seed addresses.

        If ANY address in an entity is a known exchange,
        the entire entity gets that label.

        Args:
            seed_addresses: {exchange_id: [known_addresses]}

        Returns:
            Number of entities labeled
        """
        labeled_count = 0

        for exchange_id, addrs in seed_addresses.items():
            for addr in addrs:
                root = self.uf.find(addr)
                if root not in self.entity_labels:
                    self.entity_labels[root] = exchange_id
                    labeled_count += 1

        return labeled_count

    def get_exchange_addresses(self) -> Dict[str, List[str]]:
        """Get all addresses grouped by exchange."""
        result = defaultdict(list)

        for root, label in self.entity_labels.items():
            members = self.uf.get_members(root)
            result[label].extend(members)

        return dict(result)

    def get_stats(self) -> Dict:
        """Get clustering statistics."""
        clusters = self.uf.get_all_clusters()
        sizes = [len(members) for members in clusters.values()]

        return {
            "total_addresses": sum(sizes),
            "total_entities": len(clusters),
            "labeled_entities": len(self.entity_labels),
            "largest_entity": max(sizes) if sizes else 0,
            "avg_entity_size": sum(sizes) / len(sizes) if sizes else 0,
            "single_address_entities": sum(1 for s in sizes if s == 1)
        }


class ConsolidationTracker:
    """
    Track consolidation transactions to identify exchanges.

    ANY transaction with 50+ inputs is almost certainly an exchange.
    """

    CONSOLIDATION_THRESHOLD = 50

    def __init__(self):
        self.consolidations: List[Dict] = []
        self.exchange_addresses: Set[str] = set()
        self.hot_wallets: Set[str] = set()

    def process_transaction(self, txid: str, inputs: List[str], outputs: List[str], btc: float = 0):
        """
        Check if transaction is a consolidation.

        If 50+ inputs, ALL inputs are exchange deposit addresses,
        and outputs are hot wallets.
        """
        if len(inputs) < self.CONSOLIDATION_THRESHOLD:
            return False

        self.consolidations.append({
            "txid": txid,
            "input_count": len(inputs),
            "output_count": len(outputs),
            "inputs": inputs,
            "outputs": outputs,
            "btc": btc
        })

        # All inputs are exchange addresses
        self.exchange_addresses.update(inputs)

        # Outputs are hot wallets
        self.hot_wallets.update(outputs)

        return True

    def get_all_exchange_addresses(self) -> Set[str]:
        """Get all addresses identified as exchange."""
        return self.exchange_addresses | self.hot_wallets

    def get_stats(self) -> Dict:
        """Get consolidation stats."""
        return {
            "consolidations": len(self.consolidations),
            "exchange_addresses": len(self.exchange_addresses),
            "hot_wallets": len(self.hot_wallets),
            "total_unique": len(self.get_all_exchange_addresses())
        }
