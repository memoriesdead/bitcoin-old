"""
Bitcoin Core RPC - Direct node communication.
Zero external dependencies. Pure blockchain data.
"""
import json
import base64
from urllib.request import Request, urlopen
from urllib.error import URLError
from typing import Dict, List, Optional, Any


class BitcoinRPC:
    """
    Bitcoin Core JSON-RPC interface.

    Connects directly to your node. No third parties.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8332,
        user: str = "bitcoin",
        password: str = "bitcoin",
        timeout: int = 30
    ):
        self.url = f"http://{host}:{port}"
        self.auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.timeout = timeout
        self.request_id = 0

    def call(self, method: str, params: List = None) -> Any:
        """Execute RPC call."""
        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or []
        }

        req = Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Basic {self.auth}",
                "Content-Type": "application/json"
            }
        )

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
                if result.get("error"):
                    raise Exception(f"RPC Error: {result['error']}")
                return result.get("result")
        except URLError as e:
            raise Exception(f"RPC Connection Error: {e}")

    # ==========================================================================
    # BLOCK METHODS
    # ==========================================================================

    def getblockcount(self) -> int:
        """Get current block height."""
        return self.call("getblockcount")

    def getblockhash(self, height: int) -> str:
        """Get block hash at height."""
        return self.call("getblockhash", [height])

    def getblock(self, blockhash: str, verbosity: int = 2) -> Dict:
        """
        Get block data.

        verbosity=0: hex string
        verbosity=1: block with txids
        verbosity=2: block with full tx data (what we need)
        """
        return self.call("getblock", [blockhash, verbosity])

    def getblockbyheight(self, height: int, verbosity: int = 2) -> Dict:
        """Get block by height (convenience method)."""
        blockhash = self.getblockhash(height)
        return self.getblock(blockhash, verbosity)

    # ==========================================================================
    # TRANSACTION METHODS
    # ==========================================================================

    def getrawtransaction(self, txid: str, verbose: bool = True) -> Dict:
        """
        Get transaction data.

        verbose=True: decoded transaction
        verbose=False: hex string
        """
        return self.call("getrawtransaction", [txid, verbose])

    def decoderawtransaction(self, hex_string: str) -> Dict:
        """Decode raw transaction hex."""
        return self.call("decoderawtransaction", [hex_string])

    # ==========================================================================
    # MEMPOOL METHODS
    # ==========================================================================

    def getmempoolinfo(self) -> Dict:
        """Get mempool statistics."""
        return self.call("getmempoolinfo")

    def getrawmempool(self, verbose: bool = False) -> List:
        """Get all mempool transaction IDs."""
        return self.call("getrawmempool", [verbose])

    # ==========================================================================
    # ADDRESS METHODS (requires txindex=1)
    # ==========================================================================

    def scantxoutset(self, action: str, scanobjects: List) -> Dict:
        """
        Scan UTXO set for addresses.

        action: "start", "abort", "status"
        scanobjects: [{"desc": "addr(ADDRESS)"}]

        Returns UTXOs for address - useful for finding outputs to trace.
        """
        return self.call("scantxoutset", [action, scanobjects])

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def getblockchaininfo(self) -> Dict:
        """Get blockchain info (chain, blocks, headers, etc)."""
        return self.call("getblockchaininfo")

    def getnetworkinfo(self) -> Dict:
        """Get network info."""
        return self.call("getnetworkinfo")

    def test_connection(self) -> bool:
        """Test if RPC connection works."""
        try:
            info = self.getblockchaininfo()
            return info.get("chain") is not None
        except:
            return False


class RPCBatchProcessor:
    """
    Batch RPC calls for efficiency.

    Instead of 1000 individual calls, send 1 batch request.
    """

    def __init__(self, rpc: BitcoinRPC, batch_size: int = 100):
        self.rpc = rpc
        self.batch_size = batch_size
        self.pending = []

    def add(self, method: str, params: List = None):
        """Add call to batch."""
        self.pending.append({"method": method, "params": params or []})

    def execute(self) -> List:
        """Execute all pending calls."""
        if not self.pending:
            return []

        results = []
        for i in range(0, len(self.pending), self.batch_size):
            batch = self.pending[i:i + self.batch_size]
            batch_results = self._execute_batch(batch)
            results.extend(batch_results)

        self.pending = []
        return results

    def _execute_batch(self, batch: List) -> List:
        """Execute a single batch."""
        payload = []
        for idx, call in enumerate(batch):
            payload.append({
                "jsonrpc": "2.0",
                "id": idx,
                "method": call["method"],
                "params": call["params"]
            })

        req = Request(
            self.rpc.url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Basic {self.rpc.auth}",
                "Content-Type": "application/json"
            }
        )

        with urlopen(req, timeout=self.rpc.timeout * 2) as resp:
            results = json.loads(resp.read().decode())
            # Sort by id to maintain order
            results.sort(key=lambda x: x.get("id", 0))
            return [r.get("result") for r in results]


def get_rpc_from_env() -> BitcoinRPC:
    """Create RPC connection from environment or defaults."""
    import os
    return BitcoinRPC(
        host=os.getenv("BITCOIN_RPC_HOST", "127.0.0.1"),
        port=int(os.getenv("BITCOIN_RPC_PORT", "8332")),
        user=os.getenv("BITCOIN_RPC_USER", "bitcoin"),
        password=os.getenv("BITCOIN_RPC_PASSWORD", "bitcoin")
    )
