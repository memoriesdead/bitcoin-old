"""
Exchange Flow Detector - INFLOW = SHORT, OUTFLOW = LONG.
"""
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from .exchange_wallets import ExchangeWalletTracker, EXCHANGE_DATABASE


@dataclass
class ExchangeFlow:
    timestamp: float
    exchange_id: str
    direction: int  # 1=OUTFLOW (LONG), -1=INFLOW (SHORT)
    btc_amount: float
    txid: str
    flow_type: str


class SimpleExchangeFlowDetector:
    """Detect BTC flowing TO/FROM exchanges."""

    def __init__(self, on_flow: Optional[Callable[[ExchangeFlow], None]] = None):
        self.on_flow = on_flow
        self.tracker = ExchangeWalletTracker()
        self.address_to_exchange: Dict[str, str] = {}
        for ex_id, info in EXCHANGE_DATABASE.items():
            for addr in info.addresses:
                self.address_to_exchange[addr] = ex_id

        self.txs_processed = 0
        self.inflows_detected = 0
        self.outflows_detected = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0
        print(f"[FLOW] Tracking {len(self.address_to_exchange)} addresses")

    def process_transaction(self, txid: str, inputs: List[tuple], outputs: List[tuple]) -> List[ExchangeFlow]:
        self.txs_processed += 1
        ts = time.time()
        flows = []

        for addr, btc in inputs:
            if addr in self.address_to_exchange:
                flow = ExchangeFlow(ts, self.address_to_exchange[addr], 1, btc, txid, "outflow")
                flows.append(flow)
                self.outflows_detected += 1
                self.total_outflow_btc += btc
                if self.on_flow:
                    self.on_flow(flow)

        for addr, btc in outputs:
            if addr in self.address_to_exchange:
                flow = ExchangeFlow(ts, self.address_to_exchange[addr], -1, btc, txid, "inflow")
                flows.append(flow)
                self.inflows_detected += 1
                self.total_inflow_btc += btc
                if self.on_flow:
                    self.on_flow(flow)

        return flows

    def get_stats(self) -> Dict:
        return {
            "txs_processed": self.txs_processed,
            "inflows": self.inflows_detected, "outflows": self.outflows_detected,
            "total_inflow_btc": self.total_inflow_btc, "total_outflow_btc": self.total_outflow_btc,
            "net_flow_btc": self.total_outflow_btc - self.total_inflow_btc,
        }
