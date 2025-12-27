"""
ZMQ Subscriber - Bitcoin Core real-time connection.
INFLOW = SHORT, OUTFLOW = LONG. 10-60s edge over WebSocket.
"""
import threading
import time
from typing import Callable, Optional, List
from collections import deque

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("[ZMQ] WARNING: pip install pyzmq")


class BlockchainZMQ:
    """Direct ZMQ connection to Bitcoin Core for nanosecond latency."""

    DEFAULT_RAWTX = "tcp://127.0.0.1:28332"
    DEFAULT_RAWBLOCK = "tcp://127.0.0.1:28333"

    def __init__(self, rawtx_endpoint: str = None, rawblock_endpoint: str = None,
                 on_transaction: Callable[[bytes], None] = None,
                 on_block: Callable[[bytes], None] = None):
        if not HAS_ZMQ:
            raise ImportError("pyzmq required: pip install pyzmq")

        self.rawtx_endpoint = rawtx_endpoint or self.DEFAULT_RAWTX
        self.rawblock_endpoint = rawblock_endpoint or self.DEFAULT_RAWBLOCK
        self.on_transaction = on_transaction
        self.on_block = on_block

        self.context = zmq.Context()
        self.tx_socket = None
        self.block_socket = None
        self.running = False
        self.connected = False
        self.tx_thread = None
        self.block_thread = None

        self.tx_count = 0
        self.block_count = 0
        self.last_tx_time = 0.0
        self.recent_txs = deque(maxlen=1000)
        self.lock = threading.Lock()

    def connect(self) -> bool:
        try:
            # TX socket with stability settings
            self.tx_socket = self.context.socket(zmq.SUB)
            self.tx_socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
            self.tx_socket.setsockopt(zmq.RCVTIMEO, 500)  # 500ms timeout for faster shutdown
            self.tx_socket.setsockopt(zmq.RCVHWM, 10000)  # High water mark
            self.tx_socket.connect(self.rawtx_endpoint)
            self.tx_socket.setsockopt(zmq.SUBSCRIBE, b"rawtx")

            # Block socket with stability settings
            self.block_socket = self.context.socket(zmq.SUB)
            self.block_socket.setsockopt(zmq.LINGER, 0)
            self.block_socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.block_socket.connect(self.rawblock_endpoint)
            self.block_socket.setsockopt(zmq.SUBSCRIBE, b"rawblock")

            self.connected = True
            print(f"[ZMQ] Connected to {self.rawtx_endpoint}")
            return True
        except Exception as e:
            print(f"[ZMQ] Connection failed: {e}")
            return False

    def start(self) -> bool:
        if not self.connected and not self.connect():
            return False

        self.running = True
        self.tx_thread = threading.Thread(target=self._listen_tx, daemon=True)
        self.block_thread = threading.Thread(target=self._listen_blocks, daemon=True)
        self.tx_thread.start()
        self.block_thread.start()
        print("[ZMQ] Listening...")
        return True

    def stop(self):
        self.running = False
        self.connected = False

        # Give threads time to exit gracefully
        time.sleep(0.1)

        # Close sockets first (this will unblock recv calls)
        for sock in [self.tx_socket, self.block_socket]:
            try:
                if sock:
                    sock.close()
            except:
                pass

        # Now wait for threads
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)
        if self.block_thread and self.block_thread.is_alive():
            self.block_thread.join(timeout=1.0)

        # Terminate context last
        try:
            self.context.term()
        except:
            pass

    def _listen_tx(self):
        while self.running:
            try:
                if not self.tx_socket or self.tx_socket.closed:
                    break
                msg = self.tx_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(msg) >= 2:
                    raw_tx = msg[1]
                    with self.lock:
                        self.tx_count += 1
                        self.last_tx_time = time.time()
                        self.recent_txs.append({'raw': raw_tx, 'time': self.last_tx_time})
                    if self.on_transaction:
                        try:
                            self.on_transaction(raw_tx)
                        except Exception as e:
                            print(f"[ZMQ] Callback error: {e}")
            except zmq.Again:
                time.sleep(0.01)  # Small sleep on no data
            except zmq.ZMQError as e:
                if self.running:
                    time.sleep(0.1)
            except Exception as e:
                if self.running:
                    print(f"[ZMQ] TX error: {e}")
                break

    def _listen_blocks(self):
        while self.running:
            try:
                if not self.block_socket or self.block_socket.closed:
                    break
                msg = self.block_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(msg) >= 2:
                    with self.lock:
                        self.block_count += 1
                    if self.on_block:
                        self.on_block(msg[1])
            except zmq.Again:
                time.sleep(0.1)  # Blocks are less frequent
            except zmq.ZMQError:
                if self.running:
                    time.sleep(0.1)
            except Exception as e:
                if self.running:
                    print(f"[ZMQ] Block error: {e}")
                break

    def get_stats(self) -> dict:
        with self.lock:
            return {
                'connected': self.connected, 'tx_count': self.tx_count,
                'block_count': self.block_count, 'last_tx_time': self.last_tx_time,
            }
