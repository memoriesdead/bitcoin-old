"""
Transaction Decoder - Parse raw Bitcoin transactions.
Supports P2PKH (1...), P2SH (3...), P2WPKH/P2WSH (bc1q...), P2TR (bc1p...).
"""
import hashlib
import struct
from typing import List, Dict, Optional
from io import BytesIO

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


class TransactionDecoder:
    """Decode raw Bitcoin transactions from ZMQ."""

    def __init__(self):
        self.decoded_count = 0

    def decode(self, raw_tx: bytes) -> Optional[Dict]:
        try:
            stream = BytesIO(raw_tx)
            tx = self._parse_tx(stream)
            self.decoded_count += 1
            return tx
        except:
            return None

    def _parse_tx(self, stream: BytesIO) -> Dict:
        start = stream.tell()
        version = struct.unpack('<I', stream.read(4))[0]

        marker = stream.read(1)
        is_segwit = False
        if marker == b'\x00':
            flag = stream.read(1)
            if flag == b'\x01':
                is_segwit = True
            else:
                stream.seek(stream.tell() - 2)
        else:
            stream.seek(stream.tell() - 1)

        input_count = self._varint(stream)
        inputs = [self._parse_input(stream) for _ in range(input_count)]

        output_count = self._varint(stream)
        outputs = []
        total_btc = 0.0
        for _ in range(output_count):
            out = self._parse_output(stream)
            outputs.append(out)
            total_btc += out['btc']

        if is_segwit:
            for i in range(input_count):
                wit_count = self._varint(stream)
                wit_items = [stream.read(self._varint(stream)) for _ in range(wit_count)]
                if inputs[i]['address'] is None and len(wit_items) >= 2:
                    pk = wit_items[-1]
                    if len(pk) == 33 and pk[0] in (0x02, 0x03):
                        inputs[i]['address'] = self._pk_to_p2wpkh(pk)

        locktime = struct.unpack('<I', stream.read(4))[0]
        end = stream.tell()
        stream.seek(start)
        tx_bytes = stream.read(end - start)
        txid = hashlib.sha256(hashlib.sha256(tx_bytes).digest()).digest()[::-1].hex()

        return {'txid': txid, 'inputs': inputs, 'outputs': outputs, 'total_btc': total_btc, 'is_segwit': is_segwit}

    def _parse_input(self, stream: BytesIO) -> Dict:
        prev_txid = stream.read(32)[::-1].hex()
        prev_vout = struct.unpack('<I', stream.read(4))[0]
        script_len = self._varint(stream)
        script_sig = stream.read(script_len)
        sequence = struct.unpack('<I', stream.read(4))[0]
        address = self._addr_from_scriptsig(script_sig)
        return {'prev_txid': prev_txid, 'prev_vout': prev_vout, 'address': address, 'btc': 0}

    def _parse_output(self, stream: BytesIO) -> Dict:
        value = struct.unpack('<Q', stream.read(8))[0]
        btc = value / 100_000_000
        script_len = self._varint(stream)
        script = stream.read(script_len)
        address = self._script_to_addr(script)
        return {'btc': btc, 'address': address, 'script_pubkey': script.hex()}

    def _varint(self, stream: BytesIO) -> int:
        first = struct.unpack('<B', stream.read(1))[0]
        if first < 0xfd:
            return first
        elif first == 0xfd:
            return struct.unpack('<H', stream.read(2))[0]
        elif first == 0xfe:
            return struct.unpack('<I', stream.read(4))[0]
        return struct.unpack('<Q', stream.read(8))[0]

    def _addr_from_scriptsig(self, script: bytes) -> Optional[str]:
        if len(script) < 34:
            return None
        try:
            if len(script) >= 34:
                pk_len = script[-34]
                if pk_len == 33:
                    pk = script[-33:]
                    if pk[0] in (0x02, 0x03):
                        return self._pk_to_addr(pk)
            if len(script) >= 66:
                pk_len = script[-66]
                if pk_len == 65:
                    pk = script[-65:]
                    if pk[0] == 0x04:
                        return self._pk_to_addr(pk)
        except:
            pass
        return None

    def _pk_to_addr(self, pk: bytes) -> str:
        h160 = hashlib.new('ripemd160', hashlib.sha256(pk).digest()).digest()
        return self._h160_to_addr(h160, 0x00)

    def _pk_to_p2wpkh(self, pk: bytes) -> str:
        h160 = hashlib.new('ripemd160', hashlib.sha256(pk).digest()).digest()
        return self._bech32('bc', 0, h160)

    def _script_to_addr(self, script: bytes) -> Optional[str]:
        if len(script) == 0:
            return None
        # P2PKH: 76 a9 14 <20> 88 ac
        if len(script) == 25 and script[:3] == b'\x76\xa9\x14' and script[23:] == b'\x88\xac':
            return self._h160_to_addr(script[3:23], 0x00)
        # P2SH: a9 14 <20> 87
        if len(script) == 23 and script[:2] == b'\xa9\x14' and script[22] == 0x87:
            return self._h160_to_addr(script[2:22], 0x05)
        # P2WPKH: 00 14 <20>
        if len(script) == 22 and script[:2] == b'\x00\x14':
            return self._bech32('bc', 0, script[2:22])
        # P2WSH: 00 20 <32>
        if len(script) == 34 and script[:2] == b'\x00\x20':
            return self._bech32('bc', 0, script[2:34])
        # P2TR: 51 20 <32>
        if len(script) == 34 and script[:2] == b'\x51\x20':
            return self._bech32m('bc', 1, script[2:34])
        return None

    def _h160_to_addr(self, h160: bytes, ver: int) -> str:
        payload = bytes([ver]) + h160
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return self._b58_encode(payload + checksum)

    def _b58_encode(self, data: bytes) -> str:
        num = int.from_bytes(data, 'big')
        result = ''
        while num > 0:
            num, rem = divmod(num, 58)
            result = BASE58_ALPHABET[rem] + result
        for b in data:
            if b == 0:
                result = '1' + result
            else:
                break
        return result

    def _bech32(self, hrp: str, ver: int, data: bytes) -> str:
        return self._bech32_impl(hrp, ver, data, False)

    def _bech32m(self, hrp: str, ver: int, data: bytes) -> str:
        return self._bech32_impl(hrp, ver, data, True)

    def _bech32_impl(self, hrp: str, ver: int, data: bytes, m: bool) -> str:
        bits, acc, conv = 0, 0, [ver]
        for b in data:
            acc = (acc << 8) | b
            bits += 8
            while bits >= 5:
                bits -= 5
                conv.append((acc >> bits) & 31)
        if bits > 0:
            conv.append((acc << (5 - bits)) & 31)
        const = 0x2bc830a3 if m else 1
        poly = self._polymod(self._hrp_expand(hrp) + conv + [0]*6) ^ const
        checksum = [(poly >> 5*(5-i)) & 31 for i in range(6)]
        return hrp + '1' + ''.join(BECH32_CHARSET[d] for d in conv + checksum)

    def _polymod(self, vals: List[int]) -> int:
        gen = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
        chk = 1
        for v in vals:
            top = chk >> 25
            chk = (chk & 0x1ffffff) << 5 ^ v
            for i in range(5):
                chk ^= gen[i] if ((top >> i) & 1) else 0
        return chk

    def _hrp_expand(self, hrp: str) -> List[int]:
        return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
