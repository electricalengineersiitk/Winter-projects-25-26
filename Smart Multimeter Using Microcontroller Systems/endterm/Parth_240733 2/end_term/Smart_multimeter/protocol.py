"""
protocol.py — OTG Serial Packet Format
Smart Multimeter Simulation

Packet structure (12 bytes, little-endian):
┌──────┬──────┬──────────┬──────────┬──────┬──────┐
│ SOF  │ MODE │  VALUE   │  ERROR   │RANGE │ CRC  │
│ 0xAA │ 1 B  │  4 B f32 │  2 B u16 │ 1 B  │ 2 B  │
└──────┴──────┴──────────┴──────────┴──────┴──────┘
Total: 1 + 1 + 4 + 2 + 1 + 2 = 11 bytes  +  SOF = 12 bytes

SOF  : 0xAA  (start of frame marker)
MODE : 0x01 = Resistance | 0x02 = Capacitance | 0x03 = Inductance
VALUE: 32-bit float, measured value in SI units (Ω / F / H)
ERROR: 16-bit unsigned, error × 100 (e.g. 0.75% → 75)
RANGE: 1–5 (active range index)
CRC  : CRC-16/CCITT-FALSE over bytes [SOF .. RANGE]
"""

import struct
import json

# ---- Mode codes ----
MODE_RESISTANCE  = 0x01
MODE_CAPACITANCE = 0x02
MODE_INDUCTANCE  = 0x03

MODE_CODES = {
    "R": MODE_RESISTANCE,
    "C": MODE_CAPACITANCE,
    "L": MODE_INDUCTANCE,
}

SOF = 0xAA
BAUD_RATE = 115200


# ---------------------------------------------------------------------------
# CRC-16/CCITT-FALSE
# ---------------------------------------------------------------------------

def _crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF, no reflect)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Packet builder
# ---------------------------------------------------------------------------

def build_packet(mode: str, value: float, error_pct: float, range_idx: int) -> bytes:
    """
    Encode a measurement result into the 12-byte OTG serial packet.

    Args:
        mode      : "R", "C", or "L"
        value     : Measured value in SI units
        error_pct : Measurement error as a percentage (e.g. 0.75)
        range_idx : Active range index (1–5)

    Returns:
        12-byte packet as bytes
    """
    mode_byte  = MODE_CODES[mode]
    error_u16  = min(int(error_pct * 100), 0xFFFF)   # centipercent, clamped
    range_byte = max(1, min(range_idx, 5))

    payload = struct.pack("<BfHB", mode_byte, value, error_u16, range_byte)
    crc     = _crc16(bytes([SOF]) + payload)
    crc_bytes = struct.pack("<H", crc)

    return bytes([SOF]) + payload + crc_bytes


# ---------------------------------------------------------------------------
# Packet parser
# ---------------------------------------------------------------------------

def parse_packet(packet: bytes) -> dict | None:
    """
    Decode a 12-byte OTG serial packet.

    Returns a dict on success, or None if CRC fails / wrong length.
    """
    if len(packet) != 12:
        return None

    if packet[0] != SOF:
        return None

    # Verify CRC over [SOF .. RANGE] (first 10 bytes)
    received_crc = struct.unpack_from("<H", packet, 10)[0]
    computed_crc = _crc16(packet[:10])
    if received_crc != computed_crc:
        return None

    mode_byte, value, error_u16, range_byte = struct.unpack_from("<BfHB", packet, 1)
    mode_name = {v: k for k, v in MODE_CODES.items()}.get(mode_byte, "?")

    return {
        "mode":      mode_name,
        "value":     value,
        "error_pct": error_u16 / 100.0,
        "range":     range_byte,
    }


def packet_to_json(packet: bytes) -> str:
    """Convenience: decode packet and return JSON string."""
    parsed = parse_packet(packet)
    if parsed is None:
        return json.dumps({"error": "bad_packet"})
    return json.dumps(parsed, indent=2)


# ---------------------------------------------------------------------------
# Serial framing helpers
# ---------------------------------------------------------------------------

class SerialFramer:
    """
    Accumulate raw bytes from UART and yield complete 12-byte packets.
    Usage:
        framer = SerialFramer()
        for byte_chunk in uart_stream:
            for pkt in framer.feed(byte_chunk):
                process(pkt)
    """

    PACKET_LEN = 12

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        """Feed raw bytes; yield complete packets as they are found."""
        self._buf.extend(data)
        while len(self._buf) >= self.PACKET_LEN:
            # Scan for SOF
            sof_pos = self._buf.find(SOF)
            if sof_pos == -1:
                self._buf.clear()
                break
            if sof_pos > 0:
                del self._buf[:sof_pos]   # discard garbage before SOF
            if len(self._buf) < self.PACKET_LEN:
                break
            candidate = bytes(self._buf[:self.PACKET_LEN])
            parsed = parse_packet(candidate)
            if parsed is not None:
                yield candidate
                del self._buf[:self.PACKET_LEN]
            else:
                del self._buf[0]   # not a valid packet at this position


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== protocol.py demo ===\n")

    # Build a packet for R = 4 700 Ω, 0.43% error, Range 2
    pkt = build_packet("R", 4700.0, 0.43, 2)
    print(f"Packet (hex): {pkt.hex(' ').upper()}")
    print(f"Length: {len(pkt)} bytes")

    # Parse it back
    decoded = parse_packet(pkt)
    print(f"\nDecoded: {decoded}")
    print(f"\nJSON:\n{packet_to_json(pkt)}")

    # Corrupt one byte and confirm CRC failure
    bad_pkt = bytearray(pkt)
    bad_pkt[5] ^= 0xFF
    result = parse_packet(bytes(bad_pkt))
    print(f"\nCorrupted packet → parse_packet returns: {result}  (None = CRC caught it)")
