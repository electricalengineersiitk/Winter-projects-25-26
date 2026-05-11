"""
protocol.py — OTG Serial Packet Format for the Smart Multimeter.

Packet structure (12 bytes total):
┌────────┬────────┬──────────────────────┬────────┬────────┐
│ START  │  MODE  │       VALUE          │ RANGE  │  CRC8  │
│ 1 byte │ 1 byte │       4 bytes        │ 1 byte │ 1 byte │
└────────┴────────┴──────────────────────┴────────┴────────┘

  START  : 0xAA  (sync byte)
  MODE   : 0x01 = R,  0x02 = C,  0x03 = L
  VALUE  : IEEE-754 float32 (big-endian)  — measured value in SI units
  RANGE  : 1–5 (active range index)
  CRC8   : XOR of all preceding bytes (simple integrity check)

Usage:
    packet = encode_packet("R", 4750.3, 2)
    decoded = decode_packet(packet)
"""

import struct
from typing import Literal

START_BYTE = 0xAA

MODE_CODES = {"R": 0x01, "C": 0x02, "L": 0x03}
CODE_MODES = {v: k for k, v in MODE_CODES.items()}


# ---------------------------------------------------------------------------
# CRC-8 (simple XOR checksum)
# ---------------------------------------------------------------------------
def crc8(data: bytes) -> int:
    """Return the XOR checksum of all bytes in data."""
    result = 0
    for byte in data:
        result ^= byte
    return result


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def encode_packet(
    mode: Literal["R", "C", "L"],
    value: float,
    range_idx: int,
    status: str = "SETTLED",
) -> bytes:
    """
    Build a 12-byte OTG serial packet.

    Args:
        mode      : "R", "C", or "L"
        value     : Measured value (SI units: Ω, F, H)
        range_idx : Active range (1–5)
        status    : "SETTLED" | "SWITCHING_UP" | "SWITCHING_DOWN" | "OL"

    Returns:
        12-byte packet as bytes object.
    """
    if mode not in MODE_CODES:
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'R', 'C', or 'L'.")
    if not 1 <= range_idx <= 5:
        raise ValueError(f"range_idx must be 1–5, got {range_idx}.")

    status_byte = {
        "SETTLED":       0x00,
        "SWITCHING_UP":  0x01,
        "SWITCHING_DOWN":0x02,
        "OL":            0xFF,
    }.get(status, 0x00)

    header = bytes([START_BYTE, MODE_CODES[mode]])
    value_bytes = struct.pack(">f", value)           # big-endian float32
    tail = bytes([range_idx & 0xFF, status_byte])

    payload = header + value_bytes + tail
    checksum = crc8(payload)

    return payload + bytes([checksum])               # 9 + 1 = 10 bytes total


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def decode_packet(packet: bytes) -> dict:
    """
    Parse a 10-byte OTG serial packet.

    Returns dict with keys: start_ok, mode, value, range, status, crc_ok
    Raises ValueError if packet length is wrong.
    """
    if len(packet) != 9:
        raise ValueError(f"Expected 10-byte packet, got {len(packet)} bytes.")

    start   = packet[0]
    mode_b  = packet[1]
    value_b = packet[2:6]
    range_b = packet[6]
    status_b= packet[7]
    recv_crc= packet[8]

    calc_crc = crc8(packet[:9])

    value = struct.unpack(">f", value_b)[0]

    status_map = {0x00: "SETTLED", 0x01: "STEP_UP", 0x02: "STEP_DOWN", 0xFF: "OL"}

    return {
        "start_ok": start == START_BYTE,
        "mode":     CODE_MODES.get(mode_b, f"UNKNOWN(0x{mode_b:02X})"),
        "value":    value,
        "range":    range_b,
        "status":   status_map.get(status_b, "UNKNOWN"),
        "crc_ok":   recv_crc == calc_crc,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
UNITS = {"R": "Ω", "C": "F", "L": "H"}

def format_packet_hex(packet: bytes) -> str:
    return " ".join(f"{b:02X}" for b in packet)


# ---------------------------------------------------------------------------
# Demo when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== protocol.py self-test ===\n")

    test_cases = [
        ("R", 4750.3,  2, "SETTLED"),
        ("C", 47e-9,   1, "SWITCHING_UP"),
        ("L", 0.0033,  3, "SETTLED"),
        ("R", 1_500_000, 5, "OL"),
    ]

    for mode, val, rng, sts in test_cases:
        pkt = encode_packet(mode, val, rng, sts)
        dec = decode_packet(pkt)
        print(f"Mode={mode}  value={val:.4e}  range={rng}  status={sts}")
        print(f"  Packet : {format_packet_hex(pkt)}")
        print(f"  Decoded: {dec}")
        print(f"  CRC OK : {dec['crc_ok']}\n")
