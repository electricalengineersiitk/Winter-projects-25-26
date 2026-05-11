"""
protocol.py — OTG Serial Packet Format for Smart Multimeter
Formats measurement data into a binary-compatible serial packet
for transmission over USB OTG to a mobile app.

Packet Structure (16 bytes total):
┌──────────┬──────────┬──────────────┬──────────┬──────────┬──────────┐
│  START   │   MODE   │    VALUE     │  RANGE   │  FLAGS   │   CRC8   │
│  1 byte  │  1 byte  │   8 bytes    │  1 byte  │  1 byte  │  1 byte  │
│  0xAA    │  R/C/L   │  IEEE 754 64 │  1–5     │ bitfield │ checksum │
└──────────┴──────────┴──────────────┴──────────┴──────────┴──────────┘
Total: 13 bytes core + 3 bytes header/footer = 16 bytes

Flags byte:
  Bit 0 — SETTLED   (1 = range is stable)
  Bit 1 — OVERLOAD  (1 = value exceeds all ranges)
  Bit 2 — ERROR     (1 = measurement error)
  Bit 3–7 — reserved (0)
"""

import struct
import time

# Constants
PACKET_START = 0xAA
PACKET_END   = 0x55
PACKET_SIZE  = 14   # bytes

MODE_CODES = {
    "R": 0x01,
    "C": 0x02,
    "L": 0x03,
}
MODE_NAMES = {v: k for k, v in MODE_CODES.items()}

# Flag bits
FLAG_SETTLED  = 0b00000001
FLAG_OVERLOAD = 0b00000010
FLAG_ERROR    = 0b00000100


def crc8(data: bytes) -> int:
    """
    CRC-8 (polynomial 0x07, Dallas/Maxim standard).
    Provides basic integrity check for serial transmission.
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


def encode_packet(
    mode: str,
    value: float,
    range_index: int,
    settled: bool = True,
    overload: bool = False,
    error: bool = False,
) -> bytes:
    """
    Encode a measurement result into a 14-byte OTG serial packet.

    Args:
        mode:        Measurement mode — "R", "C", or "L"
        value:       Measured value in SI units (Ω, F, H)
        range_index: 0-based range index (0–4)
        settled:     True if auto-ranging has settled
        overload:    True if value exceeds all ranges
        error:       True if measurement failed

    Returns:
        14-byte packet as bytes object

    Raises:
        ValueError: if mode is not recognized
    """
    if mode not in MODE_CODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(MODE_CODES.keys())}")

    mode_byte = MODE_CODES[mode]
    range_byte = range_index + 1  # store 1-based

    flags = 0x00
    if settled:
        flags |= FLAG_SETTLED
    if overload:
        flags |= FLAG_OVERLOAD
    if error:
        flags |= FLAG_ERROR

    # Pack: mode(1) + value(8, double) + range(1) + flags(1)
    payload = struct.pack(">BdBB", mode_byte, value, range_byte, flags)

    checksum = crc8(payload)

    # Full packet: START(1) + payload(11) + CRC(1) + END(1) = 14 bytes
    packet = bytes([PACKET_START]) + payload + bytes([checksum, PACKET_END])
    return packet


def decode_packet(packet: bytes) -> dict:
    """
    Decode a 14-byte OTG serial packet.

    Args:
        packet: 14-byte bytes object

    Returns:
        dict with keys: mode, value, range_index, settled, overload, error, crc_ok

    Raises:
        ValueError: if packet length or framing bytes are invalid
    """
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(packet)}")
    if packet[0] != PACKET_START:
        raise ValueError(f"Bad START byte: 0x{packet[0]:02X}")
    if packet[-1] != PACKET_END:
        raise ValueError(f"Bad END byte: 0x{packet[-1]:02X}")

    payload = packet[1:-2]
    received_crc = packet[-2]
    computed_crc = crc8(payload)
    crc_ok = received_crc == computed_crc

    mode_byte, value, range_byte, flags = struct.unpack(">BdBB", payload)

    return {
        "mode":        MODE_NAMES.get(mode_byte, "?"),
        "value":       value,
        "range_index": range_byte - 1,          # back to 0-based
        "settled":     bool(flags & FLAG_SETTLED),
        "overload":    bool(flags & FLAG_OVERLOAD),
        "error":       bool(flags & FLAG_ERROR),
        "crc_ok":      crc_ok,
        "raw_hex":     packet.hex(" ").upper(),
    }


def format_packet_hex(packet: bytes) -> str:
    """Return a human-readable hex dump of a packet."""
    labels = ["START", "MODE ", "VALUE(8B)                       ",
              "RANGE", "FLAGS", "CRC  ", "END  "]
    parts = [
        packet[0:1], packet[1:2], packet[2:10],
        packet[10:11], packet[11:12], packet[12:13], packet[13:14]
    ]
    lines = ["  OTG Packet Hex Dump:"]
    for label, part in zip(labels, parts):
        hex_str = " ".join(f"{b:02X}" for b in part)
        lines.append(f"    {label}: {hex_str}")
    return "\n".join(lines)


# sample test
if __name__ == "__main__":
    test_cases = [
        ("R", 4700.0,   2, True,  False, False),
        ("C", 47e-9,    1, True,  False, False),
        ("L", 1.5e-3,   3, False, False, False),
        ("R", 1.1e6,    4, False, True,  False),  # overload
    ]

    print("\nprotocol.py — OTG Packet Encoder/Decoder Self-Test")
    print("=" * 60)
    for mode, value, ridx, settled, overload, error in test_cases:
        pkt = encode_packet(mode, value, ridx, settled, overload, error)
        decoded = decode_packet(pkt)
        crc_status = "✓" if decoded["crc_ok"] else "✗"

        print(f"\n  Mode={mode}  Value={value:.4g}  Range={ridx+1}"
              f"  Settled={settled}  Overload={overload}")
        print(format_packet_hex(pkt))
        print(f"  Decoded: mode={decoded['mode']}  value={decoded['value']:.4g}"
              f"  range={decoded['range_index']+1}"
              f"  settled={decoded['settled']}  overload={decoded['overload']}"
              f"  CRC {crc_status}")

    print(f"\n  Packet size: {PACKET_SIZE} bytes per reading")
    print("  Protocol: START(0xAA) | MODE(1B) | VALUE(8B) | RANGE(1B) | FLAGS(1B) | CRC8(1B) | END(0x55)")
