"""
protocol.py — OTG Serial Packet Format
Smart Multimeter Simulation | End-Term Project

Defines the packet structure used to transmit measurement results
from the microcontroller to the mobile app over USB OTG serial.

Packet layout (16 bytes total):
    [0]       SOF       — Start of frame: 0xAA
    [1]       MODE      — 0x01=R, 0x02=C, 0x03=L
    [2]       RANGE     — Active range index (1–5), 0xFF = OL
    [3-6]     VALUE     — 32-bit float, little-endian (measured value in SI units)
    [7-10]    ERROR     — 32-bit float, little-endian (% error, informational)
    [11-12]   SEQ       — 16-bit unsigned sequence number (little-endian), rolls at 65535
    [13]      FLAGS     — Bit 0: overload, Bit 1: settling, Bit 2: range_changed
    [14]      CHECKSUM  — XOR of bytes [0..13]
    [15]      EOF       — End of frame: 0x55

Total: 16 bytes per packet at up to 115200 baud → ~720 packets/second theoretical max.
Practical rate: 10–50 Hz for stable readings.
"""

import struct

SOF = 0xAA
EOF_BYTE = 0x55

MODE_R = 0x01
MODE_C = 0x02
MODE_L = 0x03

MODE_MAP = {"R": MODE_R, "C": MODE_C, "L": MODE_L}
MODE_NAMES = {MODE_R: "R", MODE_C: "C", MODE_L: "L"}


class PacketFlags:
    OVERLOAD      = 0b00000001
    SETTLING      = 0b00000010
    RANGE_CHANGED = 0b00000100


def _checksum(data: bytes) -> int:
    """XOR checksum of all bytes in data."""
    result = 0
    for b in data:
        result ^= b
    return result


def encode_packet(
    mode: str,
    range_idx: int,
    value: float,
    error_pct: float,
    seq: int,
    overload: bool = False,
    settling: bool = False,
    range_changed: bool = False,
) -> bytes:
    """
    Build a 16-byte OTG serial packet.

    Args:
        mode:          "R", "C", or "L"
        range_idx:     Active range (1–5), or 0 for OL
        value:         Measured value in SI units (Ω, F, or H)
        error_pct:     Measurement error percentage (float)
        seq:           Sequence number (0–65535)
        overload:      True if reading exceeds all ranges
        settling:      True if range transition pending
        range_changed: True if range just switched

    Returns:
        16-byte packet as bytes
    """
    mode_byte = MODE_MAP.get(mode.upper(), 0x00)
    range_byte = 0xFF if overload else (range_idx & 0xFF)

    value_bytes = struct.pack("<f", float(value))
    error_bytes = struct.pack("<f", float(error_pct))
    seq_bytes   = struct.pack("<H", seq & 0xFFFF)

    flags = 0
    if overload:      flags |= PacketFlags.OVERLOAD
    if settling:      flags |= PacketFlags.SETTLING
    if range_changed: flags |= PacketFlags.RANGE_CHANGED

    payload = bytes([SOF, mode_byte, range_byte]) + value_bytes + error_bytes + seq_bytes + bytes([flags])
    # payload is 14 bytes at this point
    cs = _checksum(payload)
    packet = payload + bytes([cs, EOF_BYTE])
    assert len(packet) == 16, f"Packet length error: {len(packet)}"
    return packet


def decode_packet(packet: bytes) -> dict:
    """
    Parse a 16-byte OTG packet.

    Returns:
        dict with fields, or raises ValueError on malformed packet.
    """
    if len(packet) != 16:
        raise ValueError(f"Expected 16 bytes, got {len(packet)}")
    if packet[0] != SOF:
        raise ValueError(f"Bad SOF: 0x{packet[0]:02X}")
    if packet[15] != EOF_BYTE:
        raise ValueError(f"Bad EOF: 0x{packet[15]:02X}")

    cs_computed = _checksum(packet[:14])
    cs_received = packet[14]
    if cs_computed != cs_received:
        raise ValueError(f"Checksum mismatch: computed=0x{cs_computed:02X}, received=0x{cs_received:02X}")

    mode_byte  = packet[1]
    range_byte = packet[2]
    value,     = struct.unpack_from("<f", packet, 3)
    error_pct, = struct.unpack_from("<f", packet, 7)
    seq,       = struct.unpack_from("<H", packet, 11)
    flags      = packet[13]

    return {
        "mode":          MODE_NAMES.get(mode_byte, "?"),
        "range":         None if range_byte == 0xFF else range_byte,
        "value":         value,
        "error_pct":     error_pct,
        "seq":           seq,
        "overload":      bool(flags & PacketFlags.OVERLOAD),
        "settling":      bool(flags & PacketFlags.SETTLING),
        "range_changed": bool(flags & PacketFlags.RANGE_CHANGED),
    }


def pretty_print_packet(pkt_dict: dict) -> str:
    """Return a human-readable summary of a decoded packet."""
    mode  = pkt_dict["mode"]
    units = {"R": "Ω", "C": "F", "L": "H"}.get(mode, "?")
    rng   = "OL" if pkt_dict["overload"] else f"Range {pkt_dict['range']}"
    flags = []
    if pkt_dict["overload"]:      flags.append("OVERLOAD")
    if pkt_dict["settling"]:      flags.append("SETTLING")
    if pkt_dict["range_changed"]: flags.append("RANGE_CHANGED")
    flag_str = ", ".join(flags) if flags else "—"

    return (
        f"[Seq {pkt_dict['seq']:05d}] Mode={mode} | {rng} | "
        f"Value={pkt_dict['value']:.6g} {units} | "
        f"Error={pkt_dict['error_pct']:.3f}% | Flags: {flag_str}"
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== protocol.py self-test ===\n")

    # Encode a resistance packet
    pkt = encode_packet("R", range_idx=3, value=4_723.1, error_pct=0.49,
                        seq=1, range_changed=True)
    print(f"Encoded packet ({len(pkt)} bytes): {pkt.hex(' ').upper()}\n")

    # Decode it back
    decoded = decode_packet(pkt)
    print("Decoded:")
    print(pretty_print_packet(decoded))

    # Encode a capacitance OL packet
    pkt_ol = encode_packet("C", range_idx=0, value=0.0, error_pct=0.0,
                           seq=2, overload=True)
    decoded_ol = decode_packet(pkt_ol)
    print("\nOverload packet:")
    print(pretty_print_packet(decoded_ol))
