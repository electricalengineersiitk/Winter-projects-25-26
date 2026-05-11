"""
protocol.py  —  OTG Serial Packet Format
Smart Multimeter Simulation | End-Term Project

Defines how measurement data is packed into bytes and sent over
a USB OTG serial connection from the microcontroller to the mobile app.

Packet structure (12 bytes total):
  Byte 0     : Start byte         (0xAA)
  Byte 1     : Mode byte          (0x01=R, 0x02=C, 0x03=L)
  Byte 2     : Range index        (0x01 to 0x05)
  Bytes 3-6  : Measured value     (32-bit float, big-endian)
  Bytes 7-8  : Error * 100        (16-bit unsigned int, e.g. 42 = 0.42%)
  Byte 9     : Status byte        (0x00=SETTLED, 0x01=STEP_UP,
                                   0x02=STEP_DOWN, 0x03=OL)
  Byte 10    : Checksum           (XOR of bytes 1 through 9)
  Byte 11    : Stop byte          (0xFF)
"""

import struct

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

START_BYTE = 0xAA
STOP_BYTE  = 0xFF

MODE_BYTES = {
    "R": 0x01,
    "C": 0x02,
    "L": 0x03,
}

STATUS_BYTES = {
    "SETTLED":   0x00,
    "STEP_UP":   0x01,
    "STEP_DOWN": 0x02,
    "OL":        0x03,
}

PACKET_SIZE = 12   # total bytes per packet


# ─────────────────────────────────────────────
# Build a packet
# ─────────────────────────────────────────────

def build_packet(mode, range_index, measured_value, error_pct, status):
    """
    Pack one measurement reading into a 12-byte OTG serial packet.

    Args:
        mode          : "R", "C", or "L"
        range_index   : active range (1 to 5)
        measured_value: the measured component value (float)
        error_pct     : percentage error (float, e.g. 0.42)
        status        : "SETTLED", "STEP_UP", "STEP_DOWN", or "OL"

    Returns:
        packet: bytes object, 12 bytes long
    """
    mode_byte   = MODE_BYTES[mode]
    range_byte  = range_index & 0xFF
    value_bytes = struct.pack(">f", measured_value)       # 4 bytes, big-endian float
    error_int   = int(round(error_pct * 100)) & 0xFFFF   # scale to int, 2 bytes
    error_bytes = struct.pack(">H", error_int)
    status_byte = STATUS_BYTES.get(status, 0x00)

    # XOR checksum over bytes 1 through 9
    payload = bytes([mode_byte, range_byte]) + value_bytes + error_bytes + bytes([status_byte])
    checksum = 0
    for b in payload:
        checksum ^= b

    packet = (
        bytes([START_BYTE])
        + payload
        + bytes([checksum])
        + bytes([STOP_BYTE])
    )

    return packet


# ─────────────────────────────────────────────
# Parse a packet back into fields
# ─────────────────────────────────────────────

def parse_packet(packet):
    """
    Decode a 12-byte OTG packet back into readable fields.

    Args:
        packet: bytes object, 12 bytes long

    Returns:
        dict with keys: mode, range_index, measured_value, error_pct, status
        Returns None if packet is invalid (bad start/stop/checksum).
    """
    if len(packet) != PACKET_SIZE:
        print(f"  [ERROR] Expected {PACKET_SIZE} bytes, got {len(packet)}")
        return None

    if packet[0] != START_BYTE:
        print(f"  [ERROR] Bad start byte: {hex(packet[0])}")
        return None

    if packet[-1] != STOP_BYTE:
        print(f"  [ERROR] Bad stop byte: {hex(packet[-1])}")
        return None

    # Verify checksum (XOR of bytes 1 through 9)
    payload   = packet[1:10]
    checksum  = 0
    for b in payload:
        checksum ^= b

    if checksum != packet[10]:
        print(f"  [ERROR] Checksum mismatch: expected {hex(checksum)}, got {hex(packet[10])}")
        return None

    # Decode fields
    mode_byte    = packet[1]
    range_index  = packet[2]
    value_bytes  = packet[3:7]
    error_bytes  = packet[7:9]
    status_byte  = packet[9]

    mode_lookup   = {v: k for k, v in MODE_BYTES.items()}
    status_lookup = {v: k for k, v in STATUS_BYTES.items()}

    measured_value = struct.unpack(">f", value_bytes)[0]
    error_pct      = struct.unpack(">H", error_bytes)[0] / 100.0

    return {
        "mode":          mode_lookup.get(mode_byte, "?"),
        "range_index":   range_index,
        "measured_value": measured_value,
        "error_pct":     error_pct,
        "status":        status_lookup.get(status_byte, "?"),
    }


def format_packet_hex(packet):
    """Return a hex string representation of a packet for debugging."""
    return " ".join(f"{b:02X}" for b in packet)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  protocol.py - OTG Serial Packet Format Self-Test")
    print("=" * 62)

    test_cases = [
        ("R", 3, 4750.25,  0.42, "SETTLED"),
        ("C", 2, 85.3e-9,  0.71, "STEP_UP"),
        ("L", 4, 8.12e-3,  0.55, "STEP_DOWN"),
        ("R", 5, 980000.0, 1.20, "OL"),
    ]

    print(f"\n  {'Mode':<6} {'Range':<7} {'Value':<14} {'Err%':<8} {'Status':<12} {'Packet (hex)'}")
    print(f"  {'-'*75}")

    all_ok = True
    for mode, ridx, val, err, status in test_cases:
        pkt    = build_packet(mode, ridx, val, err, status)
        parsed = parse_packet(pkt)

        ok = (
            parsed is not None and
            parsed["mode"]        == mode   and
            parsed["range_index"] == ridx   and
            parsed["status"]      == status
        )
        all_ok = all_ok and ok

        tag = "OK" if ok else "FAIL"
        print(f"  {mode:<6} {ridx:<7} {val:<14.4g} {err:<8.2f} {status:<12} "
              f"{format_packet_hex(pkt)}  [{tag}]")

    print()
    if all_ok:
        print("  All packets built and parsed correctly.")
    else:
        print("  Some packets failed — check build/parse logic.")

    print()
    print("  Packet structure (12 bytes):")
    print("    Byte  0     : Start byte  (0xAA)")
    print("    Byte  1     : Mode        (0x01=R  0x02=C  0x03=L)")
    print("    Byte  2     : Range index (0x01 to 0x05)")
    print("    Bytes 3-6   : Value       (32-bit float, big-endian)")
    print("    Bytes 7-8   : Error x100  (16-bit uint)")
    print("    Byte  9     : Status      (0x00=SETTLED  0x01=UP  0x02=DOWN  0x03=OL)")
    print("    Byte  10    : Checksum    (XOR of bytes 1-9)")
    print("    Byte  11    : Stop byte   (0xFF)")
    print("=" * 62)
