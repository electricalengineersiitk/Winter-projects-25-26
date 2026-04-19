"""
protocol.py
-----------
OTG serial packet formatter for settled auto-range readings.

Each packet follows a compact ASCII framing scheme suitable for USB-OTG
or UART serial transmission at 9600 / 115200 baud.

Packet layout (32 bytes fixed, null-padded):
    [SOF][MODE:1][RANGE:1][VALUE:10][UNIT:3][ERR%:5][CHECKSUM:2][EOF]
    SOF = 0x02  (ASCII STX)
    EOF = 0x03  (ASCII ETX)

──────────────────────────────────────────────────────────────────────
 HARDWARE SIGNAL CHAIN – ASCII BLOCK DIAGRAM
──────────────────────────────────────────────────────────────────────

   Physical DUT              Front-End Circuit          MCU / ADC
  ┌─────────────┐           ┌───────────────────┐      ┌──────────┐
  │  R / C / L  │──probes──▶│  Mux + Op-Amp     │─────▶│  12-bit  │
  │  Component  │           │  Signal Conditioner│      │   ADC    │
  └─────────────┘           └───────┬───────────┘      └────┬─────┘
                                    │                        │
                            ┌───────▼───────┐      ┌────────▼──────┐
                            │  Anti-alias   │      │  Auto-Range   │
                            │  LPF  Filter  │      │  Engine (SW)  │
                            └───────┬───────┘      └────────┬──────┘
                                    │                        │
                            ┌───────▼───────────────────────▼──────┐
                            │           Firmware (STM32 / ESP32)    │
                            │   • Range switching via relay/MUX     │
                            │   • Hysteresis logic                   │
                            │   • OTG packet serialiser (this file) │
                            └───────────────────────┬───────────────┘
                                                    │  USB-OTG / UART
                                                    ▼
                                           ┌─────────────────┐
                                           │  Host PC / App  │
                                           │  (Console / GUI)│
                                           └─────────────────┘

──────────────────────────────────────────────────────────────────────
"""

import struct
import time
from dataclasses import dataclass

# Frame markers (ASCII STX / ETX)
SOF = b"\x02"
EOF = b"\x03"

UNIT_MAP = {"R": "OHM", "C": "  F", "L": "  H"}
MODE_NAMES = {"R": "Resistance", "C": "Capacitance", "L": "Inductance"}


@dataclass
class OTGPacket:
    """Represents one serialised measurement packet."""

    mode: str
    active_range: int
    measured_value: float
    error_pct: float
    timestamp_ms: int

    # ── Encoding ───────────────────────────────────────────────────────────────

    def encode(self) -> bytes:
        """
        Encode the packet into a fixed 32-byte OTG serial frame.

        Frame format (all ASCII / big-endian where applicable):
          Byte  0       : SOF (0x02)
          Byte  1       : Mode code  ('R', 'C', or 'L')
          Byte  2       : Range digit ('1'-'5')
          Bytes 3-12    : Value field – scientific notation, 10 chars
          Bytes 13-15   : Unit string (3 chars, space-padded)
          Bytes 16-20   : Error % – 5 chars, fixed decimal "XX.XX"
          Bytes 21-28   : Timestamp ms – 8 digit decimal
          Bytes 29-30   : XOR checksum of bytes 1-28 as 2 hex chars
          Byte  31      : EOF (0x03)
        """
        mode_b = self.mode.encode("ascii")
        range_b = str(self.active_range).encode("ascii")
        value_b = f"{self.measured_value:>10.4e}".encode("ascii")
        unit_b = UNIT_MAP.get(self.mode, "???").encode("ascii")
        error_b = f"{self.error_pct:>5.2f}".encode("ascii")
        ts_b = f"{self.timestamp_ms:>08d}".encode("ascii")

        payload = mode_b + range_b + value_b + unit_b + error_b + ts_b
        checksum = 0
        for byte in payload:
            checksum ^= byte
        checksum_b = f"{checksum:02X}".encode("ascii")

        return SOF + payload + checksum_b + EOF

    @classmethod
    def decode(cls, raw: bytes) -> "OTGPacket":
        """Decode a 32-byte OTG frame back into an OTGPacket."""
        if raw[0:1] != SOF or raw[-1:] != EOF:
            raise ValueError("Invalid frame markers")
        payload = raw[1:29]
        mode = payload[0:1].decode("ascii")
        active_range = int(payload[1:2])
        measured_value = float(payload[2:12].strip())
        error_pct = float(payload[15:20].strip())
        timestamp_ms = int(payload[20:28].strip())
        return cls(
            mode=mode,
            active_range=active_range,
            measured_value=measured_value,
            error_pct=error_pct,
            timestamp_ms=timestamp_ms,
        )

    def __str__(self) -> str:
        return (
            f"[OTG] {MODE_NAMES.get(self.mode, '?')} | "
            f"Range {self.active_range} | "
            f"{self.measured_value:.6g} {UNIT_MAP.get(self.mode, '?').strip()} | "
            f"Err {self.error_pct:.2f}% | "
            f"T+{self.timestamp_ms}ms"
        )


def format_packet(
    mode: str,
    active_range: int,
    measured_value: float,
    error_pct: float,
    timestamp_ms: int | None = None,
) -> OTGPacket:
    """
    Convenience factory: create + return an OTGPacket from settled readings.

    Args:
        mode          : 'R', 'C', or 'L'
        active_range  : integer 1-5
        measured_value: the settled measurement
        error_pct     : absolute % error
        timestamp_ms  : milliseconds since epoch (defaults to current time)

    Returns:
        OTGPacket ready to be encoded/transmitted.
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000) % 100_000_000  # 8-digit rollover
    return OTGPacket(
        mode=mode,
        active_range=active_range,
        measured_value=measured_value,
        error_pct=error_pct,
        timestamp_ms=timestamp_ms,
    )
