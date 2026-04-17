from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MeasurementPacket:
    mode: str
    value: float
    unit: str
    range_index: int
    error_percent: float
    status: str
    timestamp_ms: int

def encode_packet(packet: MeasurementPacket) -> bytes:
    payload = asdict(packet)
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")

def decode_packet(raw: bytes) -> MeasurementPacket:
    data = json.loads(raw.decode("utf-8").strip())
    return MeasurementPacket(
        mode=str(data["mode"]),
        value=float(data["value"]),
        unit=str(data["unit"]),
        range_index=int(data["range_index"]),
        error_percent=float(data["error_percent"]),
        status=str(data["status"]),
        timestamp_ms=int(data["timestamp_ms"]),
    )
