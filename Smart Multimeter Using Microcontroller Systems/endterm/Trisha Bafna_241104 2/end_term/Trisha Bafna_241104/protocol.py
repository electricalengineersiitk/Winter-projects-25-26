import json
import time

def format_otg_packet(mode, value, unit, active_range):
    packet = {
        "timestamp": time.time(),
        "mode": mode,
        "reading": value,
        "unit": unit,
        "range_state": active_range
    }
    return json.dumps(packet).encode('utf-8')