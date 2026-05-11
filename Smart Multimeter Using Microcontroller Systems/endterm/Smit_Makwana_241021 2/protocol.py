import json
import time

def format_otg_packet(measurement_type, value, unit, range_idx, status):
    """Formats the data into a JSON packet for OTG serial transmission."""
    packet = {
        "timestamp": time.time(),
        "type": measurement_type,
        "value": round(value, 4) if isinstance(value, float) else value,
        "unit": unit,
        "active_range": range_idx,
        "status": status
    }
    return json.dumps(packet)