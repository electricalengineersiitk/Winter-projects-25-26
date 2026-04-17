import json
import time

def create_otg_packet(mode, measured_value, active_range, status="OK"):
    """
    Formats the multimeter reading into a standard OTG serial packet (JSON format).
    This simulates what the microcontroller would send over the USB OTG cable.
    """
    
    # Create a dictionary representing the data payload
    packet = {
        "timestamp": round(time.time(), 3),  # Time of reading
        "device_id": "SmartMulti_Sim_v1",    # Hardware identifier
        "mode": mode,                        # Resistance, Capacitance, or Inductance
        "reading": measured_value,           # The raw numeric value (or "OL")
        "range_index": active_range,         # Which scale (1-5) is active
        "status": status                     # "OK" or "OL" (Overload)
    }

    # Convert the dictionary to a JSON string for serial transmission
    serial_payload = json.dumps(packet)
    
    return serial_payload


# --- VERIFICATION ---
if __name__ == "__main__":
    print("--- Testing OTG Protocol Packets ---")
    
    # Simulate a normal resistance reading
    packet1 = create_otg_packet("Resistance", 104.9, 1, "OK")
    print(f"Normal Packet  -> {packet1}")
    
    # Simulate an overload capacitance reading
    packet2 = create_otg_packet("Capacitance", "OL", 5, "OL")
    print(f"Overload Packet-> {packet2}")