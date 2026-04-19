import json  # standardized "contaiener" for data

def make_packet(value, unit, current_range):
    # put data into a dictionary first
    data_dict = {
        "value": round(value, 2),
        "unit": unit,
        "range": current_range,
        "status": "OK"
    }
    
    # turn the dictionary into a json string
    json_packet = json.dumps(data_dict)
    
    return json_packet

# testing the packets
if __name__ == "__main__":
    print("--- Multimeter JSON Protocol Test ---")
    
    # test 1: resistance
    packet_1 = make_packet(1250.556, "Ohms", 2)
    print(f"Packet 1: {packet_1}")
    
    # test 2: capacitance
    packet_2 = make_packet(0.0000047, "Farads", 3)
    print(f"Packet 2: {packet_2}")