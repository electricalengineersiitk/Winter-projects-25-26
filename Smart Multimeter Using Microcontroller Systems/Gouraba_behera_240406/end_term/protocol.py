import time

def create_packet(mode, value, error, range_id):
    return {
        "mode": mode,          
        "value": value,        
        "error": error,        
        "range": range_id,     
        "time": time.time()    
    }



def encode_packet(packet):
    return f"{packet['mode']},{packet['value']:.6f},{packet['error']:.2f},{packet['range']},{packet['time']}"


def decode_packet(data):
    parts = data.split(",")

    return {
        "mode": parts[0],
        "value": float(parts[1]),
        "error": float(parts[2]),
        "range": int(parts[3]),
        "time": float(parts[4])
    }


# Test 
if __name__ == "__main__":
    pkt = create_packet("R", 1000, 0.5, 3)

    encoded = encode_packet(pkt)
    print("Encoded:", encoded)

    decoded = decode_packet(encoded)
    print("Decoded:", decoded)