import json
import time

def make_pack(mode, val, scale):
    #mapping the units
    units = {"R": "ohms", "C": "farads", "L": "henrys"}
    
    #build dict
    data = {
        "time": int(time.time() * 1000),
        "mode": mode,
        "val": round(val, 4),
        "unit": units.get(mode, "none"),
        "scale": scale,
        "status": "OL" if scale == "OL" else "OK"
    }
    
    #dump string
    return json.dumps(data) + "\n"

if __name__ == "__main__":
    #test ok
    ok_pkt = make_pack("R", 9850.4, 3)
    print(ok_pkt)
    
    #test bad
    ol_pkt = make_pack("R", 9999999, "OL")
    print(ol_pkt)