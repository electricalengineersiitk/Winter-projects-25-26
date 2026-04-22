class AutoRanger:
    def __init__(self, mode="R"):
        #initial setup
        self.current_range = 1
        self.mode = mode
        
        self.highs = 0
        self.lows = 0
        
        #setting the limits
        self.limits = {
            "R": {1: 1e2, 2: 1e3, 3: 1e4, 4: 1e5, 5: 1e6},
            "C": {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4},
            "L": {1: 1e-5, 2: 1e-4, 3: 1e-3, 4: 1e-2, 5: 1e-1}
        }

    def process_reading(self, val):
        #finding overload
        top = self.limits[self.mode][5]
        if val > top * 1.1:
            return "OL"
            
        #set marks
        max_val = self.limits[self.mode][self.current_range]
        up_mark = 0.90 * max_val
        down_mark = 0.10 * max_val

        #step up
        if val > up_mark and self.current_range < 5:
            self.highs += 1
            self.lows = 0
            
            #switch up
            if self.highs >= 3:
                self.current_range += 1
                self.highs = 0
                
        #step down
        elif val < down_mark and self.current_range > 1:
            self.lows += 1
            self.highs = 0
            
            #switch down
            if self.lows >= 3:
                self.current_range -= 1
                self.lows = 0
                
        #stay put
        else:
            self.highs = 0
            self.lows = 0

        return self.current_range


if __name__ == "__main__":
    #initializing
    ranger = AutoRanger(mode="R")
    
    #test values
    tests = [
        15, 15, 15,
        95, 95, 95,
        400, 400, 400,
        9500, 9500, 9500,
        1000000, 1000000, 1000000,
        5000000000, 50000, 50000
    ]
    
    # print headers
    print(f"{'val':<15} | {'scale':<15} | {'highs':<15} | {'lows'}")
    print("-" * 65)
    
    # run loop
    for v in tests:
        now = ranger.process_reading(v)
        print(f"{v:<15} | {now:<15} | {ranger.highs:<15} | {ranger.lows}")