class AutoRanger:
    def __init__(self, mode="R"):
        self.mode = mode
        self.current_range = 1  # start at the first range
        
        # counters for the 3-sample rule
        self.high_count = 0
        self.low_count = 0
        
        # limits for each component type
        self.limits = {
            "R": {1: 100, 2: 1000, 3: 10000, 4: 100000, 5: 1000000},      # Ohms
            "C": {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4},          # Farads
            "L": {1: 1e-5, 2: 1e-4, 3: 1e-3, 4: 1e-2, 5: 1e-1}           # Henries
        }

    def determine_range(self, reading):
        # get max limit for current range
        current_max = self.limits[self.mode][self.current_range]
        
        # 90% and 10% marks for switching
        up_mark = 0.90 * current_max
        down_mark = 0.10 * current_max

        # check if value is too high for this range
        if reading > up_mark and self.current_range < 5:
            self.high_count += 1
            self.low_count = 0  # reset the low counter
            
            # switch up if it stays high for 3 times
            if self.high_count >= 3:
                self.current_range += 1
                self.high_count = 0  
                
        # check if value is too low
        elif reading < down_mark and self.current_range > 1:
            self.low_count += 1
            self.high_count = 0
            
            # switch down if it stays low for 3 times
            if self.low_count >= 3:
                self.current_range -= 1
                self.low_count = 0
                
        # reset counts if value is in the middle
        else:
            self.high_count = 0
            self.low_count = 0

        return self.current_range