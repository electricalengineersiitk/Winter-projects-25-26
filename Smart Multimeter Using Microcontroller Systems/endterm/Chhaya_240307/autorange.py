class AutoRanger:
    def __init__(self):
        # We start at Range 1
        self.current_range = 1
        
        # Counters for hysteresis (require 3 consecutive triggers)
        self.step_up_count = 0
        self.step_down_count = 0
        
        # Define the maximum values for the 5 ranges (works for R, C, and L)
        self.range_max = {
            1: 100.0,       # 100 Ohms / 10 nF / 10 uH
            2: 1000.0,      # 1 kOhm / 100 nF / 100 uH
            3: 10000.0,     # 10 kOhm / 1 uF / 1 mH
            4: 100000.0,    # 100 kOhm / 10 uF / 10 mH
            5: 1000000.0    # 1 MOhm / 100 uF / 100 mH
        }

    def process_reading(self, reading):
        """
        Takes a raw reading, checks thresholds, applies hysteresis, 
        and switches range if necessary.
        """
        # Safety fallback: Exceeds absolute maximum
        if reading > self.range_max[5]:
            return "OL", self.current_range

        current_max = self.range_max[self.current_range]
        
        # Step UP logic: Reading > 90% of current range max
        if reading > 0.9 * current_max and self.current_range < 5:
            self.step_up_count += 1
            self.step_down_count = 0  # Reset the other counter
            
            # Apply 3-sample hysteresis rule
            if self.step_up_count >= 3:
                self.current_range += 1
                self.step_up_count = 0  # Reset after switching
                
        # Step DOWN logic: Reading < 10% of current range max
        elif reading < 0.1 * current_max and self.current_range > 1:
            self.step_down_count += 1
            self.step_up_count = 0    # Reset the other counter
            
            # Apply 3-sample hysteresis rule
            if self.step_down_count >= 3:
                self.current_range -= 1
                self.step_down_count = 0  # Reset after switching
                
        # SETTLED: Reading is between 10% and 90%
        else:
            # Reset both counters since it's a stable reading
            self.step_up_count = 0
            self.step_down_count = 0
            
        return reading, self.current_range

# --- PHASE 2 VERIFICATION ---
if __name__ == "__main__":
    ranger = AutoRanger()
    
    print("--- Testing Auto-Ranger ---")
    # Simulate a sweep of readings
    test_readings = [
        50.0,    # Settled in Range 1
        95.0,    # >90%, Trigger 1
        96.0,    # >90%, Trigger 2
        98.0,    # >90%, Trigger 3 -> Should jump to Range 2
        150.0,   # Settled in Range 2
        1200000  # Overload
    ]
    
    for val in test_readings:
        reported_val, active_range = ranger.process_reading(val)
        print(f"Reading: {val:<8} | Reported: {reported_val:<8} | Active Range: {active_range}")