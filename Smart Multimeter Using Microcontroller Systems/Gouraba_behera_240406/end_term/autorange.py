STEP_UP = 0.9      # >90% → go up
STEP_DOWN = 0.1    # <10% → go down
HYSTERESIS = 3     # wait 3 times before switching


class AutoRanger:
    def __init__(self, ranges):
        self.ranges = ranges         
        self.current_range = 0
        self.up_count = 0
        self.down_count = 0

    def update(self, value):
        max_val = self.ranges[self.current_range]
        if value > self.ranges[-1]:
            return self.result("OL", None)

        if value > STEP_UP * max_val:
            self.up_count += 1
            self.down_count = 0

            if self.up_count >= HYSTERESIS:
                if self.current_range < len(self.ranges) - 1:
                    self.current_range += 1
                self.up_count = 0

            return self.result("RANGING_UP", value)


        elif value < STEP_DOWN * max_val and self.current_range > 0:
            self.down_count += 1
            self.up_count = 0

            if self.down_count >= HYSTERESIS:
                self.current_range -= 1
                self.down_count = 0

            return self.result("RANGING_DOWN", value)

        # STABLE
        else:
            self.up_count = 0
            self.down_count = 0
            return self.result("SETTLED", value)

    def result(self, status, value):
        return {
            "range": self.current_range + 1,           
            "max": self.ranges[self.current_range],   
            "status": status,
            "value": value
        }



# TEST

if __name__ == "__main__":
    import numpy as np

    # Example: resistance ranges
    ranges_R = [100, 1e3, 10e3, 100e3, 1e6]
    ranger = AutoRanger(ranges_R)

    values = np.logspace(2, 6, 20)

    for v in values:
        res = ranger.update(v)
        print(f"Value={v:.1f} | Range={res['range']} | {res['status']}")