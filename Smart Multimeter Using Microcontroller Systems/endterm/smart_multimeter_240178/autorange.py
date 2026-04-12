class AutoRange:
    def __init__(self, ranges):
        self.ranges = ranges
        self.current_range = 0
        self.up_count = 0
        self.down_count = 0
        self.HYSTERESIS = 3

    def update(self, value):
        max_val = self.ranges[self.current_range]

        # Step UP
        if value > 0.9 * max_val:
            self.up_count += 1
            self.down_count = 0

            if self.up_count >= self.HYSTERESIS:
                if self.current_range < len(self.ranges) - 1:
                    self.current_range += 1
                self.up_count = 0

        # Step DOWN
        elif value < 0.1 * max_val:
            self.down_count += 1
            self.up_count = 0

            if self.down_count >= self.HYSTERESIS:
                if self.current_range > 0:
                    self.current_range -= 1
                self.down_count = 0

        else:
            self.up_count = 0
            self.down_count = 0

        return self.current_range