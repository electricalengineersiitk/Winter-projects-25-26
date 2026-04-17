class AutoRanger:
    def __init__(self):
        self.current_range = 1
        self.trigger_count = 0
        self.last_trigger_direction = None
        
        self.ranges = {
            1: {'max': 100, 'up': 90, 'down': 0},
            2: {'max': 1000, 'up': 900, 'down': 100},
            3: {'max': 10000, 'up': 9000, 'down': 1000},
            4: {'max': 100000, 'up': 90000, 'down': 10000},
            5: {'max': 1000000, 'up': float('inf'), 'down': 100000}
        }

    def process_reading(self, reading, mode_scale=1.0):
        """Processes a reading and returns (settled_value, active_range, status)"""
        up_thresh = self.ranges[self.current_range]['up'] * mode_scale
        down_thresh = self.ranges[self.current_range]['down'] * mode_scale
        max_val = self.ranges[self.current_range]['max'] * mode_scale

        if reading > self.ranges[5]['max'] * mode_scale:
            return reading, self.current_range, "OL"

        if reading > up_thresh and self.current_range < 5:
            self._handle_hysteresis(1)
            return reading, self.current_range, "STEPPING_UP"
        elif reading < down_thresh and self.current_range > 1:
            self._handle_hysteresis(-1)
            return reading, self.current_range, "STEPPING_DOWN"
        else:
            self.trigger_count = 0 # Reset if settled
            return reading, self.current_range, "SETTLED"

    def _handle_hysteresis(self, direction):
        if self.last_trigger_direction == direction:
            self.trigger_count += 1
        else:
            self.trigger_count = 1
            self.last_trigger_direction = direction

        if self.trigger_count >= 3:
            self.current_range += direction
            self.trigger_count = 0