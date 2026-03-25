class TimeDuration:
    def __init__(self, hours, minutes):
        
        self.hours = hours + (minutes // 60)
        self.minutes = minutes % 60

    def __add__(self, other):
       
        total_hours = self.hours + other.hours
        total_minutes = self.minutes + other.minutes
        return TimeDuration(total_hours, total_minutes)

    def __str__(self):
       
        return f"{self.hours}h:{self.minutes}m"


t1 = TimeDuration(2, 45)
t2 = TimeDuration(1, 30)
t3 = t1 + t2
print(t3) # Expected Output: 4h:15m