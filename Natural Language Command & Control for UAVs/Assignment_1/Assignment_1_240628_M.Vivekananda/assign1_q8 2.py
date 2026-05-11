class timedu:
    def __init__(self,hours=0,minutes=0):
        total_minutes=hours*60+minutes
        self.hours=total_minutes//60
        self.minutes=total_minutes % 60

    def __add__(self,other):
        total_hours=self.hours+other.hours
        total_minutes=self.minutes+other.minutes
        return timedu(hours=total_hours,minutes=total_minutes)
    
    def __str__(self):
        return f"{self.hours}hrs:{self.minutes}mins"
    

t1=timedu(hours=2,minutes=70)
print(f"t1:{t1}")

t2 = timedu(hours=2, minutes=45)
t3 = timedu(hours=1, minutes=30)

t4=t2+t3
print(f"t4 : {t4}")