class employee:
    def __init__(self,first,last,salary):
        self.first=first
        self.last=last
        self.salary_1=salary

    @property
    def email(self):
        if self.first is None or self.last is None:
            return None
        return f"{self.first}.{self.last}@company.com"
    
    @property
    def salary(self):
        return self.salary_1
    
    @salary.setter
    def salary(self,value):
        if value<0:
            raise ValueError("No negative")
        self.salary_1=value

    @property
    def fullname(self):
        if self.first is None or self.last is  None:
            return None
        return f"{self.first}{self.last}"
    
    @fullname.deleter
    def fullname(self):
        self.first=None
        self.last=None


emp_1=employee("Vivek","MANSOJU",333333333333)
print(emp_1.fullname)
print(emp_1.salary)


emp_1.salary = 60000
print(f"New Salary: {emp_1.salary}")

try:
    emp_1.salary = -5000
except ValueError as e:
    print(f"Error: {e}")

del emp_1.fullname
print(f"After deletion - Full Name: {emp_1.fullname}")
print(f"After deletion - Email: {emp_1.email}")