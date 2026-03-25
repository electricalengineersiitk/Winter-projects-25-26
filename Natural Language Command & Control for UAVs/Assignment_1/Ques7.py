class Employee:
    def __init__(self, first, last, salary):
        self.first = first
        self.last = last
        self._salary = salary

    @property
    def email(self):
        if self.first is None or self.last is None:
            return None
        return f"{self.first}.{self.last}@company.com"

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, value):
        
        if value < 0:
            raise ValueError("Salary cannot be negative.")
        self._salary = value

    @property
    def fullname(self):
        return f"{self.first} {self.last}"

    @fullname.deleter
    def fullname(self):
        
        self.first = None
        self.last = None