# === PART1 : THEORATICAL QUESTIONS ===
# QUESTION 1:
# Consider the following function definition:
def add_item(item, box=[]):
    box.append(item)
    return box


print(add_item("apple"))  # The output will be: ['apple']
print(add_item("banana"))  # The output will be: ['apple', 'banana']


# Answer(a & b):
# the box list persist its data between function call even when we are not providing the second argument, because the default value is a mutable object (a list in this case)
# a mutable means that its content can be changed after its creation.
# this is why when we call the function the second time without providing the second argument, it uses the same list that was modified in the first call.
# Answer(c):
# Alternative header of the above function
def add_item_1(item, box=None):
    if box is None:
        box = []
    box.append(item)
    return box


print(add_item_1("apple"))  # output: ['apple']
print(add_item_1("banana"))  # output: ['banana']


# QUESTION 2:
# __str__ and __repr__

# Answer: __str__ is intended towards user readability and is used for creating a string representation of an object that is easy to read and understand.
# __repr__ is intended towards developer readability and is used for creating a string representation of an object that  is more detailed and unabigious and can be used for debugging and further development.


# __repr__ will be used as a fallback if the other one is missing because it covers both cases and more useful for debugging purposes.
# Example:
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"


p = Point(1, 2)
print(str(p))  # Output: Point(1, 2)
print(repr(p))  # Output: Point(x=1, y=2)


# QUESTION 3:
# === Class variables vs instance variables ===
# Answer(a)):
#  class variable is stored once in the class while the instance variable is stored individually for each instance of the class.
class Dog:
    species = "Canis familiaris"  # class variable

    def __init__(self, name, age):
        self.name = name  # instance variable
        self.age = age  # instance variable


dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog2.species)
print(dog1.name)
print(dog1.age)
print(dog2.name)
print(dog2.age)


# Answer(b)):
# if we change the class variable via the class itself, it will affect all instances.
Dog.species = "New species"
print(dog1.species)  # Output: New species
print(dog2.species)  # Output: New species


# but if we change the class variable using an instance, it will create a new instance variable for that instance only. example:
dog1.species = "NEW SPECIES"
print(dog1.species)  # Output: NEW SPECIES
print(dog2.species)  # Output: New species


# === PART 2 : PROGRAMMING CHALLENGES ===
# QUESTION 4:

server_logs = {"User1: Login; User2: Login; User1: Logout; User3: Login; User2: Logout"}


def current_status_of_users(logs):
    status = {}
    for log in logs:
        entries = log.split("; ")
        for entry in entries:
            user_action = entry.split(": ")
            user = user_action[0]
            action = user_action[1]
            if action == "Login":
                status[user] = "Online"
            elif action == "Logout":
                status[user] = "Offline"

    return status


print(current_status_of_users(server_logs))


# Output: {'User1': 'Offline', 'User2': 'Offline', 'User3': 'Online'}

# QUESTION 5:


def calculator(a, b, c):
    if (
        type(a) not in [int, float]
        or type(b) not in [int, float]
        or type(c) not in ["+", "-", "*", "/"]
    ):
        return "Invalid input"
    if c == "+":
        return a + b
    elif c == "-":
        return a - b
    elif c == "*":
        return a * b
    elif c == "/":
        if b == 0:
            return "can't divide by zero"
        return a / b
    else:
        return "Invalid input"


a = input("Enter first number: ")
b = input("Enter second number")
c = input("Enter operation :(+, -, *, /) ")
print(calculator(a, b, c))


# Example inputs and outputs:
# Input: a = 10, b = 5, c = '+'
# Output: 15


# === Advanced OOPs challenge ===
# QUESTION 6:
class Book:
    def __init__(self, title, author, is_checked_out=False):
        self.title = title
        self.author = author
        self.is_checked_out = is_checked_out


class Library:

    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)
        return f"'{book.title}' by {book.author} added to library"

    def check_out_book(self, title):
        for book in self.books:
            if book.title == title:
                if not book.is_checked_out:
                    book.is_checked_out = True
                    return f"You have checked out '{book.title}'"
                else:
                    return f"'{book.title}' is already checked out"
        return f"'{title}' not found in the library"

    def return_book(self, title):
        for book in self.books:
            if book.title == title:
                if book.is_checked_out:
                    book.is_checked_out = False
                    return f"You have returned '{book.title}'"
                else:
                    return f"'{book.title}' was not checked out"
        return f"'{title}' not found in the library"






# === Question 7 ===

class Employee:
    def __init__(self, first, last, salary):
        self.first = first
        self.last = last
        self.salary = salary
        @property
        def email(self):
            return f"{self.first.lower()}_{self.last.lower()}@company.com"
        @property
        def full_name(self):
            return f"{self.first} {self.last}"
        @full_name.deleter 
        def full_name(self):
            print("Deleting Name...")
            self.first = None
            self.last = None
        
        @property
        def salary(self):
            return self.salary
        @salary.setter
        def salary(self, value):
            if value < 0:
                raise ValueError("Salary cannot be negative")
            self.salary = value

#  Question 8: === Operator Overloading ===

class TimeDuration:
    def __init__(self, hours, minutes):
        self.hours = hours
        self.minutes = minutes
        if self.minutes > 60:
            self.hours += self.minutes // 60
            self.minutes = self.minutes % 60
    def __add__(self, other):
        total_hours = self.hours + other.hours
        total_minutes = self.minutes + other.minutes
        if total_minutes > 60:
            total_hours += total_minutes // 60
            total_minutes =  total_minutes % 60
            return TimeDuration(total_hours, total_minutes)
    def __str__(self):
        return f"{self.hours}H:{self.minutes}M"
# Example usage:
duration1 = TimeDuration(2, 45)
duration2 = TimeDuration(1, 30)
total_duration = duration1 + duration2
print(total_duration)  # Output: 4H:15M




