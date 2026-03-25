def calculator(x, y, operator):
    try:
        x = float(x)
        y = float(y)
        
        if operator == '+':
            result = x + y
        elif operator == '-':
            result = x - y
        elif operator == '*':
            result = x * y
        elif operator == '/':
            result = x / y
        else:
            print("Wrong operator")
            return
            
    except ZeroDivisionError:
        print("Just dont do it bro")
    except ValueError:
        print("Check ur number mate ")
    else:
        print(f"Result: {result}")
    finally:
        print("Execution attempt complete")


x = input("first number: ")
y = input("second number: ")
operator = input("Enter operator (+, -, *, /): ").strip()

calculator(x, y, operator)