def safe_calculator():
    print("Welcome to Safe Calculator (Type 'quit' to exit)")
    
    while True:
        try:
            val1 = input("Enter first number: ")
            if val1.lower() == 'quit': break
            num1 = float(val1)
            
            val2 = input("Enter second number: ")
            num2 = float(val2)
            
            op = input("Enter operator (+, -, *, /): ")
            
            result = 0
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2
            else:
                print("Invalid operator")
                continue
                
        except ValueError:
            
            print("Error: Please enter valid numbers.")
        except ZeroDivisionError:
            
            print("Error: Cannot divide by zero.")
        else:
        
            print(f"Result: {result}")
        finally:
            
            print("Execution attempt complete.\n")

# To run this, call safe_calculator()