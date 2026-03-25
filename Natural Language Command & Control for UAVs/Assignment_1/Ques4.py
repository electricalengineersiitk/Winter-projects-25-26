def parse_logs(log_string):
    
    user_status = {}
    
    actions = log_string.split('; ')
    
    for entry in actions:
    
        user, action = entry.split(': ')
        
        
        if action == "Login":
            user_status[user] = "Online"
        elif action == "Logout":
            user_status[user] = "Offline"
            
    return user_status


logs = "User1: Login; User2: Login; User1: Logout; User3: Login; User2: Logout"
print(parse_logs(logs))
# Expected: {'User1': 'Offline', 'User2': 'Offline', 'User3': 'Online'} [cite: 20]