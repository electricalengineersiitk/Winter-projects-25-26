def server_logs(log_string):
    user_states = {}
    logs = log_string.split('; ')
    for entry in logs:
        entry=entry.strip()
        if not entry:
            continue
        try:
            user, action = entry.split(': ')
            user = user.strip()
            action = action.strip()
        except ValueError:
            continue
        
        if action == 'Login':
                user_states[user] = 'Online'
        elif action == 'Logout':
                user_states[user] = 'Offline'
    return user_states


log_data_input = input("Strings: ")
final_status = server_logs(log_data_input)
print(final_status)