log_flag = False

def activate_logs():
    global log_flag
    log_flag = True


def deactivate_logs():
    global log_flag
    log_flag = False

def print_log(msg):
    global log_flag
    if log_flag:
        print(msg)