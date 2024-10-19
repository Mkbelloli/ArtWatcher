
log_flag = False

def activate_logs():
    """
    activate logs
    :return:
    """
    global log_flag
    log_flag = True


def deactivate_logs():
    """
    deactivate logs
    :return:
    """
    global log_flag
    log_flag = False

def print_log(msg):
    """
    print message according activation log
    :param msg: message to be printed
    :return:
    """
    global log_flag
    if log_flag:
        print(msg)

