import socket


def get_current_hostname():
    return socket.gethostname()


def is_running_in_server():
    hostname = get_current_hostname()

    server_hostnames = ["ampere", "turing", "aiken"]
    if hostname in server_hostnames:
        return True
    else:
        return False
