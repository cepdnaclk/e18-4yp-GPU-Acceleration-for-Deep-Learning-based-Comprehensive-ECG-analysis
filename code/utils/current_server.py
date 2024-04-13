import socket
import platform

server_hostnames = ["ampere", "turing", "aiken", "kepler"]


def get_current_hostname():
    return socket.gethostname()


def is_running_in_server():
    hostname = get_current_hostname()

    if hostname in server_hostnames:
        return True
    else:
        return False


def is_running_linux():
    return platform.system() == "Linux"
