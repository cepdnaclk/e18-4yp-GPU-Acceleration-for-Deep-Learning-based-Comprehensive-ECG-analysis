import subprocess
from .current_server import is_running_in_server


def get_latest_update_by():
    if not is_running_in_server():
        return "--Not running in servers--"

    try:
        with open("last_updated_by.txt", "r") as f:
            return f.read().strip()
    except:
        return "Not found"
