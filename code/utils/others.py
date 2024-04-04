import subprocess
from .current_server import is_running_in_server


def get_latest_update_by():
    if not is_running_in_server():
        return "--Not running in servers--"

    # Run the "ls -alt" command and capture its output
    process = subprocess.run(["ls", "-alt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output_str = process.stdout

    # Split the output into lines
    lines = output_str.strip().split("\n")

    # Exclude the first line (summary) and the last line (current directory)
    lines = lines[2]

    # Loop through the lines and extract the username from the latest updated files
    parts = lines.split()
    username = parts[2]
    if username.startswith("e18"):
        return username

    return "Not found"


