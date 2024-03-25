import subprocess
from .current_server import is_running_in_server


def get_latest_update_by():

    if not is_running_in_server():
        return "--Not running in servers--"

    # Run the "ls" command and capture its output
    process = subprocess.run(["ls", "-al"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Get the output as a string
    output_str = process.stdout

    # Count the occurrences of "e18098", "e18100", and "e18155" in the output string
    countIshan = output_str.count("e18098")
    countAdeepa = output_str.count("e18100")
    countRidma = output_str.count("e18155")

    # Find the maximum count and its corresponding variable name
    max_count = max(countIshan, countAdeepa, countRidma)
    max_variable = None

    if max_count == countIshan:
        max_variable = "Ishan"
    elif max_count == countAdeepa:
        max_variable = "Adeepa"
    elif max_count == countRidma:
        max_variable = "Ridma"

    return max_variable
