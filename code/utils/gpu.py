import torch
import utils.current_server as current_server
import subprocess

def get_device_with_lowest_vram():
    if torch.cuda.is_available():
        # if not running in linux, return cuda:0
        # because bash commands doesnt work in windows
        if not current_server.is_running_linux():
            print("Running in windows, Using cuda:0")
            return "cuda:0"

        # else try to find the GPU with the lowest vram usage
        try:
            # Call the shell script using subprocess
            result = subprocess.run(
                ["bash", "./find_least_used_gpu.sh"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Print the output of the shell script
            print("Least used GPU: " + result.stdout.strip())
            return "cuda:" + result.stdout.strip()

        except subprocess.CalledProcessError as e:
            # If the shell script returns a non-zero exit code, print an error message
            print(f"Error: {e}")
            print(f"Output: {e.output}")

    # Use CPU if no GPU is available or if the GPU count is 0
    print("No GPU found, using CPU")
    return torch.device("cpu")