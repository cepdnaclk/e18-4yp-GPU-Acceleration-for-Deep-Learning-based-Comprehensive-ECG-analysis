import os


def download_and_extract_deepfake_dataset_to_ram():
    # Change the current working directory to /dev/shm
    os.chdir("/dev/shm")

    # Download the file
    print("Downloading the dataset... This will only happen once every server restart")
    os.system("wget https://files.osf.io/v1/resources/6hved/providers/dropbox/filtered_all_normals_121977.tar.gz")
    print("Downloaded the dataset")

    # Extract
    print("Extracting the dataset...")
    os.system("tar -xzf filtered_all_normals_121977.tar.gz")
    print("Extracted the dataset")

    # Remove the archive file
    print("Removing the archive file...")
    os.system("rm filtered_all_normals_121977.tar.gz")
    print("Removed the archive file")

    # Download csv file
    print("Downloading the csv file... This will only happen once every server restart")
    os.system("wget https://files.osf.io/v1/resources/6hved/providers/dropbox/filtered_all_normals_121977_ground_truth.csv")
    print("Downloaded the csv file")

    # Reset the current working directory to the original directory
    original_dir = os.getcwd()
    os.chdir(original_dir)


def download_and_extract_ptb_xl_dataset_to_ram():
    # Change the current working directory to /dev/shm
    os.chdir("/dev/shm")

    # Download the file
    print("Downloading the dataset... This will only happen once every server restart")
    os.system("wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Downloaded the dataset")

    # Extract
    print("Extracting the dataset...")
    os.system("unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Extracted the dataset")

    # Remove the archive file
    print("Removing the archive file...")
    os.system("rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Removed the archive file")

    # Reset the current working directory to the original directory
    original_dir = os.getcwd()
    os.chdir(original_dir)


def download_and_extract_ptb_xl_dataset_to_ram():
    # Change the current working directory to /dev/shm
    os.chdir("/dev/shm")

    # Download the file
    print("Downloading the dataset... This will only happen once every server restart")
    os.system("wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Downloaded the dataset")

    # Extract
    print("Extracting the dataset...")
    os.system("unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Extracted the dataset")

    # Remove the archive file
    print("Removing the archive file...")
    os.system("rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    print("Removed the archive file")

    # Reset the current working directory to the original directory
    original_dir = os.getcwd()
    os.chdir(original_dir)


def download_and_extract_ptb_xl_plus_dataset_to_ram():
    # Change the current working directory to /dev/shm
    os.chdir("/dev/shm")

    # Download the file
    print("Downloading the dataset... This will only happen once every server restart")
    os.system("wget https://physionet.org/static/published-projects/ptb-xl-plus/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip")
    print("Downloaded the dataset")

    # Extract
    print("Extracting the dataset...")
    os.system("unzip ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip")
    print("Extracted the dataset")

    # Remove the archive file
    print("Removing the archive file...")
    os.system("rm ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip")
    print("Removed the archive file")

    # Reset the current working directory to the original directory
    original_dir = os.getcwd()
    os.chdir(original_dir)
