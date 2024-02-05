import os
import gdown
from config import SAVED_MODEL_PATH


def download_file_if_empty(path, google_drive_link, output_filename):
    """
    Download a file from Google Drive to the specified path if the path is empty.

    Parameters:
    - path: Path to the directory.
    - google_drive_link: Google Drive link for the file.
    - output_filename: Name to save the downloaded file.
    """
    # Check if the directory is empty
    if not os.listdir(path):
        print(f"The directory {path} is empty. Downloading file...")

        # Extract file ID from the Google Drive link
        file_id = google_drive_link.split("/")[-2]

        # Construct the direct download link
        direct_download_link = f"https://drive.google.com/uc?id={file_id}"

        # Specify the output file path
        output_file_path = os.path.join(path, output_filename)

        # Download the file using gdown
        gdown.download(direct_download_link, output_file_path, quiet=False)
        print(f"File downloaded and saved to {output_file_path}")
    else:
        print(f"The directory {path} is not empty.")
