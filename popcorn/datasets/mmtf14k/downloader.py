import os
import zipfile
import requests
from popcorn.datasets.mmtf14k.utils import BASE_URL


def downloadMMTF14k(downloadPath: str):
    """
    Downloads the MMTF14k dataset (Base) from the given URL and saves it to the given path.
    [Note]: The MMTF14K Fused dataset can be directly loaded and does not require downloading.

    Parameters
    ----------
    downloadPath: str
        The download path

    Returns
    -------
    status: bool
        The status of the download
    """
    print(f"- Downloading the MMTF-14K dataset (Base) from '{BASE_URL}' ...")
    # Create the download path if it does not exist
    downloadPath = os.path.normpath(downloadPath)
    if not os.path.exists(downloadPath):
        print(f"- Creating the download path '{downloadPath}' ...")
        os.makedirs(downloadPath)
    else:
        print(
            f"- The download path '{downloadPath}' already exists! Skipping the download ..."
        )
        return True
    # Fetch the dataset
    try:
        # Download the dataset
        print(f"- Fetching data from '{BASE_URL}' ...")
        response = requests.get(BASE_URL)
        response.raise_for_status()
        # Save the downloaded file
        datasetZip = os.path.join(downloadPath, "mmtf14k.zip")
        with open(datasetZip, "wb") as file:
            file.write(response.content)
        # Inform the user
        print("- Download completed and the dataset is saved as a 'zip' file!")
        # Extract the dataset
        print(f"- Extracting the dataset files inside '{downloadPath}' ...")
        with zipfile.ZipFile(datasetZip, "r") as zipRef:
            zipRef.extractall(downloadPath)
        print(f"- Dataset extracted to '{downloadPath}' successfully!")
        # Remove the zip file after extraction
        print(f"- Removing the zip file {datasetZip} ...")
        os.remove(datasetZip)
        print("- Zip file removed successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"- [Error] Error fetching data from {BASE_URL}: {e}\n")
        return False
