import os
import zipfile
import requests
from popcorn.datasets.mmtf14k.utils import BASE_URL

def downloadMMTF14k(downlpadPath: str):
    """
    Downloads the MMTF14k dataset from the given URL and saves it to the given path

    Parameters
    ----------
    downlpadPath: str
        The download path
    
    Returns
    -------
    status: bool
        The status of the download
    """
    print(f"- Downloading the MMTF-14K dataset from '{BASE_URL}' ...")
    # Create the download path if it does not exist
    downlpadPath = os.path.normpath(downlpadPath)
    if not os.path.exists(downlpadPath):
        print(f"- Creating the download path '{downlpadPath}' ...")
        os.makedirs(downlpadPath)
    else:
        print(
            f"- The download path '{downlpadPath}' already exists! Skipping the download ..."
        )
        return True
    # Fetch the dataset
    try:
        # Download the dataset
        print(f"- Fetching data from '{BASE_URL}' ...")
        response = requests.get(BASE_URL)
        response.raise_for_status()
        # Save the downloaded file
        datasetZip = os.path.join(downlpadPath, 'mmtf14k.zip')
        with open(datasetZip, 'wb') as file:
            file.write(response.content)
        # Inform the user
        print("- Download completed and the dataset is saved as a 'zip' file!")
        # Extract the dataset
        print(f"- Extracting the dataset files inside '{downlpadPath}' ...")
        with zipfile.ZipFile(datasetZip, 'r') as zipRef:
            zipRef.extractall(downlpadPath)
        print(f"- Dataset extracted to '{downlpadPath}' successfully!")
        # Remove the zip file after extraction
        print(f"- Removing the zip file {datasetZip} ...")
        os.remove(datasetZip)
        print("- Zip file removed successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"- [Error] Error fetching data from {BASE_URL}: {e}\n")
        return False
