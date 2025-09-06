import os
import zipfile
import requests
from popcorn.datasets.movielens.helpers import ML1M_URL, ML25M_URL, ML100K_URL

def getMovieLensURL(version: str) -> str:
    """
    Get the download URL for the specified MovieLens dataset version.
    
    Parameters
    ----------
    version: str
        The version of the MovieLens dataset ('100k', '1m', or '25m')
    
    Returns
    -------
    url: str
        The download URL for the specified version
    """
    if version == "100k":
        return ML100K_URL
    elif version == "1m":
        return ML1M_URL
    elif version == "25m":
        return ML25M_URL
    else:
        print(f"- Error: Invalid MovieLens version '{version}'. Choose from '100k', '1m', or '25m'.")
        return None

def downloadMovieLens(version: str, downlpadPath: str):
    """
    Downloads the MovieLens 25M dataset

    Parameters
    ----------
    version: str
        The version of the MovieLens dataset to download ('100k', '1m', or '25m')
    downlpadPath: str
        The path to download the dataset to
    
    Returns
    -------
    status: bool
        The status of the download
    """
    print(f"\n- Downloading the MovieLens-{version} dataset ...")
    # If the donwload path does not exist, create it and download the dataset
    downlpadPath = os.path.normpath(downlpadPath)
    downlpadPath = os.path.join(downlpadPath, f'ml-{version}')
    if not os.path.exists(downlpadPath):
        print(f"- Creating the download path '{downlpadPath}' ...")
        os.makedirs(downlpadPath)
    else:
        print(f"- The download path '{downlpadPath}' already exists! Skipping the download ...")
        return True
    # Fetch the dataset
    try:
        # Get the download URL
        url = getMovieLensURL(version)
        if url is None:
            return False
        # Download the dataset
        print(f"- Fetching data from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        # Save the downloaded file
        datasetZip = os.path.join(downlpadPath, f'ml-{version}.zip')
        with open(datasetZip, 'wb') as file:
            file.write(response.content)
        # Inform the user
        print("- Download completed and the dataset is saved as a 'zip' file!")
        # Extract the dataset
        print(f"- Now, extracting the dataset files inside {downlpadPath} ...")
        with zipfile.ZipFile(datasetZip, 'r') as zipRef:
            zipRef.extractall(downlpadPath)
        print(f"- Dataset extracted to '{downlpadPath}' successfully!")
        # Remove the zip file after extraction
        print(f"- Removing the zip file {datasetZip} ...")
        os.remove(datasetZip)
        print("- Zip file removed successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"- Error fetching data from {url}: {e}\n")
        return False