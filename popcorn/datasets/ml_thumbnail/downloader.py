import os
import zipfile
import requests
from popcorn.datasets.ml_thumbnail.utils import RAW_DATA_URL, isValidPart


def downloadMovieLensThumbnailImages(partId: str, downloadPath: str):
    """
    Downloads the MovieLens thumbnails dataset from the given URL and saves it to the given path.
    [Note]: This is mainly for the .zip files containing the thumbnails images. For reading embedding,
    no downloading is required.

    Parameters
    ----------
    partId: str
        The part ID to download
    downloadPath: str
        The download path

    Returns
    -------
    status: bool
        The status of the download
    """
    # Argument validation
    if not isValidPart(partId):
        print(f"- [Error] Invalid part ID: {partId}")
        return False
    # Variables
    url = RAW_DATA_URL.format(part_id=partId)
    print(
        f"- Downloading part '{partId}' MovieLens-25M thumbnails (raw images) from '{url}' ..."
    )
    # Create the download path if it does not exist
    downloadPath = os.path.normpath(downloadPath)
    if not os.path.exists(downloadPath):
        print(f"- Creating the download path '{downloadPath}' ...")
        os.makedirs(downloadPath)
    # Prepare the file name
    folderName = f"thumbnails_ml25m_part{partId}.zip"
    folderZip = os.path.join(downloadPath, folderName)
    if os.path.exists(folderZip):
        print(f"- The file '{folderZip}' already exists! Skipping the download ...")
        return True
    # Fetch the dataset
    try:
        # Download the dataset
        print(f"- Fetching data from '{url}' ...")
        response = requests.get(url)
        response.raise_for_status()
        # Save the downloaded file
        with open(folderZip, "wb") as file:
            file.write(response.content)
        # Inform the user
        print("- Download completed and the folder is saved as a 'zip' file!")
        # Create a folder for the extracted files
        extractedFolder = os.path.join(downloadPath, f"thumbnails_ml25m_part{partId}")
        os.makedirs(extractedFolder, exist_ok=True)
        # Extract the dataset
        print(f"- Extracting the dataset files inside '{extractedFolder}' ...")
        with zipfile.ZipFile(folderZip, "r") as zipRef:
            zipRef.extractall(extractedFolder)
        print(f"- Dataset extracted to '{extractedFolder}' successfully!")
        # Remove the zip file after extraction
        print(f"- Removing the zip file {folderZip} ...")
        os.remove(folderZip)
        print(f"- Zip file {folderZip} removed successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"- [Error] Error fetching data from {url}: {e}\n")
        return False
