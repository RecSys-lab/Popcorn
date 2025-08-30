import os
import zipfile
import requests

def downloadMMTF14k(url: str, downlpadPath: str):
    """
    Downloads the MMTF14k dataset from the given URL and saves it to the given path

    Parameters
    ----------
    url: str
        The URL of the dataset
    downlpadPath: str
        The download path
    
    Returns
    -------
    status: bool
        The status of the download
    """
    # Download the dataset
    print(f"- Downloading the dataset from '{url}' ...")
    # Create the download path if it does not exist
    if not os.path.exists(downlpadPath):
        os.makedirs(downlpadPath)
    # Fetch the dataset
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        # Save the downloaded file
        datasetZip = os.path.join(downlpadPath, 'ml-25m.zip')
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
