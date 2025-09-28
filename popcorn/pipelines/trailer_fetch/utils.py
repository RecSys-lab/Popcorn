import os
import requests

# Supported video formats
videoFormats = ["mp4", "avi", "mov", "mkv"]


def downloadVideoFile(url: str, downloadPath: str, fileName: str, format: str = "mp4"):
    """
    Downloads a video file from a given URL.

    Parameters
    ----------
    url: str
        The URL of the video file to download.
    downloadPath: str
        The path to download the video file to.
    fileName: str
        The name of the video file to save as.
    format: str
        The format of the video file (default is 'mp4').

    Returns
    -------
    success: bool
        True if the video file was downloaded successfully, False otherwise.
    """
    # Variables
    success = False
    # Check the format
    if format not in videoFormats:
        print(
            f"- [Warn] Unsupported video format '{format}'. Supported formats are: {videoFormats}. Using 'mp4' instead..."
        )
        format = "mp4"
    # Check the URL
    if url is None:
        print("- [Warn] URL is None. Cannot download the video file! Exiting ...")
        return success
    # Check the download path
    if not os.path.exists(downloadPath):
        print(
            f"- [Warn] Download path '{downloadPath}' does not exist. Creating it ..."
        )
        os.makedirs(downloadPath)
    # Download the video file
    print(f"- Downloading the video file from '{url}' ...")
    fileName = f"{fileName}.{format}"
    # Download the video file
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Create download address
        downloadPath = os.path.join(downloadPath, fileName)
        # Save the downloaded file
        with open(downloadPath, "wb") as file:
            file.write(response.content)
        success = True
    except Exception as e:
        print(f"- Error downloading the video file from '{url}'! {e}")
        success = False
    # Return the success status
    if success:
        print(f"- Video file downloaded successfully to '{downloadPath}'!")
    else:
        print(f"- Video file download failed!")
    return success
