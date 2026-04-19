import os
import cv2 as cv
import numpy as np
import pandas as pd


def indexThumbnails(root: str):
    """
    Indexes all the thumbnail files (images) in the given root directory
    and returns a DataFrame containing the indexed thumbnails (movieId, path).

    Parameters
    ----------
    root: str
        The root directory to index thumbnails from.

    Returns
    -------
    df: pd.DataFrame
        A DataFrame containing the indexed thumbnails.
    """
    print(f"- Indexing thumbnails from: {root} ...")
    # Create a list to store the indexed thumbnails
    indexedThumbnails = []
    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".jpg"):
                # Get the full path to the image file
                imgPath = os.path.join(dirpath, filename)
                # Movie-ID (file name without extension)
                movieId = os.path.splitext(filename)[0]
                # Append the image and its path to the indexed thumbnails
                indexedThumbnails.append((int(movieId), imgPath))
    # Create a DataFrame from the indexed thumbnails
    df = pd.DataFrame(indexedThumbnails, columns=["movieId", "path"])
    print(f"- Indexed {len(indexedThumbnails)} thumbnails DataFrame: \n{df.head()}")
    return df


def loadMovieThumbnail(movieId: int, config: dict) -> np.ndarray:
    """
    Load the thumbnail for a specific movie into an OpenCV image.

    Parameters
    ----------
    movieId: int
        The movie ID to load the thumbnail for.

    Returns
    -------
    np.ndarray
        The loaded thumbnail image.
    """
    # Variables
    openCVImage = None
    # Load all indexed thumbnails into a DataFrame
    root = config["datasets"]["unimodal"]["ml_thumbnail"]["download_path"]
    thumbnailsDF = indexThumbnails(root)
    # Get the thumbnail path for the specific movie ID
    thumbnailPath = thumbnailsDF[thumbnailsDF["movieId"] == movieId]["path"].values
    if len(thumbnailPath) == 0:
        print(f"- [Error] Thumbnail not found for movie ID: {movieId}")
        return openCVImage
    # Load the image using OpenCV
    openCVImage = cv.imread(thumbnailPath[0])
    print(f"- Loaded thumbnail for movie-id '{movieId}'!")
    # Return the loaded image
    return openCVImage
