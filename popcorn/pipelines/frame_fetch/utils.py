import os
import string
import cv2 as cv
import numpy as np
from glob import glob

# Supported video formats
videoFormats = ["mp4", "avi", "mkv"]

# Supported image formats
frameImageFormats = ["jpg", "jpeg", "png"]


def fetchMovieVideoFiles(configs: dict):
    """
    Pre-checks the given directory for movies and retrieves the list of video files.

    Parameters
    ----------
    configs: dict
        The configurations dictionary

    Returns
    -------
    videoFiles: list
        A list of fetched video files
    """
    # Variables
    videoFiles = []
    moviesDir = configs["pipelines"]["frame_extractor"]["movies_path"]
    framesDir = configs["pipelines"]["frame_extractor"]["frames_path"]
    # Check if the given directory exists
    if not os.path.exists(moviesDir):
        print(f"-- Input movie videos directory '{moviesDir}' does not exist!")
        return []
    print(f"-- Processing input movie videos from '{moviesDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(framesDir):
        os.mkdir(framesDir)
        print(f"-- Output frames will be saved in '{framesDir}' ...")
    # Check the supported video types
    print(
        f"-- Checking the input directory to find videos with supported formats '{videoFormats}' ..."
    )
    # Get the list of videos in the movies directory
    for format in videoFormats:
        videoFiles.extend(glob(f"{moviesDir}/*.{format}"))
    # Inform the user about the number of videos to process
    if len(videoFiles) == 0:
        print(f"-- [Warn] No video files found in the given directory '{moviesDir}'!")
        return []
    print(f"-- Found {len(videoFiles)} videos to be processed! (e.g., {videoFiles[0]})")
    # Return the list of video files
    return videoFiles


def initFramesPath(videoFilePath: str, framesRootPath: str):
    """
    Pre-checks and generates the output frames folder

    Parameters
    ----------
    videoFilePath: str
        The video file address to extract frames from
    framesRootPath: str
        The frames directory to save the extracted frames

    Returns
    -------
    generatedPath: str
        The generated frames directory path
    """
    # Variables
    generatedPath = ""
    # Accessing video file
    videoName = os.path.basename(videoFilePath)
    # Normalizing the video name to assign it to the output folder
    videoName = string.capwords(videoName.split(".")[0].replace("_", "")).replace(
        " ", ""
    )
    # Creating output folder
    if not os.path.exists(framesRootPath):
        print(f"-- Creating the frames root directory '{framesRootPath}' ...")
        os.mkdir(framesRootPath)
    generatedPath = os.path.join(framesRootPath, videoName)
    # Do not re-generate frames for movies if there is a folder with their normalized name
    if os.path.exists(generatedPath):
        print(
            f"-- Skipping movie '{videoName}'! A folder with the same name already exists in '{framesRootPath}' ..."
        )
        return ""
    else:
        os.mkdir(generatedPath)
        return generatedPath


def resizeFrame(frame: cv.Mat, networkInputSize: int = 300):
    """
    Resize the given frame while preserving its aspect ratio

    Parameters
    ----------
    frame: cv.Mat
        The frame to be resized
    networkInputSize: int
        The network input size to resize the frame to

    Returns
    -------
    pFrame: cv.Mat
        The resized frame
    """
    # Calculating frame dimensions
    height, width = frame.shape[:2]
    # Calculate the aspect ratio
    aspectRatio = width / height
    # Resize frame's width, while keeping its aspect ratio
    pFrameWidth = networkInputSize
    pFrameHeight = int(pFrameWidth / aspectRatio)
    # Scale the frame
    pFrame = cv.resize(frame, (pFrameWidth, pFrameHeight), interpolation=cv.INTER_AREA)
    # Return the resized frame
    return pFrame


def generateSquareFrame(frame: cv.Mat, networkInputSize: int = 300):
    """
    Generate a square frame from the given frame

    Parameters
    ----------
    frame: cv.Mat
        The frame to be resized and padded
    networkInputSize: int
        The network input size to resize the frame to

    Returns
    -------
    pFrame: cv.Mat
        The scaled and padded frame
    """
    # Variables
    paddingColor = [0, 0, 0]
    frameDimension = (networkInputSize, networkInputSize)
    # Calculating frame dimensions
    height, width = frame.shape[:2]
    aspectRatio = width / height
    # Choosing proper interpolation
    dimensionH, dimensionW = frameDimension
    interpolation = cv.INTER_CUBIC  # Stretch the image
    if height > dimensionH or width > dimensionW:
        interpolation = cv.INTER_AREA  # Shrink the image
    # Add paddings to the image
    if aspectRatio > 1:  # Image is horizontal
        pFrameWidth = dimensionW
        pFrameHeight = np.round(pFrameWidth / aspectRatio).astype(int)
        paddingVertical = (dimensionH - pFrameHeight) / 2
        paddingT, paddingB = np.floor(paddingVertical).astype(int), np.ceil(
            paddingVertical
        ).astype(int)
        paddingL, paddingR = 0, 0
    elif aspectRatio < 1:  # Image is vertical
        pFrameHeight = dimensionH
        pFrameWidth = np.round(pFrameHeight * aspectRatio).astype(int)
        paddingHorizontal = (dimensionW - pFrameWidth) / 2
        paddingL, paddingR = np.floor(paddingHorizontal).astype(int), np.ceil(
            paddingHorizontal
        ).astype(int)
        paddingT, paddingB = 0, 0
    else:  # image is square, so no changes is needed
        pFrameHeight, pFrameWidth = dimensionH, dimensionW
        paddingL, paddingR, paddingT, paddingB = 0, 0, 0, 0
    # Scale the frame
    pFrame = cv.resize(frame, (pFrameWidth, pFrameHeight), interpolation=interpolation)
    # Add paddings to the image
    pFrame = cv.copyMakeBorder(
        pFrame,
        paddingT,
        paddingB,
        paddingL,
        paddingR,
        borderType=cv.BORDER_CONSTANT,
        value=paddingColor,
    )
    # Return the scaled and padded frame
    return pFrame
