import os
import string
import cv2 as cv
import numpy as np
from glob import glob

def initMovieVideos(configs: dict):
    """
    Pre-checks the given directory for movies and prepares it for further processing

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    
    Returns
    -------
    videoFiles :list
        A list of fetched video files
    """
    # Variables
    videoFiles = []
    videoTypes = configs['video_formats']
    moviesDir, framesDir = configs['movies_path'], configs['frames_path']
    # Check if the given directory exists
    if not os.path.exists(moviesDir):
        print(f"Input movie videos directory '{moviesDir}' does not exist! Exiting ...")
        return False
    print(f"Movie videos will be processed from '{moviesDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(framesDir):
        os.mkdir(framesDir)
        print(f"Output frames will be saved in '{framesDir}' ...")
    # Check the supported video types
    print(f"Processing the input movies directory. Supported video formats are {videoTypes} ...")
    # Get the list of videos in the movies directory
    for type in videoTypes:
        videoFiles.extend(glob(f'{moviesDir}/*.{type}'))
    # Inform the user about the number of videos to process
    if len(videoFiles) == 0:
        print(f"No video files found in the given directory '{moviesDir}'! Exiting ...")
        return False
    print(f"Found {len(videoFiles)} videos to process! (e.g., {videoFiles[0]})\n")
    # Return the list of video files
    return videoFiles

def initFramesFolder(videoFileAddress: str, framesDir: str):
    """
    Pre-checks and generates the output frames folder

    Parameters
    ----------
    videoFileAddress: str
        The video file address to extract frames from
    framesDir: str
        The frames directory to save the extracted frames
    
    Returns
    -------
    generatedPath: str
        The generated frames directory path
    """
    # Accessing video file
    videoName = os.path.basename(videoFileAddress)
    # Normalizing the video name to assign it to the output folder
    videoName = string.capwords(
        videoName.split('.')[0].replace("_", "")).replace(" ", "")
    # Creating output folder
    if not os.path.exists(framesDir):
        os.mkdir(framesDir)
    generatedPath = os.path.join(framesDir, videoName)
    # Do not re-generate frames for movies if there is a folder with their normalized name
    if os.path.exists(generatedPath):
        print(
            f'- Skipping {videoName} due to finding an output folder with the same name!')
        return
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
    pFrame = cv.resize(
        frame, (pFrameWidth, pFrameHeight), interpolation=cv.INTER_AREA)
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
    if (height > dimensionH or width > dimensionW):
        interpolation = cv.INTER_AREA  # Shrink the image
    # Add paddings to the image
    if aspectRatio > 1:  # Image is horizontal
        pFrameWidth = dimensionW
        pFrameHeight = np.round(pFrameWidth / aspectRatio).astype(int)
        paddingVertical = (dimensionH - pFrameHeight) / 2
        paddingT, paddingB = np.floor(paddingVertical).astype(
            int), np.ceil(paddingVertical).astype(int)
        paddingL, paddingR = 0, 0
    elif aspectRatio < 1:  # Image is vertical
        pFrameHeight = dimensionH
        pFrameWidth = np.round(pFrameHeight * aspectRatio).astype(int)
        paddingHorizontal = (dimensionW - pFrameWidth) / 2
        paddingL, paddingR = np.floor(paddingHorizontal).astype(
            int), np.ceil(paddingHorizontal).astype(int)
        paddingT, paddingB = 0, 0
    else:  # image is square, so no changes is needed
        pFrameHeight, pFrameWidth = dimensionH, dimensionW
        paddingL, paddingR, paddingT, paddingB = 0, 0, 0, 0
    # Scale the frame
    pFrame = cv.resize(
        frame, (pFrameWidth, pFrameHeight), interpolation=interpolation)
    # Add paddings to the image
    pFrame = cv.copyMakeBorder(pFrame, paddingT, paddingB,
                                     paddingL, paddingR, borderType=cv.BORDER_CONSTANT, value=paddingColor)
    # Return the scaled and padded frame
    return pFrame
