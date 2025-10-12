import os
import json
import string
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from scipy import spatial
from popcorn.pipelines.utils import frameImageFormats


def fetchMovieFramesFolders(configs: dict):
    """
    Pre-checks the given directory for movie frames and retrieves the list of frame folders.

    Parameters
    ----------
    configs: dict
        The configurations dictionary

    Returns
    -------
    framesFolders: list
        A list of fetched frames folders
    """
    # Variables
    framesFolders = []
    shotsDir = configs["pipelines"]["shot_detector"]["shots_path"]
    framesDir = configs["pipelines"]["shot_detector"]["variants"]["from_frames"][
        "frames_path"
    ]
    # Check if the given directory exists
    if not os.path.exists(framesDir):
        print(
            f"- Input movie frames root directory '{framesDir}' does not exist! Exiting ..."
        )
        return []
    print(f"-- Processing input movie frames from '{framesDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(shotsDir):
        os.mkdir(shotsDir)
        print(
            f"-- Output shot-level visual embeddings will be saved in '{shotsDir}' ..."
        )
    # Check the supported frame types
    print(
        f"-- Checking the input directory to find frames with supported formats '{frameImageFormats}' ..."
    )
    # Get the list of movie folders in the root directory
    for frameDir in glob(f"{framesDir}/*/"):
        for format in frameImageFormats:
            if glob(f"{frameDir}*.{format}"):
                framesFolders.append(frameDir)
                break
    # Get the number of fetched folders
    if len(framesFolders) == 0:
        print(
            f"-- [Warn] No movie frame folder found in the given directory '{framesDir}'! Exiting ..."
        )
        return []
    print(
        f"-- Found {len(framesFolders)} folders containing frames to process! (e.g., {framesFolders[0]})\n"
    )
    # Return the list of video files
    return framesFolders


def fetchMovieEmbeddingFolders(configs: dict):
    """
    Pre-checks the given directory for movie embeddings and retrieves the list of embedding folders.

    Parameters
    ----------
    configs: dict
        The configurations dictionary

    Returns
    -------
    featuresFolders: list
        A list of fetched embeddings folders
    """
    # Variables
    featuresFolders = []
    shotsDir = configs["pipelines"]["shot_detector"]["shots_path"]
    featuresDir = configs["pipelines"]["shot_detector"]["variants"]["from_embeddings"][
        "features_path"
    ]
    # Check if the given directory exists
    if not os.path.exists(featuresDir):
        print(
            f"- Input movie embeddings root directory '{featuresDir}' does not exist! Exiting ..."
        )
        return []
    print(f"-- Processing input movie embeddings from '{featuresDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(shotsDir):
        os.mkdir(shotsDir)
        print(
            f"-- Output shot-level visual embeddings will be saved in '{shotsDir}' ..."
        )
    # Check the supported frame types
    print(
        f"-- Checking the input directory to find frame-level embeddings with 'JSON' format ..."
    )
    # Get the list of movie folders in the root directory
    for frameDir in glob(f"{featuresDir}/*/"):
        if glob(f"{frameDir}*.json"):
            featuresFolders.append(frameDir)
    # Get the number of fetched folders
    if len(featuresFolders) == 0:
        print(
            f"-- [Warn] No movie embedding folder found in the given directory '{featuresDir}'! Exiting ..."
        )
        return []
    print(
        f"-- Found {len(featuresFolders)} folders containing embeddings to process! (e.g., {featuresFolders[0]})\n"
    )
    # Return the list of video files
    return featuresFolders


def initShotsPath(inputPath: str, shotsRootPath: str):
    """
    Pre-checks and generates the output shot-level frames/embeddings folder

    Parameters
    ----------
    inputPath: str
        The frames/embeddings folder address to extract shots from
    shotsRootPath: str
        The root directory to save the extracted shot frames/embeddings

    Returns
    -------
    generatedPath: str
        The generated shot-level frames/embeddings directory path
    """
    # Variables
    generatedPath = ""
    # Take the last part of the frames directory
    folderName = os.path.basename(inputPath)
    # Normalizing the frames folder name to assign it to the output feature folder
    folderName = string.capwords(folderName.replace("_", "")).replace(" ", "")
    # Creating output folder
    if not os.path.exists(shotsRootPath):
        print(f"-- Creating the features root directory '{shotsRootPath}' ...")
        os.mkdir(shotsRootPath)
    # Creating output folder
    generatedPath = os.path.join(shotsRootPath, folderName)
    # Do not re-generate features for movie frames if there is a folder with their normalized name
    if os.path.exists(generatedPath):
        print(
            f"-- Skipping movie '{folderName}'! A folder with the same name already exists in '{shotsRootPath}' ..."
        )
        return ""
    else:
        os.mkdir(generatedPath)
        return generatedPath



def calculateCosineSimilarityFrames(pFrame: cv.Mat, cFrame: cv.Mat, threshold: float = 0.7):
    """
    Calculates the cosine similarity between sequential frames

    Parameters
    ----------
    pFrame: cv.Mat
        The previous frame to compare
    cFrame: cv.Mat
        The current frame to compare
    threshold: float, optional
        The threshold value to consider the similarity as a shot boundary, by default 0.7

    Returns
    -------
    isSimilar: bool
        The boolean value indicating the similarity between the frames
    """
    # Variables
    isSimilar = False
    # Convert the frames to grayscale
    pFrameGray = cv.cvtColor(pFrame, cv.COLOR_BGR2GRAY)
    cFrameGray = cv.cvtColor(cFrame, cv.COLOR_BGR2GRAY)
    # Check if the frames are not blank
    if np.all(pFrameGray == 0) or np.all(cFrameGray == 0):
        return isSimilar
    # Flatten the frames to a single dimension
    pFrameFlatten = pFrameGray.flatten()
    cFrameFlatten = cFrameGray.flatten()
    try:
        # Calculate the cosine similarity between the frames
        similarity = 1 - spatial.distance.cosine(pFrameFlatten, cFrameFlatten)
        # Round the similarity value
        similarity = round(similarity, 2)
        # Check if the similarity is greater than the threshold
        if similarity > threshold:
            isSimilar = True
        # Return the similarity result
        return isSimilar
    except Exception as e:
        print(f"-- [Error] Error calculating the cosine similarity: {str(e)}")
        return False


def calculateCosineSimilarityEmbeddings(featuresDF: pd.DataFrame):
    """
    Calculates the cosine similarity between sequential features of a given feature-set

    Parameters
    ----------
    featuresDF: pd.DataFrame
        The dataframe containing the visual features of the movie frames

    Returns
    -------
    similarityDF: pd.DataFrame
        The dataframe containing the cosine similarity among sequential features
    """
    # Variables
    similarityDF = pd.DataFrame(columns=["source", "destination", "similarity"])
    # Ensure the similarityDF has a consistent data type for each column to avoid issues during concatenation
    similarityDF["source"] = similarityDF["source"].astype("object")
    similarityDF["destination"] = similarityDF["destination"].astype("object")
    similarityDF["similarity"] = similarityDF["similarity"].astype("float64")
    # Calculate the cosine similarity between sequential features
    for index in range(len(featuresDF) - 1):
        # Similarity calculation
        similarity = 1 - spatial.distance.cosine(
            featuresDF["features"][index], featuresDF["features"][index + 1]
        )
        # Round the similarity value
        similarity = round(similarity, 2)
        # Create a row for the similarity dataframe
        row = pd.DataFrame(
            [
                {
                    "source": featuresDF["frameId"][index],
                    "destination": featuresDF["frameId"][index + 1],
                    "similarity": similarity,
                }
            ]
        )
        # Append the similarity to the dataframe
        if not row.isna().all().all():
            similarityDF = pd.concat([similarityDF, row], ignore_index=True)
    # Return the similarity dataframe
    return similarityDF

def mergePacketsIntoDataFrame(packetsFolder: str):
    """
    Merges all the visual features in JSON files into a single DataFrame

    Parameters
    ----------
    packetsFolder: str
        Path to the folder containing the JSON files (packets) of extracted visual features

    Returns
    -------
    mergedDataFrame: DataFrame
        DataFrame containing all the visual features in JSON files

    """
    # Variables
    mergedDataFrame = pd.DataFrame(columns=["frameId", "features"])
    # Iterate over the packet files to collect them all in a single dataframe
    for packetIdx, packetFile in enumerate(glob(f"{packetsFolder}/*.json")):
        # Inform the user about the processing packet
        if packetIdx % 50 == 0:
            print(f"-- Fetching packet #{packetIdx} ...")
        # Reading each packet's data
        jsonFile = open(
            packetFile,
        )
        # Load the JSON data
        packetData = json.load(jsonFile)
        # Iterate on each frames of array
        for frameData in packetData:
            mergedDataFrame = pd.concat(
                [
                    mergedDataFrame,
                    pd.DataFrame(
                        [
                            {
                                "frameId": frameData["frameId"],
                                "features": frameData["features"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        # Close the JSON file
        jsonFile.close()
    return mergedDataFrame





def calculateShotBoundaries(similarityDF: pd.DataFrame, threshold: float = 0.7):
    """
    Detects shot boundaries in the similarity dataframe and returns the middle frames of the shots

    Parameters
    ----------
    similarityDF: pd.DataFrame
        The similarity dataframe containing the cosine similarity among sequential features

    Returns
    -------
    boundaryFrames: list
        List of the middle frames between sequential shot boundaries
    """
    # Variables
    boundaryFrames = []
    print("- Calculating shot boundaries based on the similarity DataFrame ...")
    # Filter similarityDF to only include rows with similarity less than threshold (shot boundaries)
    boundariesDF = similarityDF[similarityDF["similarity"] < threshold]
    # Get the index of shot boundaries
    boundariesList = boundariesDF.index.tolist()
    boundariesList = [int(bndry) for bndry in boundariesList]
    # Get the middle index of the shot boundaries
    for item1, item2 in zip(boundariesList, boundariesList[1:]):
        middleItem = int((item1 + item2) / 2)
        boundaryFrames.append(middleItem)
    # Return the list of keyframes
    return boundaryFrames
