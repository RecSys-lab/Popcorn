import os
import json
import string
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from scipy import spatial

def initFramesFoldersForShotDetection(configs: dict):
    """
    Pre-checks the given directory for movie frames and prepares it for shot detection

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    
    Returns
    -------
    framesFolders :list
        A list of fetched frames folders
    """
    # Variables
    framesFolders = []
    imageTypes = configs['image_formats']
    movieFramesRootDir, shotFramesRootDir = configs['frames_path'], configs['shot_frames_path']
    # Check if the given directory exists
    if not os.path.exists(movieFramesRootDir):
        print(f"Input movie frames root directory '{movieFramesRootDir}' does not exist! Exiting ...")
        return False
    print(f"Movie frames will be processed from the root directory '{movieFramesRootDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(shotFramesRootDir):
        os.mkdir(shotFramesRootDir)
        print(f"Selected shot frames will be saved in '{shotFramesRootDir}' ...")
    # Check the supported frame types
    print(f"Processing the input frames directory. Supported frame formats are {imageTypes} ...")
    # Get the list of movie folders in the root directory
    for frameFolder in glob(f'{movieFramesRootDir}/*/'):
        for imageType in imageTypes:
            if glob(f'{frameFolder}*.{imageType}'):
                framesFolders.append(frameFolder)
                break  # Exit inner loop if at least one frame is found in this folder
    # Inform the user about the number of frame folders to process
    if len(framesFolders) == 0:
        print(f"No movie frame folders found in the given directory '{movieFramesRootDir}'! Exiting ...")
        return False
    print(f"Found {len(framesFolders)} folders containing frames to process! (e.g., {framesFolders[0]})\n")
    # Return the list of video files
    return framesFolders

def initFeaturesFoldersForShotDetection(configs: dict):
    """
    Pre-checks the given directory for extracted movie features and prepares it for shot detection

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    
    Returns
    -------
    featuresFolders :list
        A list of fetched features folders
    """
    # Variables
    featuresFolders = []
    movieFeaturesRootDir, shotFeaturesRootDir = configs['features_path'], configs['shot_features_path']
    movieFeaturesRootDir = os.path.normpath(movieFeaturesRootDir)
    shotFeaturesRootDir = os.path.normpath(shotFeaturesRootDir)
    # Check if the given directory exists
    if not os.path.exists(movieFeaturesRootDir):
        print(f"Input movie features root directory '{movieFeaturesRootDir}' does not exist! Exiting ...")
        return False
    print(f"Movie features will be processed from the root directory '{movieFeaturesRootDir}' ...")
    # Check if the output directory exists and create it if not
    if not os.path.exists(shotFeaturesRootDir):
        os.mkdir(shotFeaturesRootDir)
        print(f"Selected shot features will be saved in '{shotFeaturesRootDir}' ...")
    # Check the supported feature types
    print(f"Processing the input features directory. Supported feature format is 'json' ...")
    # Get the list of movie folders in the root directory
    for featureFolder in glob(f'{movieFeaturesRootDir}/*/'):
        if glob(f'{featureFolder}*.json'):
            featuresFolders.append(featureFolder)
    # Inform the user about the number of feature folders to process
    if len(featuresFolders) == 0:
        print(f"No movie feature folders found in the given directory '{movieFeaturesRootDir}'! Exiting ...")
        return False
    print(f"Found {len(featuresFolders)} folders containing features to process! (e.g., {featuresFolders[0]})\n")
    # Return the list of video files
    return featuresFolders

def initShotsFolder(featuresDir: str, outputDir: str):
    """
    Pre-checks and generates the output shots visual features folder

    Parameters
    ----------
    featuresDir: str
        The features folder address to pick shots visual features from
    outputDir: str
        The shots visual features directory to save the extracted features packets
    
    Returns
    -------
    generatedPath: str
        The generated shots visual features directory path
    """
    # Take the last part of the frames directory
    folderName = os.path.basename(featuresDir)
    # Normalizing the frames folder name to assign it to the output feature folder
    folderName = string.capwords(folderName.replace("_", "")).replace(" ", "")
    # Creating output folder
    generatedPath = os.path.join(outputDir, folderName)
    # Do not re-generate feature packets for movie features if there is a folder with their normalized name
    if os.path.exists(generatedPath):
        print(
            f'- Skipping "{folderName}" due to finding an output folder with the same name!')
        return
    else:
        os.mkdir(generatedPath)
        return generatedPath

def mergePacketsIntoDataFrame(packetsFolder: str):
    """
    Merges all the visual features in JSON files into a single DataFrame

    Parameters
    ----------
    packetsFolder : str
        Path to the folder containing the JSON files (packets) of extracted visual features

    Returns
    -------
    mergedDataFrame : DataFrame
        DataFrame containing all the visual features in JSON files

    """
    # Variables
    mergedDataFrame = pd.DataFrame(columns=['frameId', 'features'])
    # Iterate over the packet files to collect them all in a single dataframe
    for packetIdx, packetFile in enumerate(glob(f'{packetsFolder}/*.json')):
        # Inform the user about the processing packet
        if (packetIdx % 50 == 0):
            print(f'-- Fetching packet #{packetIdx} ...')
        # Reading each packet's data
        jsonFile = open(packetFile,)
        # Load the JSON data
        packetData = json.load(jsonFile)
        # Iterate on each frames of array
        for frameData in packetData:
            mergedDataFrame = pd.concat([mergedDataFrame, pd.DataFrame([{'frameId': frameData['frameId'], 'features': frameData['features']}])],
                                           ignore_index=True)
        # Close the JSON file
        jsonFile.close()
    return mergedDataFrame

def calculateCosineSimilarity(movieId: str, featuresDF: pd.DataFrame):
    """
    Calculates the cosine similarity between sequential features of a given feature-set

    Parameters
    ----------
    movieId : str
        The movie identifier to be used in the process
    featuresDF : pd.DataFrame
        The dataframe containing the visual features of the movie frames

    Returns
    -------
    similarityDF: pd.DataFrame
        The dataframe containing the cosine similarity among sequential features
    """
    # Variables
    similarityDF = pd.DataFrame(
        columns=['source', 'destination', 'similarity'])
    # Ensure the similarityDF has a consistent data type for each column to avoid issues during concatenation
    similarityDF['source'] = similarityDF['source'].astype('object')
    similarityDF['destination'] = similarityDF['destination'].astype('object')
    similarityDF['similarity'] = similarityDF['similarity'].astype('float64')
    # Inform the user about the process
    print(
        f'- Calculating cosine similarity among sequential frames of "{movieId}" ...')
    # Calculate the cosine similarity between sequential features
    for index in range(len(featuresDF)-1):
        # Similarity calculation
        similarity = 1 - spatial.distance.cosine(
            featuresDF['features'][index],
            featuresDF['features'][index + 1])
        # Round the similarity value
        similarity = round(similarity, 2)
        # Create a row for the similarity dataframe
        row = pd.DataFrame([{
            'source': featuresDF['frameId'][index],
            'destination': featuresDF['frameId'][index + 1],
            'similarity': similarity
        }])
        # Append the similarity to the dataframe
        if not row.isna().all().all():
            similarityDF = pd.concat([similarityDF, row], ignore_index=True)
    # Return the similarity dataframe
    return similarityDF

def calculateCosineSimilarity(prevFrame: cv.Mat, currFrame: cv.Mat, threshold: float = 0.7):
    """
    Calculates the cosine similarity between sequential frames

    Parameters
    ----------
    prevFrame : cv.Mat
        The previous frame to compare
    currFrame : cv.Mat
        The current frame to compare
    threshold : float, optional
        The threshold value to consider the similarity as a shot boundary, by default 0.7

    Returns
    -------
    isSimilar: bool
        The boolean value indicating the similarity between the frames
    """
    # Variables
    isSimilar = False
    # Convert the frames to grayscale
    pFrameGray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    cFrameGray = cv.cvtColor(currFrame, cv.COLOR_BGR2GRAY)
    # Check if the frames are not blank
    if (np.all(pFrameGray == 0) or np.all(cFrameGray == 0)):
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
        print(f"- Error calculating the cosine similarity: {str(e)}")
        return False


def calculateShotBoundaries(similarityDF: pd.DataFrame, threshold: float = 0.7):
    """
    Detects shot boundaries in the similarity dataframe and returns the middle frames of the shots

    Parameters
    ----------
    similarityDF : pd.DataFrame
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
    boundariesDF = similarityDF[similarityDF['similarity'] < threshold]
    # Get the index of shot boundaries
    boundariesList = boundariesDF.index.tolist()
    boundariesList = [int(bndry) for bndry in boundariesList]
    # Get the middle index of the shot boundaries
    for item1, item2 in zip(boundariesList, boundariesList[1:]):
        middleItem = int((item1 + item2)/2)
        boundaryFrames.append(middleItem)
    # Return the list of keyframes
    return boundaryFrames