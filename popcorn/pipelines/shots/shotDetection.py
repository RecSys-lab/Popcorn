import os
import time
import cv2 as cv
import pandas as pd
from popcorn.pipelines.visual_features.utils import packetManager
from popcorn.pipelines.shots.utils import calculateCosineSimilarity, calculateShotBoundaries, initShotsFolder, mergePacketsIntoDataFrame

def extractShotsFromMovieFrames(configs: dict, movieFramesPaths: list):
    """
    Extracts shots from the given set of extracted movie frames

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    movieFramesPaths: list
        The list of movie frames paths
    """
    print("Extracting movie shots from the given set of movie frames ...")
    # Iterate on all frames folders in the given directory
    for framesFolder in movieFramesPaths:
        # Preparing the output shots frames directory
        framesFolder = os.path.normpath(framesFolder)
        outputDir = initShotsFolder(framesFolder, configs['shot_frames_path'])
        if not outputDir:
            continue
        # Picking shot features from the given features folder
        try:
            # Variables
            totalShots = 0
            prevFrame = None
            startTime = time.time()
            folderName = os.path.basename(framesFolder)
            totalFrames = len(os.listdir(framesFolder))
            # Loop over the frames to pick the shots
            if totalFrames < 1:
                print(f'- No frames found in "{framesFolder}"! Skipping ...')
                continue
            print(f'- Processing {totalFrames} frames of movie "{folderName}" ...')
            for frameFile in os.listdir(framesFolder):
                # Read the frame file
                framePath = os.path.join(framesFolder, frameFile)
                currFrame = cv.imread(framePath)
                # Check if the frame is read successfully
                if currFrame is None:
                    print(f'- Error reading frame file "{frameFile}" in "{framesFolder}"! Skipping ...')
                    continue
                # If the previous frame is not None, fill it with the current frame
                if prevFrame is None:
                    prevFrame = currFrame.copy()
                    continue
                # Check the cosine similarity of the frame with the next one
                isShot = calculateCosineSimilarity(prevFrame, currFrame, configs['threshold'])
                if isShot:
                    # Save the current frame as a shot
                    shotPath = os.path.join(outputDir, frameFile)
                    cv.imwrite(shotPath, currFrame)
                    totalShots += 1
            # Inform the user
            elapsedTime = '{:.2f}'.format(time.time() - startTime)
            print(
                f'- Extracted {totalShots} shots from {totalFrames} frames of "{folderName}" in {elapsedTime} seconds!')
        except Exception as error:
            print(f'- Error while picking the shots of "{folderName}" in "{framesFolder}": {str(error)}')
            continue

def extractShotsFromMovieFeatures(configs: dict, movieFeaturesFolders: list):
    """
    Extracts shots from the given set of extracted movie features

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    movieFeaturesFolders: list
        The list of movie features paths
    """
    print("Extracting movie shots from the given set of movie features ...")
    # Iterate on all features folders in the given directory
    for featuresFolder in movieFeaturesFolders:
        # Preparing the output shots features directory
        featuresFolder = os.path.normpath(featuresFolder)
        outputDir = initShotsFolder(featuresFolder, configs['shot_features_path'])
        if not outputDir:
            continue
        # Picking shot features from the given features folder
        try:
            # Variables
            packetCounter = 0
            startTime = time.time()
            folderName = os.path.basename(featuresFolder)
            packetIndex = 1  # Holds the name of the packet, e.g. Packet0001
            shotsDataFrame = pd.DataFrame(columns=['frameId', 'features'])
            # Ensure the shotsDataFrame has a consistent data type for each column to avoid issues during concatenation
            shotsDataFrame['frameId'] = shotsDataFrame['frameId'].astype('int32')
            shotsDataFrame['features'] = shotsDataFrame['features'].astype('object')
            # Explore the folder containing JSON files (packets) of extracted visual features
            totalPackets = len(os.listdir(featuresFolder))
            print(f'- Processing {totalPackets} packets of movie "{folderName}" ...')
            # Iterate over the packet files to collect them all in a single dataframe
            featuresDF = mergePacketsIntoDataFrame(featuresFolder)
            # Check if the features dataframe is empty
            if featuresDF.empty:
                print(f'- The DataFrame containing packets data of "{folderName}" is empty! Skipping ...')
                continue
            # Print the number of frames in the features dataframe
            print(f'- {len(featuresDF)} packets combined into a single DataFrame for processing!')
            # Cosine similarity calculation
            similarityDF = calculateCosineSimilarity(folderName, featuresDF)
            # Find shot boundaries and select the middle frame of each shot
            boundaryFrames = calculateShotBoundaries(similarityDF, configs['threshold'])
            # Keep only the boundary frames from the features dataframe
            boundaryDF = featuresDF[featuresDF.index.isin(boundaryFrames)]
            print(f'- {len(boundaryDF)} shot boundaries found in "{folderName}"!')
            # Iterate over the keyframes to save them in packets
            remainingFramesCount = len(boundaryDF)
            for index, row in boundaryDF.iterrows():
                # Create a row for the similarity dataframe
                shotRow = pd.DataFrame([{
                    'frameId': row['frameId'],
                    'features': row['features']
                }])
                # Append the similarity to the dataframe
                if not shotRow.isna().all().all():
                    shotsDataFrame = pd.concat([shotsDataFrame, shotRow], ignore_index=True)
                packetCounter += 1
                # Reset the counter only if packetCounter reaches the limit and there is no more frames for process
                remainingFramesCount -= 1
                resetCounter = (packetCounter == configs['packet_size']) or (remainingFramesCount == 0)
                if (resetCounter):
                    # Save dataFrame as packet in a file
                    packetManager(packetIndex, shotsDataFrame,
                                    folderName, outputDir)
                    # Clear dataFrame rows
                    shotsDataFrame.drop(shotsDataFrame.index, inplace=True)
                    packetCounter = 0
                    packetIndex += 1
            # Inform the user
            elapsedTime = '{:.2f}'.format(time.time() - startTime)
            print(
                f'- Extracted {packetIndex-1} shot packets from {totalPackets} packets of "{folderName}" in {elapsedTime} seconds!')
        except Exception as error:
            print(f'- Error while picking the shots of "{folderName}" in "{featuresFolder}": {str(error)}')
            continue