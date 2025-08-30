import os
import time
import numpy as np
import pandas as pd
from popcorn.utils import loadJsonFromFilePath

def aggregateMovieFeatures(configs: dict):
    """
    Aggregates features from the given set of extracted movie features (offline/downloaded mode)

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    """
    print("Aggregating visual features from the given set of extracted movie features ...")
    # Variables
    rootDir = configs['features_path']
    aggFeaturesDir = configs['agg_features_path']
    aggMethods = configs['aggregation_models'] # Such as ["Max", "Mean"]
    # Iterate on all feature folders in the given root directory
    for featureFolder in os.listdir(rootDir):
        # Preparing the output features directory
        featureFolder = os.path.normpath(os.path.join(rootDir, featureFolder))
        print(f"- Aggregating features from the features in '{featureFolder}' ...")
        outputDir = os.path.normpath(aggFeaturesDir)
        outputFile = os.path.join(outputDir, f"{os.path.basename(featureFolder)}.json")
        # Skip if the output file already exists
        if os.path.exists(outputFile):
            print(f"-- The output file '{outputFile}' already exists! Skipping ...")
            continue
        # Otherwise, prepare variables
        packetCounter = 0
        movieAggFeatures = []
        movieAggFeat_Max = []
        movieAggFeat_Mean = []
        startTime = time.time()
        numPacketFiles = len(os.listdir(featureFolder))
        print(f"-- Aggregating {numPacketFiles} packet files ...")
        # Iterate on all packet files in the given feature folder
        for packetFile in os.listdir(featureFolder):
            # Read the packet file
            packetFilePath = os.path.join(featureFolder, packetFile)
            # Read the packet file (JSON)
            packetData = loadJsonFromFilePath(packetFilePath)
            # Iterate over each item in the packet data
            for frameData in packetData:
                # Get the features
                features = frameData['features']
                features = np.array(features)
                # Aggregate the features
                movieAggFeatures.append(features)
            # Increment the packet counter
            packetCounter += 1
            # Show progress every 50 packet files
            if packetCounter % 50 == 0:
                print(f"-- Aggregated {packetCounter} packet files ...")
        # Aggregate the features
        movieAggFeatures = np.array(movieAggFeatures)
        if "Max" in aggMethods:
            movieAggFeat_Max = np.max(movieAggFeatures, axis=0)
            movieAggFeat_Max = np.round(movieAggFeat_Max, 6)
        if "Mean" in aggMethods:
            movieAggFeat_Mean = np.mean(movieAggFeatures, axis=0)
            movieAggFeat_Mean = np.round(movieAggFeat_Mean, 6)
        # Save the aggregated features in a dataFrame
        dataFrame = pd.DataFrame(columns=aggMethods)
        dataFrame = pd.concat([dataFrame, pd.DataFrame([{'Max': movieAggFeat_Max, 'Mean': movieAggFeat_Mean}])], ignore_index=True)
        # Save the dataFrame as a JSON file
        dataFrame.to_json(outputFile)
        # Better logging for the user
        elapsedTime = time.time() - startTime
        print(f"-- Aggregated {packetCounter} packet files in {elapsedTime:.2f} seconds and saved in '{outputDir}' ...")