import os
import time
import numpy as np
import pandas as pd
from popcorn.utils import loadJsonFromFilePath
from popcorn.pipelines.utils import aggMethods


def aggregateVisualEmbeddings(configs: dict):
    """
    Aggregates visual embeddings from the given set of extracted movie visual embeddings (offline/downloaded mode)

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    """
    print(
        "Aggregating visual features from the given set of extracted movie features ..."
    )
    # Variables
    embeddingsDir = configs["pipelines"]["visual_embedding_aggregator"]["features_path"]
    outputDir = configs["pipelines"]["visual_embedding_aggregator"]["agg_features_path"]
    methods = configs["pipelines"]["visual_embedding_aggregator"]["aggregation_methods"]
    # Check the validity of arguments and configurations
    if not os.path.exists(embeddingsDir):
        print(f"- [Warn] The given features path '{embeddingsDir}' does not exist! Exiting ...")
        return
    if not os.path.exists(outputDir):
        print(f"- [Warn] The given output path '{outputDir}' does not exist! Creating it ...")
        os.makedirs(outputDir)
    for method in methods:
        if method not in aggMethods:
            print(
                f"- [Warn] The given aggregation method '{method}' is not valid (should be one of {aggMethods})! Setting it to 'Mean'."
            )
            methods = ["Mean"]
            break
    # Iterate on all feature folders in the given root directory
    for featureFolder in os.listdir(embeddingsDir):
        # Preparing the output features directory
        featureFolder = os.path.normpath(os.path.join(embeddingsDir, featureFolder))
        print(f"- Aggregating features from the features in '{featureFolder}' ...")
        outputDir = os.path.normpath(outputDir)
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
                features = frameData["features"]
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
        if "Max" in methods:
            movieAggFeat_Max = np.max(movieAggFeatures, axis=0)
            movieAggFeat_Max = np.round(movieAggFeat_Max, 6)
        if "Mean" in methods:
            movieAggFeat_Mean = np.mean(movieAggFeatures, axis=0)
            movieAggFeat_Mean = np.round(movieAggFeat_Mean, 6)
        # Save the aggregated features in a dataFrame
        dataFrame = pd.DataFrame(columns=methods)
        dataFrame = pd.concat(
            [
                dataFrame,
                pd.DataFrame([{"Max": movieAggFeat_Max, "Mean": movieAggFeat_Mean}]),
            ],
            ignore_index=True,
        )
        # Save the dataFrame as a JSON file
        dataFrame.to_json(outputFile)
        # Better logging for the user
        elapsedTime = time.time() - startTime
        print(
            f"-- Aggregated {packetCounter} packet files in {elapsedTime:.2f} seconds and saved in '{outputDir}' ..."
        )
