import os
import time
import pandas as pd
from popcorn.pipelines.visual_embedding.utils import packetManager
from popcorn.pipelines.shot_detect.utils import (
    initShotsPath,
    calculateShotBoundaries,
    mergePacketsIntoDataFrame,
    fetchMovieEmbeddingFolders,
    calculateCosineSimilarityEmbeddings
)


def extractShotsFromEmbeddings(configs: dict):
    """
    Extracts shots from the given set of movie visual embeddings

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    """
    # Variables
    pipelineName = configs["pipelines"]["shot_detector"]["name"]
    threshold = configs["pipelines"]["shot_detector"]["threshold"]
    shotsPath = configs["pipelines"]["shot_detector"]["shots_path"]
    packetSize = configs["pipelines"]["shot_detector"]["variants"]["from_embeddings"][
        "packet_size"
    ]
    shotsPath = os.path.normpath(shotsPath)
    print(f"- Starting the '{pipelineName}' (from embeddings) pipeline ...")
    # Check the validity of arguments and configurations
    if threshold < 0.1 or threshold > 1.0:
        print(
            f"- [Warn] The given threshold '{threshold}' is not valid (0.1 <= threshold <= 1.0)! Setting it to 0.7."
        )
        threshold = 0.7
    if packetSize <= 0 or packetSize > 50:
        print(
            f"- [Warn] The given packet size '{packetSize}' is not valid (0 < size <= 50)! Setting it to 25."
        )
        packetSize = 25
    # Fetch the list of movie embedding folders (if any)
    fetchedEmbeddings = fetchMovieEmbeddingFolders(configs)
    if len(fetchedEmbeddings) == 0 or not fetchedEmbeddings:
        print(f"- [Warn] No valid embedding folders found! Exiting ...")
        return
    print("- Extracting movie shots from the given set of movie embeddings ...")
    # Iterate on all features folders in the given directory
    for featuresDir in fetchedEmbeddings:
        # Preparing the output shots features directory
        featuresDir = os.path.normpath(featuresDir)
        print(f"- Extracting movie shots from the embeddings in '{featuresDir}' ...")
        outputDir = initShotsPath(featuresDir, shotsPath)
        if not outputDir:
            continue
        # Picking shot features from the given features folder
        try:
            # Variables
            packetCounter = 0
            startTime = time.time()
            folderName = os.path.basename(featuresDir)
            packetIndex = 1  # Holds the name of the packet, e.g. Packet0001
            shotsDF = pd.DataFrame(columns=["frameId", "features"])
            # Ensure the shotsDF has a consistent data type for each column to avoid issues during concatenation
            shotsDF["frameId"] = shotsDF["frameId"].astype("int32")
            shotsDF["features"] = shotsDF["features"].astype("object")
            # Explore the folder containing JSON files (packets) of extracted visual features
            totalPackets = len(os.listdir(featuresDir))
            print(f"-- Processing '{totalPackets}' packets of movie '{folderName}' ...")
            # Iterate over the packet files to collect them all in a single dataframe
            featuresDF = mergePacketsIntoDataFrame(featuresDir)
            # Check if the features dataframe is empty
            if featuresDF.empty:
                print(
                    f"-- [Error] The DataFrame containing packets data of '{folderName}' is empty! Skipping ..."
                )
                continue
            # Print the number of frames in the features dataframe
            print(
                f"-- {len(featuresDF)} packets combined into a single DataFrame for processing!"
            )
            # Cosine similarity calculation
            similarityDF = calculateCosineSimilarityEmbeddings(featuresDF)
            # Find shot boundaries and select the middle frame of each shot
            boundaryFrames = calculateShotBoundaries(similarityDF, threshold)
            # Keep only the boundary frames from the features dataframe
            boundaryDF = featuresDF[featuresDF.index.isin(boundaryFrames)]
            print(f"-- {len(boundaryDF)} shot boundaries found in '{folderName}'!")
            # Iterate over the keyframes to save them in packets
            remainingFramesCount = len(boundaryDF)
            for index, row in boundaryDF.iterrows():
                # Create a row for the similarity dataframe
                shotRow = pd.DataFrame(
                    [{"frameId": row["frameId"], "features": row["features"]}]
                )
                # Append the similarity to the dataframe
                if not shotRow.isna().all().all():
                    shotsDF = pd.concat(
                        [shotsDF, shotRow], ignore_index=True
                    )
                packetCounter += 1
                # Reset the counter only if packetCounter reaches the limit and there is no more frames for process
                remainingFramesCount -= 1
                resetCounter = (packetCounter == packetSize) or (
                    remainingFramesCount == 0
                )
                if resetCounter:
                    # Save dataFrame as packet in a file
                    packetManager(packetIndex, shotsDF, folderName, outputDir)
                    # Clear dataFrame rows
                    shotsDF.drop(shotsDF.index, inplace=True)
                    packetCounter = 0
                    packetIndex += 1
            # Inform the user
            elapsedTime = "{:.2f}".format(time.time() - startTime)
            print(
                f"- Extracted {packetIndex-1} shot packets from {totalPackets} packets of '{folderName}' in {elapsedTime} seconds!"
            )
        except Exception as error:
            print(
                f"- [Error] Error while picking the shots of '{folderName}' in '{featuresDir}': {str(error)}"
            )
            continue
