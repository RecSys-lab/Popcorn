import os
import time
import cv2 as cv
from popcorn.pipelines.utils import frameImageFormats
from popcorn.pipelines.shot_detect.utils import (
    initShotsPath,
    fetchMovieFramesFolders,
    calculateCosineSimilarityFrames,
)


def extractShotsFromFrames(configs: dict):
    """
    Extracts shots from the given set of movie frames

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    """
    # Variables
    pipelineName = configs["pipelines"]["shot_detector"]["name"]
    threshold = configs["pipelines"]["shot_detector"]["threshold"]
    shotsPath = configs["pipelines"]["shot_detector"]["shots_path"]
    frameFormat = configs["pipelines"]["shot_detector"]["variants"]["from_frames"][
        "frame_format"
    ]
    shotsPath = os.path.normpath(shotsPath)
    print(f"- Starting the '{pipelineName}' (from frames) pipeline ...")
    # Check the validity of arguments and configurations
    if frameFormat not in frameImageFormats:
        print(
            f"- [Warn] The given frame format '{frameFormat}' is not valid (should be one of {frameImageFormats})! Setting it to 'jpg'."
        )
        frameFormat = "jpg"
    if threshold < 0.1 or threshold > 1.0:
        print(
            f"- [Warn] The given threshold '{threshold}' is not valid (0.1 <= threshold <= 1.0)! Setting it to 0.7."
        )
        threshold = 0.7
    # Fetch the list of movie frames folders (if any)
    fetchedFrames = fetchMovieFramesFolders(configs)
    if len(fetchedFrames) == 0 or not fetchedFrames:
        print(f"- [Warn] No valid frames folders found! Exiting ...")
        return
    print("- Extracting movie shots from the given set of movie frames ...")
    # Iterate on all frames folders in the given directory
    for framesDir in fetchedFrames:
        # Preparing the output shots frames directory
        framesDir = os.path.normpath(framesDir)
        print(f"- Extracting movie shots from the frames in '{framesDir}' ...")
        outputDir = initShotsPath(framesDir, shotsPath)
        # Skip if the output directory already exists
        if not outputDir:
            continue
        # Picking shot features from the given features folder
        try:
            # Variables
            totalShots = 0
            prevFrame = None
            startTime = time.time()
            folderName = os.path.basename(framesDir)
            totalFrames = len(os.listdir(framesDir))
            # Loop over the frames to pick the shots
            if totalFrames < 1:
                print(f"-- [Warn] No frames found in '{framesDir}'! Skipping ...")
                continue
            print(f"-- Processing {totalFrames} frames of movie '{folderName}' ...")
            for frameFile in os.listdir(framesDir):
                # Read the frame file
                framePath = os.path.join(framesDir, frameFile)
                currFrame = cv.imread(framePath)
                # Check if the frame is read successfully
                if currFrame is None:
                    print(
                        f"-- [Error] Error reading frame file '{frameFile}' in '{framesDir}'! Skipping ..."
                    )
                    continue
                # If the previous frame is not None, fill it with the current frame
                if prevFrame is None:
                    prevFrame = currFrame.copy()
                    continue
                # Check the cosine similarity of the frame with the next one
                isShot = calculateCosineSimilarityFrames(prevFrame, currFrame, threshold)
                if isShot:
                    # Save the current frame as a shot
                    shotPath = os.path.join(outputDir, frameFile)
                    cv.imwrite(shotPath, currFrame)
                    totalShots += 1
            # Inform the user
            elapsedTime = "{:.2f}".format(time.time() - startTime)
            print(
                f"- Extracted {totalShots} shots from {totalFrames} frames of '{folderName}' in {elapsedTime} seconds!"
            )
        except Exception as error:
            print(
                f"- [Error] Error while picking the shots of '{folderName}' in '{framesDir}': {str(error)}"
            )
            continue
