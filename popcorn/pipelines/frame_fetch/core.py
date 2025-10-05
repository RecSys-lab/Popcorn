import os
import time
import cv2 as cv
from popcorn.pipelines.utils import frameImageFormats
from popcorn.pipelines.frame_fetch.utils import (
    fetchMovieVideoFiles,
    initFramesPath,
    resizeFrame
)


def extractMovieFrames(configs: dict):
    """
    Extracts frames from the given set of fetched movies

    Parameters
    ----------
    configs: dict
        The configurations dictionary of the framework.
    """
    # Variables
    pipelineName = configs["pipelines"]["frame_extractor"]["name"]
    frequency = configs["pipelines"]["frame_extractor"]["frequency"]
    frameWidth = configs["pipelines"]["frame_extractor"]["frame_width"]
    outputFormat = configs["pipelines"]["frame_extractor"]["frame_format"]
    framesRootPath = configs["pipelines"]["frame_extractor"]["frames_path"]
    print(f"- Starting the '{pipelineName}' pipeline ...")
    # Check the validity of arguments and configurations
    if frequency <= 0 or frequency > 25:
        print(
            f"- [Warn] The given frequency '{frequency}' is not valid (0 < freq <= 25)! Setting it to 1 fps."
        )
        frequency = 1
    if frameWidth <= 0 or frameWidth > 1920:
        print(
            f"- [Warn] The given frame width '{frameWidth}' is not valid (0 < width <= 1920)! Setting it to 420 px."
        )
        frameWidth = 420
    if outputFormat not in frameImageFormats:
        print(
            f"- [Warn] The given output format '{outputFormat}' is not valid (should be one of {frameImageFormats})! Setting it to 'jpg'."
        )
        outputFormat = "jpg"
    # Fetch the list of movie video files (if any)
    fetchedVideos = fetchMovieVideoFiles(configs)
    if len(fetchedVideos) == 0 or not fetchedVideos:
        print(f"- [Warn] No valid video files found! Exiting ...")
        return
    print("- Extracting frames from the given set of movie videos ...")
    # Iterate on all video files in the given directory
    for videoFilePath in fetchedVideos[:3]:
        # Preparing the output frames directory
        framesDir = initFramesPath(videoFilePath, framesRootPath)
        if framesDir == "":
            continue
        # Capturing video
        try:
            # Variables
            frameIndex = 0
            frameCounter = 0
            frameIndexToPick = 0
            startTime = time.time()
            videoName = os.path.basename(framesDir)
            # Extract frames from the video
            capturedVideo = cv.VideoCapture(videoFilePath)
            # Get the frame rate and compare it with the given frame rate
            frameRate = int(capturedVideo.get(cv.CAP_PROP_FPS))
            if frequency > frameRate:
                print(
                    f"-- [Warn] The given frequency '{frequency}' is higher than the video frame rate '{frameRate}'! Setting it to '{frameRate}' fps."
                )
                frequency = frameRate
            # Start extracting frames
            print(
                f"-- Extracting frames of '{videoName}' with the frequency of '{frequency}' fps ..."
            )
            # Set a frame to pick
            framePickingRate = int(frameRate / frequency)
            while True:
                success, frame = capturedVideo.read()
                # If the end of the video is reached
                if not success:
                    # Finished extracting frames
                    elapsedTime = "{:.2f}".format(time.time() - startTime)
                    print(
                        f"-- Extraction finished for '{videoName}' (took '{elapsedTime}' seconds to extract '{frameCounter}' frames, saved in '{framesDir}')!"
                    )
                    break
                # Otherwise, continue extracting frames
                # Pick only the frames with the given frequency
                if frameIndex == frameIndexToPick:
                    # Save frame file name as: frame1 --> frame0000001
                    fileName = "{0:07d}".format(frameCounter)
                    # Showing progress every 1000 frames
                    if frameCounter > 0 and frameCounter % 100 == 0:
                        elapsedTime = "{:.2f}".format(time.time() - startTime)
                        print(
                            f"--- Processing frame #{frameIndex} of the video (took '{elapsedTime}' seconds to extract '{frameCounter}' frames so far) ..."
                        )
                    # Resizing the image, while preserving its aspect-ratio
                    pFrame = resizeFrame(frame, frameWidth)
                    # Save the frame as a file
                    cv.imwrite(
                        f"{framesDir}/frame{fileName}.{outputFormat}", pFrame
                    )
                    # Increment the frame counter and set the next frame to pick
                    frameCounter += 1
                    frameIndexToPick += framePickingRate
                # Increment the frame index
                frameIndex += 1
        except cv.error as openCVError:
            print(f"- [Error] Error while processing video frames: {str(openCVError)}")
        except Exception as otherError:
            print(f"- [Error] Error while processing video frames: {str(otherError)}")
