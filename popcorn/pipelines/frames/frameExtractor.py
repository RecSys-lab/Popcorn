import os
import time
import cv2 as cv
from popcorn.pipelines.frames.utils import initFramesFolder, resizeFrame

def extractMovieFrames(configs: dict, fetchedMoviesPaths: list):
    """
    Extracts frames from the given set of fetched movies

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    fetchedMoviesPaths :list
        The list of fetched movie paths
    """
    print("Extracting frames from the given set of movie videos ...")
    # Iterate on all video files in the given directory
    for videoFile in fetchedMoviesPaths:
        # Preparing the output frames directory
        outputDir = initFramesFolder(videoFile, configs['frames_path'])
        if not outputDir:
            continue
        # Capturing video
        try:
            # Variables
            frameIndex = 0
            frameCounter = 0
            frameIndexToPick = 0
            startTime = time.time()
            frequency = configs['frequency']
            videoName = os.path.basename(outputDir)
            savedFrameFormat = configs['output_format']
            modelInputSize = configs['model_input_size']
            # Extract frames from the video
            capturedVideo = cv.VideoCapture(videoFile)
            # Get the frame rate and compare it with the given frame rate
            frameRate = int(capturedVideo.get(cv.CAP_PROP_FPS))
            if frequency > frameRate:
                frequency = frameRate
            # Start extracting frames
            print(f'- Extracting frames of {videoName} with the frequency of {frequency} fps ...')
            # Set a frame to pick
            framePickingRate = int(frameRate / frequency)
            print(f'--- Frame rate: {frameRate} fps, Frequency: {frequency} fps, Frame picking rate: {framePickingRate}')
            while True:
                success, frame = capturedVideo.read()
                # If the end of the video is reached
                if not success:
                    # Finished extracting frames
                    elapsedTime = '{:.2f}'.format(time.time() - startTime)
                    print(f'- Extraction finished for {videoName} (took {elapsedTime} seconds to extract {frameCounter} frames, saved in {outputDir})!\n')
                    break
                # Otherwise, continue extracting frames
                # Pick only the frames with the given frequency
                if (frameIndex == frameIndexToPick):
                    # Save frame file name as: frame1 --> frame0000001
                    fileName = '{0:07d}'.format(frameCounter)
                    # Showing progress every 1000 frames
                    if (frameCounter > 0 and frameCounter % 100 == 0):
                        elapsedTime = '{:.2f}'.format(time.time() - startTime)
                        print(
                            f'--- Processing frame #{frameIndex} of the video (took {elapsedTime} seconds to extract {frameCounter} frames so far) ...')
                    # Resizing the image, while preserving its aspect-ratio
                    pFrame = resizeFrame(frame, modelInputSize)
                    # Save the frame as a file
                    cv.imwrite(
                        f"{outputDir}/frame{fileName}.{savedFrameFormat}", pFrame)
                    # Increment the frame counter and set the next frame to pick
                    frameCounter += 1
                    frameIndexToPick += framePickingRate
                # Increment the frame index
                frameIndex += 1
        except cv.error as openCVError:
            print(f'Error while processing video frames: {str(openCVError)}')
        except Exception as otherError:
            print(f'Error while processing video frames: {str(otherError)}')