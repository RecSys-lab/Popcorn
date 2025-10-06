#!/usr/bin/env python3

from popcorn.pipelines.shots.utils import initFramesFoldersForShotDetection
from popcorn.pipelines.shots.utils import initFeaturesFoldersForShotDetection
from popcorn.pipelines.shots.shotDetection import extractShotsFromMovieFrames, extractShotsFromMovieFeatures


def runShotDetectionFromFrames(configs: dict):
    """
    Runs the shot detection pipeline from the movie frames

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    """
    print("Running the pipeline for shot detection from given movie frames ...")
    # Pre-check the input directory
    movieFramesPaths = initFramesFoldersForShotDetection(configs)
    if not movieFramesPaths:
        return
    # Extract shots from the fetched frames
    extractShotsFromMovieFrames(configs, movieFramesPaths)

def runShotDetectionFromFeatures(configs: dict):
    """
    Runs the shot detection pipeline from the extracted movie features

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    """
    print("Running the pipeline for shot detection from extracted movie features ...")
    # Pre-check the input directory
    movieFeaturesPaths = initFeaturesFoldersForShotDetection(configs)
    if not movieFeaturesPaths:
        return
    # Extract shots from the fetched features
    extractShotsFromMovieFeatures(configs, movieFeaturesPaths)