#!/usr/bin/env python3

import pandas as pd

def mergeMainGenreMoviesDFWithRatingsDF(moviesDataFrame: pd.DataFrame, ratingsDataFrame: pd.DataFrame):
    """
    Merges the movies DataFrame with the ratings DataFrame.

    Parameters:
    ----------
    moviesDataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    ratingsDataFrame: pd.DataFrame
        The DataFrame containing the ratings data.

    Returns:
    -------
    mergedDataFrame: pd.DataFrame
        The merged DataFrame containing the movies and ratings data.
    """
    # Merge the movies and ratings DataFrames
    mergedDataFrame = pd.merge(moviesDataFrame, ratingsDataFrame, on='movieId')
    # Remove rows where isAction or isComedy or isDrama or isHorror is 0
    mergedDataFrame = mergedDataFrame[(mergedDataFrame['isAction'] == 1) | (mergedDataFrame['isComedy'] == 1) 
                                      | (mergedDataFrame['isDrama'] == 1) | (mergedDataFrame['isHorror'] == 1)]
    # Return the merged DataFrame
    return mergedDataFrame