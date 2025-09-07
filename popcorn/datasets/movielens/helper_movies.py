import pandas as pd
from popcorn.datasets.movielens.utils import allGenres


def filterMoviesByGenre(itemsDF: pd.DataFrame, genre: str):
    """
    Fetches movies by a given genre.

    Parameters:
    ----------
    itemsDF: pd.DataFrame
        The DataFrame containing the movie data.
    genre: str
        The genre to filter the movies by.

    Returns:
    -------
    itemsDF_filtered: pd.DataFrame
        The DataFrame containing the movies of the given genre.
    """
    # Check if input arguments are valid
    if itemsDF is None or itemsDF.empty:
        print(
            "- [Error] The input DataFrame is empty or None. Returning the original DataFrame ..."
        )
        return itemsDF
    if not isinstance(genre, str) or not genre:
        print(
            "- [Error] The genre must be a non-empty string. Returning the original DataFrame ..."
        )
        return itemsDF
    # Check if the genre is valid
    if genre not in allGenres:
        print(f"- [Error] Genre '{genre}' is not valid. Valid genres are: {allGenres}")
        return itemsDF
    # Filter the DataFrame by the given genre
    print(f"- Filtering {len(itemsDF)} movies by genre '{genre}' ...")
    itemsDF_filtered = itemsDF[itemsDF["genres"].apply(lambda gList: genre in gList)]
    print(f"- Kept {len(itemsDF_filtered)} movies with genre '{genre}'.")
    # Return the filtered DataFrame
    return itemsDF_filtered


def filterMoviesWithMainGenres(dataFrame: pd.DataFrame):
    """
    Filters movies by the main genres.

    Parameters:
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing the movie data.

    Returns:
    -------
    mainGenresMovies: pd.DataFrame
        The DataFrame containing the movies of the main genres.
    """
    # Filter the DataFrame by the main genres
    mainGenresMovies = dataFrame[dataFrame["genres"].str.contains("|".join(mainGenres))]
    # Return the filtered DataFrame
    return mainGenresMovies


def augmentMoviesDFWithBinarizedGenres(
    moviesDataFrame: pd.DataFrame, binarizedGenresDataFrame: pd.DataFrame
):
    """
    Augments the movies DataFrame with binarized genres.

    Parameters:
    ----------
    moviesDataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    binarizedGenresDataFrame: pd.DataFrame
        The DataFrame containing the binarized genres.

    Returns:
    -------
    augmentedMoviesDataFrame: pd.DataFrame
        The augmented DataFrame containing the movie data with binarized genres.
    """
    # Merge the movies DataFrame with the binarized genres DataFrame
    augmentedMoviesDataFrame = pd.merge(
        moviesDataFrame, binarizedGenresDataFrame, on="movieId"
    )
    # Return the augmented DataFrame
    return augmentedMoviesDataFrame
