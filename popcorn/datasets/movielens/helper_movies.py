import pandas as pd
from popcorn.datasets.movielens.utils import allGenres, mainGenres


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


def filterMoviesWithMainGenres(itemsDF: pd.DataFrame):
    """
    Filters movies to keep only those belonging to the main genres.

    Parameters:
    ----------
    itemsDF: pd.DataFrame
        The DataFrame containing the movie data.

    Returns:
    -------
    mainGenresMovies: pd.DataFrame
        The DataFrame containing the movies of the main genres.
    """
    # Check if input argument is valid
    if itemsDF is None or itemsDF.empty:
        print(
            "- [Error] The input DataFrame is empty or None. Returning the original DataFrame ..."
        )
        return itemsDF
    # Filter the DataFrame by the main genres
    print(
        f"- Filtering {len(itemsDF)} movies containing the main genres '{mainGenres}' ..."
    )
    # Filtration
    itemsDF_filtered = itemsDF[
        itemsDF["genres"].apply(
            lambda gList: (
                any(genre in gList for genre in mainGenres)
                if isinstance(gList, list)
                else False
            )
        )
    ]
    print(f"- Kept {len(itemsDF_filtered)} movies containing the main genres.")
    # Return the filtered DataFrame
    return itemsDF_filtered


def augmentMoviesWithBinarizedGenres(
    itemsDF: pd.DataFrame, itemsDF_binGenre: pd.DataFrame
):
    """
    Augments the movies DataFrame with binarized genres.

    Parameters:
    ----------
    itemsDF: pd.DataFrame
        The DataFrame containing the movie data.
    itemsDF_binGenre: pd.DataFrame
        The DataFrame containing the binarized genres.

    Returns:
    -------
    itemsDF_augmented: pd.DataFrame
        The augmented DataFrame containing the movie data with binarized genres.
    """
    # Check if input arguments are valid
    if (
        itemsDF is None
        or itemsDF.empty
        or itemsDF_binGenre is None
        or itemsDF_binGenre.empty
    ):
        print(
            "- [Error] The input movies DataFrame is empty or None. Returning the original DataFrame ..."
        )
        return itemsDF
    if itemsDF_binGenre.columns.tolist() != [
        "item_id",
        "isAction",
        "isComedy",
        "isDrama",
        "isHorror",
    ]:
        print(
            "- [Error] The binarized genres DataFrame must contain the columns "
            "['item_id', 'isAction', 'isComedy', 'isDrama', 'isHorror']. Returning the original DataFrame ..."
        )
        return itemsDF
    # Merge the movies DataFrame with the binarized genres DataFrame
    itemsDF_augmented = pd.merge(itemsDF, itemsDF_binGenre, on="item_id")
    # Return the augmented DataFrame
    return itemsDF_augmented
