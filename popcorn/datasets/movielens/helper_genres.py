import os
import json
import pandas as pd
from popcorn.datasets.movielens.utils import mainGenres, allGenres


def normalizeGenres(value):
    """
    Normalize the genres value to a list of genres.
    This function avoids having inconsistant entries like below:
    - Row1: "['Crime', 'Thriller']"
    - Row2: ['Comedy']
    - Row3: "['Animation', ""Children's"", 'Comedy']"

    Parameters:
    ----------
    value: str or list
        The genres value to normalize.

    Returns:
    -------
    list
        A list of genres.
    """
    if isinstance(value, str):
        # Try to parse list-like strings safely
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            # fallback: split by common separators
            return [g.strip() for g in value.replace("|", ",").split(",") if g.strip()]
    elif isinstance(value, list):
        return value
    return []


def getGenreDict(itemsDF: pd.DataFrame, config: dict, saveOutput: bool = True) -> dict:
    """
    Get a dictionary mapping item IDs to their genres.

    Parameters:
    ----------
    itemsDF: pd.DataFrame
        The DataFrame containing the movie data (["item_id", "title", "genres"])
    config: dict
        The configuration dictionary loaded from the config.yml file.
    saveOutput: bool
        Whether to save the genres DataFrame as a CSV file.

    Returns:
    -------
    genreDict: dict
        A dictionary mapping item IDs to their genres.
    """
    # Variables
    genresDF = itemsDF[["item_id", "genres"]].copy()
    ROOT_PATH = config["general"]["root_path"]
    OUTPUT_PATH = config["general"]["output_path"]
    VERSION = config["datasets"]["unimodal"]["movielens"]["version"]
    # Normalize genres column
    genresDF["genres"] = genresDF["genres"].apply(normalizeGenres)
    # Create a dictionary mapping item_id to genres
    genresDict = dict(zip(genresDF.item_id, genresDF.genres))
    # Save the genres DataFrame as a CSV file
    if saveOutput:
        try:
            genresDFSaveFilename = f"item_genre_ml-{VERSION}.csv"
            genresDFSavePath = (
                os.path.join(ROOT_PATH, "outputs") if OUTPUT_PATH == "" else OUTPUT_PATH
            )
            print(
                f"\n- Preparing to save the genres DataFrame in '{genresDFSavePath}' ..."
            )
            # Create the output directory if it doesn't exist
            if not os.path.exists(genresDFSavePath):
                os.makedirs(genresDFSavePath)
            # Save the genres DataFrame to a CSV file
            genresDFSavePath = os.path.join(genresDFSavePath, genresDFSaveFilename)
            genresDF.to_csv(genresDFSavePath, index=False)
            print(f"- Genres DataFrame saved to '{genresDFSavePath}'!")
        except Exception as e:
            print(f"- Error in saving the genres DataFrame: {e}")
    else:
        print(f"- Skipping saving the genres DataFrame ...")
    # Return the genres dictionary
    return genresDict


def binarizeGenres(itemsDF: pd.DataFrame):
    """
    Receives the movies DataFrame and returns a DataFrame with separate columns
    for each main genre, indicating presence (1) or absence (0).
    [Note]: the main genres are ["Action", "Comedy", "Drama", "Horror"]

    Parameters:
    ----------
    itemsDF: pd.DataFrame
        The DataFrame containing the movie data.

    Returns:
    -------
    itemsDF_binGenre: pd.DataFrame
        The DataFrame containing the binarized genres.
    """
    # Variables
    itemsDF_binGenre = pd.DataFrame(
        columns=["item_id", "isAction", "isComedy", "isDrama", "isHorror"]
    )
    # Add the item_id column
    itemsDF_binGenre["item_id"] = itemsDF["item_id"]
    # Iterate over the main genres
    for genre in mainGenres:
        # Create a new column for each genre
        itemsDF_binGenre[f"is{genre}"] = itemsDF["genres"].apply(
            lambda gList: int(genre in gList) if isinstance(gList, list) else 0
        )
    # Return the binarized DataFrame
    return itemsDF_binGenre

def getMainGenres():
    """
    Returns the list of main genres.

    Returns:
    -------
    mainGenres: list
        A list of main genres.
    """
    return mainGenres

def getAllGenres():
    """
    Returns the list of all genres.

    Returns:
    -------
    allGenres: list
        A list of all genres.
    """
    return allGenres