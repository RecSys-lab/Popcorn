import pandas as pd

# Supported dataset variants for thumbnail fetching
SUPPORTED_DATASET_VARIANTS = ["dummy", "mmtf", "popcorn"]

# Supported poster sizes for TMDB images
SUPPORTED_POSTER_SIZES = ["w92", "w154", "w185", "w342", "w500", "w780", "original"]

DEFAULT_OUTPUT_DIR_PREFIX = f"thumbnails_"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_FIND_URL = "https://api.themoviedb.org/3/find/{imdb_id}"

HEADERS = {
    "User-Agent": "movie-thumbnail-downloader/1.0",
    "Accept": "application/json",
}


def movielensToIMDB(mlId: int, linksDF: pd.DataFrame) -> str | None:
    """
    Convert a MovieLens ID to an IMDb ID string (e.g. 'tt0111161').

    Parameters
    ----------
    mlId: int
        The MovieLens ID to convert.
    linksDF: pd.DataFrame
        DataFrame containing the mapping from MovieLens IDs to IMDb IDs.

    Returns
    -------
    str | None
        IMDb ID string, or None if not found.
    """
    # Find the row in linksDF corresponding to the given MovieLens ID
    row = linksDF[linksDF["movieId"] == mlId]
    if row.empty:
        return None
    # Create the IMDb ID
    imdbNum = str(row.iloc[0]["imdbId"]).zfill(7)
    # Return the resulting IMDb ID string
    return f"tt{imdbNum}"
