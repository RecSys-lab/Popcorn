import pandas as pd
from popcorn.utils import loadJsonFromUrl
from popcorn.datasets.popcorn.helper_metadata import fetchAllMovieIds
from popcorn.datasets.popcorn.utils import (
    RAW_DATA_URL,
    METADATA_URL,
    isValidCNN,
    isValidAggEmbeddingSource,
)


def generateMovieAggEmbeddingUrl(embedding: str, cnn: str, movieId: int) -> str:
    """
    Generates the address of an embedding packet file based on the given parameters.

    Parameters
    ----------
    embedding: str
        The embedding type (e.g., "full_movies").
    cnn: str
        The CNN model used for feature extraction (e.g., "incp3").
    movieId: int
        The ID of the movie, aligned with MovieLens 25M dataset.

    Returns
    -------
    aggEmbeddingUrl: str
        The generated address of the embedding packet file.
    """
    # Variables
    aggEmbeddingUrl = ""
    # Check arguments
    isValidCnn = isValidCNN(cnn)
    isValidEmbedding = isValidAggEmbeddingSource(embedding)
    if not isValidCnn or not isValidEmbedding:
        return aggEmbeddingUrl
    # Format movie ID
    movieId = f"{int(movieId):010d}"
    # Create address
    aggEmbeddingUrl = RAW_DATA_URL + f"{embedding}/{cnn}/{movieId}" + ".json"
    # Return
    return aggEmbeddingUrl


def generateMoviesAggEmbeddingUrls(
    embeddings: list, cnns: list, movieIds: list
) -> dict:
    """
    Generates all addresses of aggregated embedding packet files based on the given parameters.

    Parameters
    ----------
    embeddings: list
        A list of embedding types (e.g., ["full_movies"]).
    cnns: list
        A list of CNN models (e.g., ["incp3", "vgg19"]).
    movieIds: list
        A list of movie IDs (e.g., [1, 2, 3]).

    Returns
    -------
    aggEmbeddingUrlDict: dict
        A dictionary containing lists of generated addresses of the embedding packet files.
    """
    # Variables
    aggEmbeddingUrlDict = {}
    # Check arguments
    if not embeddings or not cnns or not movieIds:
        print(f"- [Error] No valid input lists provided! Stopping ...")
        return aggEmbeddingUrlDict
    isValidCnn = all(isValidCNN(cnn) for cnn in cnns)
    isValidEmbedding = all(isValidAggEmbeddingSource(emb) for emb in embeddings)
    if not isValidCnn or not isValidEmbedding:
        print(f"- [Error] Invalid CNN or embedding type found! Stopping ...")
        return aggEmbeddingUrlDict
    # Fill the dictionary
    for embedding in embeddings:
        aggEmbeddingUrlDict[embedding] = {}
        for cnn in cnns:
            aggEmbeddingUrlDict[embedding][cnn] = []
    # Loop over all movies
    for movieId in movieIds:
        # Format movie ID
        if not isinstance(movieId, int):
            movieId = int(movieId)
        movieId = f"{int(movieId):010d}"
        # Loop over all feature models
        for cnn in cnns:
            # Loop over all aggregated feature sources
            for embedding in embeddings:
                # Generate address
                address = RAW_DATA_URL + f"{embedding}/{cnn}/{movieId}.json"
                aggEmbeddingUrlDict[embedding][cnn].append(address)
    # Return
    return aggEmbeddingUrlDict


def generateAllAggEmbeddingUrls(configs: dict) -> dict:
    """
    Fetches all aggregated features of movies from the dataset based on the given configuration.

    Parameters:
    -----------
    configs: dict
        A dictionary containing the dataset configuration.

    """
    # Variables
    aggEmbeddingUrlDict = {}
    # Read configurations
    cnns = configs["datasets"]["multimodal"]["popcorn"]["cnns"]
    embeddings = configs["datasets"]["multimodal"]["popcorn"]["agg_embedding_sources"]
    # Fetch JSON metadata from the URL
    print(f"- Fetching URL from '{METADATA_URL}' ...")
    jsonData = loadJsonFromUrl(METADATA_URL)
    if jsonData is None:
        print("- Error in loading the Popcorn dataset metadata! Exiting ...")
        return aggEmbeddingUrlDict
    # Fetch all movie IDs
    print(f"- Fetching all movie IDs ...")
    movieIds = fetchAllMovieIds(jsonData)
    # Generating a list of addresses to fetch the aggregated features
    print(
        f"- Generating a list of addresses to fetch the aggregated features of all movies (CNNs: {cnns}, Embeddings: {embeddings}) ..."
    )
    aggEmbeddingUrlDict = generateMoviesAggEmbeddingUrls(embeddings, cnns, movieIds)
    if not aggEmbeddingUrlDict:
        print("- Error in generating the addresses! Exiting ...")
        return {}
    # Count all members of the generated addresses
    count = (
        len(aggEmbeddingUrlDict)
        * len(aggEmbeddingUrlDict[embeddings[0]])
        * len(aggEmbeddingUrlDict[embeddings[0]][cnns[0]])
    )
    print(
        f"- Generated {count} aggregated feature addresses, e.g., {aggEmbeddingUrlDict['full_movies_agg']['incp3'][0]} ..."
    )
    # Return
    return aggEmbeddingUrlDict


def fetchMovieAggEmbeddings(embedding: str, cnn: str, movieId: int) -> list:
    """
    Fetches all aggregated features of a movie from the given embedding type and CNN model.

    Parameters
    ----------
    embedding: str
        The embedding type (e.g., "full_movies").
    cnn: str
        The CNN model used for feature extraction (e.g., "incp3").
    movieId: int
        The ID of the movie, aligned with MovieLens 25M dataset.

    Returns
    -------
    aggEmbeddingList: list
        A list of all aggregated features of the movie.
    """
    # Variables
    aggEmbeddingList = []
    # Check arguments
    isValidCnn = isValidCNN(cnn)
    isValidEmbedding = isValidAggEmbeddingSource(embedding)
    if not isValidCnn or not isValidEmbedding:
        return aggEmbeddingList
    # Generate packet address
    aggEmbeddingUrl = generateMovieAggEmbeddingUrl(embedding, cnn, movieId)
    # Check if address is valid
    if aggEmbeddingUrl == "":
        print(f"- [Error] No valid address generated! Stopping ...")
        return aggEmbeddingList
    # Fetch JSON data
    aggEmbeddingList = loadJsonFromUrl(aggEmbeddingUrl)
    # Return
    return aggEmbeddingList


def loadAggEmbeddings(aggEmbeddingUrlList: list) -> tuple:
    """
    Loads aggregated features from a list of URLs into two DataFrames (Max and Mean).

    Parameters
    ----------
    aggEmbeddingUrlList: list
        A list of URLs pointing to the aggregated feature JSON files.

    Returns
    -------
    dfAggEmbedsMax: pd.DataFrame
        A DataFrame containing the 'Max' aggregated features.
    dfAggEmbedsMean: pd.DataFrame
        A DataFrame containing the 'Mean' aggregated features.
    """
    # Variables
    counter = 0
    dfAggEmbedsMax = pd.DataFrame()
    dfAggEmbedsMean = pd.DataFrame()
    # Check input
    if not aggEmbeddingUrlList or not isinstance(aggEmbeddingUrlList, list):
        print(f"- [Error] No valid list of addresses provided! Stopping ...")
        return pd.DataFrame(), pd.DataFrame()
    # Loop over all addresses
    for url in aggEmbeddingUrlList:
        # Variables
        itemId = url.split("/")[-1].split(".")[0]
        # Fetch the JSON data from the URL
        jsonData = loadJsonFromUrl(url)
        aggFeatMax, aggFeatMean = jsonData[0]["Max"], jsonData[0]["Mean"]
        # Convert the list to a string, like "0.1,0.2,0.3"
        aggFeatMax = ",".join(map(str, aggFeatMax))
        aggFeatMean = ",".join(map(str, aggFeatMean))
        # Append to the DataFrames
        dfAggEmbedsMax = pd.concat(
            [
                dfAggEmbedsMax,
                pd.DataFrame([{"item_id": int(itemId), "embedding": aggFeatMax}]),
            ],
            ignore_index=True,
        )
        dfAggEmbedsMean = pd.concat(
            [
                dfAggEmbedsMean,
                pd.DataFrame([{"item_id": int(itemId), "embedding": aggFeatMean}]),
            ],
            ignore_index=True,
        )
        # Better logging for the user
        counter += 1
        if counter % 2 == 0:
            print(
                f"- Loading aggregated features ({int(counter / len(aggEmbeddingUrlList) * 100)}%) ..."
            )
    # Return
    return dfAggEmbedsMax, dfAggEmbedsMean
