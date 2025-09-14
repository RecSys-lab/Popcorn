from popcorn.utils import loadJsonFromUrl
from popcorn.datasets.popcorn.utils import (
    RAW_DATA_URL,
    isValidCNN,
    isValidEmbeddingSource,
)


def generatePacketUrl(embedding: str, cnn: str, movieId: int, packetId: int) -> str:
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
    packetId: int
        The ID of the packet (1, 2, 3, ...).

    Returns
    -------
    packetUrl: str
        The generated address of the embedding packet file.
    """
    # Variables
    packetUrl = ""
    # Check arguments
    isValidCnn = isValidCNN(cnn)
    isValidEmbedding = isValidEmbeddingSource(embedding)
    if not isValidCnn or not isValidEmbedding:
        return packetUrl
    # Format movie ID and packet ID
    movieId = f"{int(movieId):010d}"
    packetId = str(packetId).zfill(4)
    # Create address
    packetUrl = (
        RAW_DATA_URL
        + f"{embedding}/{cnn}/{movieId}/packet"
        + str(packetId).zfill(4)
        + ".json"
    )
    # Return
    return packetUrl


def fetchAllPackets(embedding: str, cnn: str, movieId: int) -> list:
    """
    Fetches all packets of a movie from the given embedding type and CNN model.

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
    movieEmbeddings: list
        A list of all movie embeddings fetched from the packets.
    """
    # Variables
    counter = 0
    movieEmbeddings = []
    print(f"- Fetching all packets of the movie #{movieId} ({embedding}, {cnn}) ...")
    # Check arguments
    isValidCnn = isValidCNN(cnn)
    isValidEmbedding = isValidEmbeddingSource(embedding)
    if not isValidCnn or not isValidEmbedding:
        return movieEmbeddings
    # Loop over all possible files
    while True:
        counter += 1
        # Generate packet URL
        packetUrl = generatePacketUrl(embedding, cnn, movieId, counter)
        # Fetch JSON data
        jsonData = loadJsonFromUrl(packetUrl)
        if jsonData:
            print(f"- Fetched JSON data from the URL '{packetUrl}'!")
            movieEmbeddings += jsonData
        else:
            print(
                f"- [Error] No JSON data found at the URL '{packetUrl}'! Stopping ..."
            )
            break
    # Return
    return movieEmbeddings
