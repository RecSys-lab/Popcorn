import pandas as pd
from popcorn.datasets.ml_thumbnail.utils import (
    MAX_PARTS,
    isValidPart,
    isValidVariant,
    EMBEDDINGS_URL,
    SUPPORTED_VARIANTS,
)


def loadMovieLensThumbnailEmbeddings(partId: int, variant: str) -> pd.DataFrame:
    """
    Load MovieLens thumbnail embeddings based on the given part ID and variant (VLM).

    Parameters
    ----------
    partId: int
        The part ID to load
    variant: str
        The variant to load (e.g., "dino-v2", "clip")

    Returns
    -------
    embeddings: pd.DataFrame
        The loaded embeddings in a DataFrame format
    """
    # Variables
    embeddings = pd.DataFrame()
    # Argument validation
    if not isValidPart(partId):
        print(f"- [Error] Invalid part ID '{partId}'! Exiting...")
        return embeddings
    if not isValidVariant(variant) or variant == "raw_frame":
        print(f"- [Error] Invalid variant '{variant}'! Exiting...")
        return embeddings
    # Create the URL to achieve the CSV file
    url = EMBEDDINGS_URL.format(part_id=partId, variant=variant)
    print(f"- Fetching embeddings from '{url}' ...")
    try:
        embeddings = pd.read_csv(url)
        print(f"- {len(embeddings)} thumbnail embeddings loaded successfully!")
    except Exception as e:
        print(f"- [Error] Error loading embeddings from '{url}': {e}")
    return embeddings


def loadAllMovieLensThumbnailEmbeddings(variant: str) -> pd.DataFrame:
    """
    Load all MovieLens thumbnail embeddings.

    Parameters
    ----------
    variant: str
        The variant to load (e.g., "dino-v2", "clip")

    Returns
    -------
    allEmbeddings: pd.DataFrame
        The loaded embeddings in a DataFrame format
    """
    # Variables
    allEmbeddings = pd.DataFrame()
    # Argument validation
    if not isValidVariant(variant) or variant == "raw_frame":
        print(f"- [Error] Invalid variant '{variant}'! Exiting...")
        return pd.DataFrame()
    # Load embeddings for all valid variants
    for partId in range(1, MAX_PARTS + 1):
        embeddings = loadMovieLensThumbnailEmbeddings(partId, variant)
        if not embeddings.empty:
            allEmbeddings = pd.concat([allEmbeddings, embeddings], ignore_index=True)
    # Print the shape of the loaded embeddings
    print(
        f"- Loaded all thumbnail embeddings (total: {len(allEmbeddings)})! Sample:\n{allEmbeddings.head(3)}"
    )
    return allEmbeddings
