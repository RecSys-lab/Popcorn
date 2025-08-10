import numpy as np

# Multimodal fusion variants
MULTI_VARIANTS = [
    ("concat", None),
    ("pca", 0.95),
    ("cca", 40),
]

def parseEmbedding(givenEmbedding):
    """
    Parses the embedding string or list into a numpy array. It can work with textual of visual embeddings.

    Parameters
    ----------
    givenEmbedding: str or list
        Given embedding in the form of a string or list.
    
    Returns
    -------
    np.ndarray
        Parsed embedding as a numpy array.
    """
    # Check the type of the given embedding
    if isinstance(givenEmbedding, str):
        arr = [float(x) for x in givenEmbedding.strip().split()]
        return np.array(arr, dtype=np.float32)
    elif isinstance(givenEmbedding, (list, np.ndarray)):
        return np.array(givenEmbedding, dtype=np.float32)
    else:
        return None