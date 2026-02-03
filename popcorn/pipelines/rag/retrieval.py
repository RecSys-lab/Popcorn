import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors


def buildItemMatrix(embedDict: dict) -> tuple:
    """
    Converts embedding dictionary to item matrix and list of item IDs.

    Parameters
    ----------
    embedDict: dict
        Dictionary mapping item IDs to their embedding vectors.
    Returns
    -------
    tuple
        A tuple containing:
        - itemMatrix: 2D numpy array of shape (num_items, embedding_dim)
        - itemIds: List of item IDs corresponding to the rows in itemMatrix
    """
    # Check if embedDict is empty
    if not embedDict:
        print(
            "- [Warn] The embedding dictionary is empty! Returning empty matrix and list."
        )
        return np.array([]), []
    # Build item matrix and list of item IDs
    itemIds = list(embedDict.keys())
    itemMatrix = np.array([embedDict[i] for i in itemIds], dtype="float32")
    # Return the item matrix and item IDs
    return itemMatrix, itemIds


def retrieveTopNItems(
    userEmbedding: np.ndarray, itemMatrix: np.ndarray, itemIds: list, N: int = 5
):
    """
    Uses scikit-learn's NearestNeighbors for retrieval with cosine distance.
    Imputes missing values using KNNImputer (n_neighbors=3) before fitting.

    Parameters
    ----------
    userEmbedding: np.ndarray
        The embedding vector of the user (1D numpy array).
    itemMatrix: np.ndarray
        2D numpy array where each row is an item's embedding vector.
    itemIds: list
        List of item IDs corresponding to the rows in itemMatrix.
    N: int
        Number of top items to retrieve.

    Returns
    -------
    items: list
        A list of tuples for the top-N nearest items
    """
    # Variables
    items = []
    # Check for empty inputs
    if itemMatrix.size == 0 or len(itemIds) == 0:
        print(
            "- [Warn] The item matrix or item IDs list is empty! Returning empty results."
        )
        return []
    if userEmbedding.size == 0:
        print("- [Warn] The user embedding is empty! Returning empty results.")
        return []
    # KNN-based imputation of NaN values in item_matrix and user_emb
    imputer = KNNImputer(n_neighbors=3)
    # Fit on the item_matrix (which can contain multiple items)
    itemMatrixImputed = imputer.fit_transform(itemMatrix)
    # Transform user_emb (reshape to 2D first so imputer can handle a single row)
    userEmbImputed = imputer.transform(userEmbedding.reshape(1, -1))[0]
    # Fit NearestNeighbors on the imputed item_matrix
    nn = NearestNeighbors(n_neighbors=N, metric="cosine")
    nn.fit(itemMatrixImputed)
    # Retrieve neighbors for userEmbImputed
    distances, indices = nn.kneighbors([userEmbImputed])
    # Build and return sorted results [(item_id, distance), ...]
    results = [(itemIds[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    # Limit to top-N results
    items = sorted(results, key=lambda x: x[1])[:N]
    return items
