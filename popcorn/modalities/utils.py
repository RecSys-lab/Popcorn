import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from popcorn.core.utils import parseEmbedding


def applyPCA(
    textualDF: pd.DataFrame, visualDF: pd.DataFrame, componentCount: int = 128
):
    """
    Concatenates textual and visual embeddings (both dataFrames), then reduces them with PCA.

    Parameters
    ----------
    textualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_textual].
    visualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_visual].
    componentCount: int, optional
        Number of components to keep after PCA, by default 128.

    Returns
    -------
    fusedDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings] for fused embeddings.
    """
    # Variables
    mergedDF = pd.merge(textualDF, visualDF, on="itemId", suffixes=("_text", "_vis"))

    # Parse string or list to np.array
    visualEmbeddings = mergedDF["embeddings_vis"].apply(parseEmbedding).values
    textualEmbeddings = mergedDF["embeddings_text"].apply(parseEmbedding).values

    # Create final arrays
    X_text = np.stack(textualEmbeddings, axis=0)
    X_vis = np.stack(visualEmbeddings, axis=0)

    # Simple concatenation
    X_concat = np.hstack((X_text, X_vis))

    # Apply PCA
    pca = PCA(n_components=componentCount)
    X_pca = pca.fit_transform(X_concat)

    # Store back
    fusedList = []
    for i, item_id in enumerate(mergedDF["itemId"]):
        fusedVec = X_pca[i]
        fusedList.append((item_id, fusedVec))
    fusedDF = pd.DataFrame(fusedList, columns=["itemId", "embeddings"])

    # Convert embeddings to string (optional)
    fusedDF["embeddings"] = fusedDF["embeddings"].apply(
        lambda arr: " ".join(str(x) for x in arr)
    )

    # Return
    return fusedDF


def applyCCA(
    textualDF: pd.DataFrame, visualDF: pd.DataFrame, componentCount: int = 64
):
    """
    Uses Canonical Correlation Analysis (CCA) to fuse textual and visual embeddings (both dataFrames).
    The shared latent space (the correlated components) are used for fusion.

    Parameters
    ----------
    textualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_textual].
    visualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_visual].
    componentCount: int, optional
        Number of components to keep after CCA, by default 64.

    Returns
    -------
    fusedDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings] for fused embeddings.
    """
    # Variables
    mergedDF = pd.merge(textualDF, visualDF, on="itemId", suffixes=("_text", "_vis"))

    # Parse string or list to np.array
    textualEmbeddings = mergedDF["embeddings_text"].apply(parseEmbedding).values
    visualEmbeddings = mergedDF["embeddings_vis"].apply(parseEmbedding).values

    # Create final arrays
    X_text = np.stack(textualEmbeddings, axis=0)
    X_vis = np.stack(visualEmbeddings, axis=0)

    # Fit CCA
    cca = CCA(n_components=componentCount)
    X_text_c, X_vis_c = cca.fit_transform(X_text, X_vis)

    # Option 1: we can simply concatenate the two canonical subspaces.
    X_fused = np.hstack([X_text_c, X_vis_c])

    # Or Option 2: use just one side, e.g. X_text_c, depending on your approach.
    fusedList = []
    for i, item_id in enumerate(mergedDF["itemId"]):
        fusedVec = X_fused[i]
        fusedList.append((item_id, fusedVec))
    fusedDF = pd.DataFrame(fusedList, columns=["itemId", "embeddings"])

    # Convert embeddings to string (optional)
    fusedDF["embeddings"] = fusedDF["embeddings"].apply(
        lambda arr: " ".join(str(x) for x in arr)
    )

    # Return
    return fusedDF


def applyAverage(textualDF: pd.DataFrame, visualDF: pd.DataFrame):
    """
    Uses a simple average fusion approach to combine textual and visual embeddings.

    Parameters
    ----------
    textualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_textual].
    visualDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings_visual].

    Returns
    -------
    fusedDF: pd.DataFrame
        DataFrame with columns [itemId, embeddings] for fused embeddings.
    """
    # Variables
    mergedDF = pd.merge(textualDF, visualDF, on="itemId", suffixes=("_text", "_vis"))

    # Parse string or list to np.array
    visualEmbeddings = mergedDF["embeddings_vis"].apply(parseEmbedding).values
    textualEmbeddings = mergedDF["embeddings_text"].apply(parseEmbedding).values

    fusedList = []
    for i, item_id in enumerate(mergedDF["itemId"]):
        v_text = textualEmbeddings[i]
        v_vis = visualEmbeddings[i]

        # Check dimension match; if mismatch, handle accordingly
        if len(v_text) == len(v_vis):
            fused_vec = (v_text + v_vis) / 2.0
        else:
            # Fallback to just concatenation or skip
            fused_vec = np.hstack([v_text, v_vis])

        fusedList.append((item_id, fused_vec))
    fusedDF = pd.DataFrame(fusedList, columns=["itemId", "embeddings"])

    # Convert embeddings to string (optional)
    fusedDF["embeddings"] = fusedDF["embeddings"].apply(
        lambda arr: " ".join(str(x) for x in arr)
    )

    # Return
    return fusedDF
