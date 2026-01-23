import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from cornac.data import ImageModality, FeatureModality

SUPPORTED_MODALITIES = ["audio_mmtf", "visual_mmtf", "text_rag_plus", "visual_popcorn"]

SUPPORTED_FUSION_METHODS = ["concat", "cca", "pca"]

SUPPORTED_TOP_N = [2, 5, 10, 15, 20, 25, 30, 50]


def getImageModality(df: pd.DataFrame, col: str) -> ImageModality:
    """
    Create an ImageModality from a DataFrame column.
    [Note]: the IDs are taken from 'item_id' column.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the image data.
    col: str
        The column name in the DataFrame that contains image features.

    Returns
    -------
    ImageModality
        An ImageModality instance with features from the specified column.
    """
    return ImageModality(features=np.vstack(df[col]), ids=df.item_id, normalized=True)


def getFeatureModality(df: pd.DataFrame, col: str) -> FeatureModality:
    """
    Create a FeatureModality from a DataFrame column.
    [Note]: the IDs are taken from 'item_id' column.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the feature data.
    col: str
        The column name in the DataFrame that contains feature vectors.

    Returns
    -------
    FeatureModality
        A FeatureModality instance with features from the specified column.
    """
    return FeatureModality(features=np.vstack(df[col]), ids=df.item_id, normalized=True)


def applyPCAModality(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply PCA to the specified modality in the configuration.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the modality data.
    config: dict
        The configuration dictionary containing PCA parameters.

    Returns
    -------
    df: pd.DataFrame
        The DataFrame with the PCA-applied modality added.
    name: str
        The name of the new PCA modality column.
    """
    # Variables
    reg = config["modalities"]["fusion_methods"]["pca_reg"]
    ratio = config["modalities"]["fusion_methods"]["pca_variance"]
    print(f"- Applying PCA with variance ratio '{ratio}' ...")
    # Check ratio validity
    if not (0.0 < ratio < 1.0):
        print(
            f"- [Warn] PCA variance ratio must be between 0 and 1, but given '{ratio}'. Setting to 0.95 ..."
        )
        ratio = 0.95
    if not (0.0 <= reg <= 1.0):
        print(
            f"- [Warn] PCA regularization must be between 0 and 1, but given '{reg}'. Setting to 0.0 ..."
        )
        reg = 0.0
    # Set the name
    name = f"pca_{int(ratio * 100)}"
    # Apply PCA
    mat = StandardScaler().fit_transform(np.vstack(df["all"]))
    # Add ridge regularization: X^T X + λI
    mat_reg = np.vstack([mat, np.sqrt(reg) * np.eye(mat.shape[1])])
    # Fit and transform
    pca_result = PCA(ratio, random_state=42).fit_transform(mat_reg[:mat.shape[0]])
    df[name] = list(pca_result.astype(np.float32))
    # Log and return
    print(f"- Applied PCA-{int(ratio*100)} and generated dimensions {mat.shape[1]}!")
    return df, name


def applyCCAModality(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply CCA to the specified modality in the configuration.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the modality data.
    config: dict
        The configuration dictionary containing CCA parameters.

    Returns
    -------
    df: pd.DataFrame
        The DataFrame with the CCA-applied modality added.
    name: str
        The name of the new CCA modality column.
    """
    # Variables
    reg = config["modalities"]["fusion_methods"]["cca_reg"]
    comps = config["modalities"]["fusion_methods"]["cca_components"]
    print(f"- Applying CCA with components '{comps}' ...")
    # Check components validity
    if not (isinstance(comps, int) and comps > 0):
        print(
            f"- [Warn] CCA components must be a positive integer, but given '{comps}'. Setting to 40 ..."
        )
        comps = 40
    if not (0.0 <= reg <= 1.0):
        print(
            f"- [Warn] CCA regularization must be between 0 and 1, but given '{reg}'. Setting to 0.0 ..."
        )
        reg = 0.0
    # Set the name
    name = f"cca_{comps}"
    # Apply CCA
    half = len(df["all"][0]) // 2
    big = np.vstack(df["all"])
    X, Y = big[:, :half], big[:, half:]
    # Add ridge regularization by augmenting the data
    if reg > 0:
        n_features_x = X.shape[1]
        n_features_y = Y.shape[1]
        max_features = max(n_features_x, n_features_y)
        # Create regularization matrices with matching dimensions
        X_reg_part = np.sqrt(reg) * np.eye(max_features)[:, :n_features_x]
        Y_reg_part = np.sqrt(reg) * np.eye(max_features)[:, :n_features_y]
        # Stack to create augmented data
        X_reg = np.vstack([X, X_reg_part])
        Y_reg = np.vstack([Y, Y_reg_part])
        # Fit CCA on regularized data
        cca = CCA(n_components=comps).fit(X_reg, Y_reg)
        # Transform only original data (not regularization rows)
        result = cca.transform(X, Y)[0]
    else:
        # No regularization
        cca = CCA(n_components=comps).fit(X, Y)
        result = cca.transform(X, Y)[0]
    df[name] = list(result.astype(np.float32))
    # Log and return
    print(f"- Applied CCA-{comps} and generated dimensions {comps}!")
    return df, name
