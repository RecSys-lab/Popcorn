import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from cornac.data import ImageModality, FeatureModality

SUPPORTED_MODALITIES = ["audio_mmtf", "visual_mmtf", "text_rag_plus", "visual_popcorn"]

SUPPORTED_FUSION_METHODS = ["concat", "cca", "pca"]

SUPPORTED_TOP_N = [5, 10, 15, 20, 25, 30, 50]


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
    ratio = config["modalities"]["fusion_methods"]["pca_variance"]
    print(f"- Applying PCA with variance ratio '{ratio}' ...")
    # Check ratio validity
    if not (0.0 < ratio < 1.0):
        print(
            f"- [Warn] PCA variance ratio must be between 0 and 1, but given '{ratio}'. Setting to 0.95 ..."
        )
        ratio = 0.95
    # Set the name
    name = f"pca_{int(ratio * 100)}"
    # Apply PCA
    mat = StandardScaler().fit_transform(np.vstack(df["all"]))
    mat = PCA(ratio, random_state=42).fit_transform(mat)
    df[name] = list(mat.astype(np.float32))
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
    comps = config["modalities"]["fusion_methods"]["cca_components"]
    print(f"- Applying CCA with components '{comps}' ...")
    # Check components validity
    if not (isinstance(comps, int) and comps > 0):
        print(
            f"- [Warn] CCA components must be a positive integer, but given '{comps}'. Setting to 40 ..."
        )
        comps = 40
    # Set the name
    name = f"cca_{comps}"
    # Apply CCA
    half = len(df["all"][0]) // 2
    big = np.vstack(df["all"])
    X, Y = big[:, :half], big[:, half:]
    cca = CCA(n_components=comps).fit(X, Y)
    df[name] = list(cca.transform(X, Y)[0].astype(np.float32))
    # Log and return
    print(f"- Applied CCA-{comps} and generated dimensions {comps}!")
    return df, name
