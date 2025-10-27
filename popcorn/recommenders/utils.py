import numpy as np
import pandas as pd
from cornac.data import ImageModality, FeatureModality

SUPPORTED_MODALITIES = ["audio_mmtf", "visual_mmtf", "text_rag_plus", "visual_popcorn"]

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
    return ImageModality(
        features=np.vstack(df[col]), ids=df.item_id, normalized=True
    )


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
    return FeatureModality(
        features=np.vstack(df[col]), ids=df.item_id, normalized=True
    )
