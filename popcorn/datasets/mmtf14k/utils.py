import pandas as pd

# URLs for accessing Popcorn dataset
BASE_URL = "https://drive.google.com/drive/folders/1sBD8drB2H0WHl_MSsSCH-FA-bonjStr_?usp=sharing"


def normalizeMMTF14kDataFrame(dataFrame: pd.DataFrame):
    """
    Unify the given DataFrame loaded from the MMTF14k dataset for processing.

    Parameters
    ----------
    dataFrame: pd.DataFrame
        Given dataset in the form of a DataFrame.

    Returns
    -------
    dataFrame: pd.DataFrame
        Modified DataFrame.
    """
    # Check if DataFrame is valid
    if dataFrame is None or dataFrame.empty:
        return pd.DataFrame()
    # Drop ignored columns
    dataFrame = dataFrame.drop(columns=["title", "genres"], errors="ignore")
    # Rename the columns
    dataFrame = dataFrame.rename(columns={"embedding": "embeddings"})
    # Change the data types
    dataFrame["embeddings"] = dataFrame["embeddings"].astype(str).str.replace(",", " ")
    # Return the modified DataFrame
    return dataFrame
