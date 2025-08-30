import pandas as pd

def standardizeMMTF14kDataFrame(dataFrame: pd.DataFrame):
    """
    Unify the given DataFrame loaded from the MMTF14k dataset (e.g., average, median, etc.) for recommendation tasks.

    Parameters
    ----------
    dataFrame: pd.DataFrame
        Given dataset in the form of a DataFrame.

    Returns
    -------
    dataFrame: pd.DataFrame
        Modified DataFrame.
    """
    # Drop ignored columns
    dataFrame = dataFrame.drop(columns=['title', 'genres'], errors='ignore')
    # Rename the columns
    dataFrame = dataFrame.rename(columns={'embedding': 'embeddings'})
    # Change the data types
    dataFrame['embeddings'] = dataFrame['embeddings'].astype(str).str.replace(',', ' ')
    # Return the modified DataFrame
    return dataFrame