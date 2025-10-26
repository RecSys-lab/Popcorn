import numpy as np
import pandas as pd

def createMultimodalDF(unimodalDict: dict):
    """
    Fuses multiple unimodal DataFrames into a single multimodal DataFrame based on common item IDs.
    It supports text, audio, and visual modalities, each with (item_id, text/audio/visual) columns.

    Parameters
    ----------
    unimodalDict: dict
        A dictionary containing unimodal pandas DataFrames

    Returns
    -------
    fusedDF: dict
        A pandas DataFrame containing the fused multimodal data
    keep: set
        A set of item IDs that are present in the fused DataFrame
    """
    # Variables
    fusedDF = None
    keep, common = set(), set()
    modalities = list(unimodalDict.keys())
    supportedModalities = ["text", "audio", "visual"]
    print("- Creating multimodal DataFrame from unimodal DataFrames ...")
    # Check to avoid empty dictionary
    if not unimodalDict:
        print("- [Warn] No unimodal dataframes provided for fusion! Returning empty dictionary ...")
        return {}
    # Check for supported modalities
    for key in unimodalDict.keys():
        if key not in supportedModalities:
            print(f"- [Warn] Unsupported modality '{key}' found! Supported modalities are {supportedModalities}. Returning empty dictionary ...")
            return {}
    # Check to see which modalities are missing
    for modality in supportedModalities:
        if modality not in modalities:
            print(f"- [Warn] Modality '{modality}' not found in input dictionary. Proceeding with available modalities ...")
    # Normalize item ID columns (if necessary) to 'item_id'
    for key, df in unimodalDict.items():
        if 'itemId' in df.columns:
            print(f"- Renaming 'itemId' column to 'item_id' in '{key}' dataframe ...")
            df = df.rename(columns={'itemId': 'item_id'})
            unimodalDict[key] = df
    # Find common items across all modalities
    for key, df in unimodalDict.items():
        common = common & set(df.item_id) if common else set(df.item_id)
    print(f"- Found {len(common):,} common items across modalities ...")
    # Filter each DataFrame to keep only common items
    for key in unimodalDict.keys():
        df = unimodalDict[key]
        df = df[df.item_id.isin(common)].reset_index(drop=True)
        unimodalDict[key] = df
    # Merge DataFrames on 'item_id'
    for key, df in unimodalDict.items():
        if fusedDF is None:
            fusedDF = df
        else:
            fusedDF = pd.merge(fusedDF, df, on="item_id", how="inner")
    print(f"- Created a Fused DataFrame ({fusedDF.shape}) with modalities: {list(unimodalDict.keys())} ...\n{fusedDF.head(3)}")
    # Guard against NaN/Inf
    for col in modalities:
        fusedDF[col] = fusedDF[col].apply(
            lambda v: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        )
    # Combine all available modalities into a single 'all' column
    fusedDF["all"] = fusedDF.apply(
        lambda r: np.hstack([r[mod] for mod in modalities]), axis=1
    )
    # Kept items
    keep = set(fusedDF.item_id)
    print(f"- Final fused DataFrame has {len(keep):,} items after combining all modalities ...")
    # Return the fused DataFrame
    return fusedDF, keep