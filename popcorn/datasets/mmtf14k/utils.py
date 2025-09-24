import pandas as pd

# URLs for accessing Popcorn dataset
BASE_URL = "https://drive.google.com/drive/folders/1sBD8drB2H0WHl_MSsSCH-FA-bonjStr_?usp=sharing"

# Base URL for MMTF-14K audio features fused with textual features
AUD_FUSED_URL = (
    "https://raw.githubusercontent.com/RecSys-lab/"
    "reproducibility_data/refs/heads/main/fused_textual_audio/"
)

# Base URL for MMTF-14K visual features fused with textual features
VIS_FUSED_BASE = (
    "https://raw.githubusercontent.com/RecSys-lab/"
    "reproducibility_data/refs/heads/main/fused_textual_visual/"
)

# Mapping of MMTF-14K audio feature variants to their corresponding file names
AUD_FUSED_FILE_MAP = {
    "log": "fused_llm_mmtf_audio_log.csv",
    "delta": "fused_llm_mmtf_audio_delta.csv",
    "spect": "fused_llm_mmtf_audio_spectral.csv",
    "corr": "fused_llm_mmtf_audio_correlation.csv",
    "ivec": "i-vector/fused_llm_mmtf_audio_IVec_splitItem_fold_1_gmm_128_tvDim_20.csv",
}

# Mapping of MMTF-14K visual feature variants to their corresponding file names
VIS_FUSED_FILE_MAP = {
    "cnn": "fused_llm_mmtf_avg.csv",
    "avf": "fused_llm_mmtf_avf_avg.csv",
}

# Supported audio feature variants
SUPPORTED_AUD_VARIANTS = ["ivec", "blf"]
AUD_BLF_VARIANTS = ["corr", "delta", "log", "spect"]

# Data Frames column names for MMTF fused features
fusedAudCols = ["itemId", "title", "genres", "embedding"]


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
    # Change the data types
    dataFrame["embedding"] = dataFrame["embedding"].astype(str).str.replace(",", " ")
    # Return the modified DataFrame
    return dataFrame
