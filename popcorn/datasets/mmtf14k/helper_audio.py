import numpy as np
import pandas as pd
from popcorn.utils import parseSafe
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from popcorn.datasets.mmtf14k.utils import (
    AUD_FUSED_URL,
    AUD_BLF_VARIANTS,
    AUD_FUSED_FILE_MAP,
    SUPPORTED_AUD_VARIANTS,
)


def loadAudioFusedCsv(url: str) -> pd.DataFrame:
    """
    Read and process MMTF audio embeddings (fused with llm-augmented text) from a CSV file into a DataFrame.

    Parameters
    ----------
    url: str
        The URL or file path to the CSV file containing audio embeddings.

    Returns
    -------
    audioFusedDF: pd.DataFrame
        A DataFrame with loaded audio embeddings (columns: item_id, embedding).
    """
    # Variables
    audioFusedDF = pd.DataFrame()
    # Check if the URL is valid
    if not url or not isinstance(url, str):
        print(f"- [Error] Invalid URL: '{url}'! Returning an empty DataFrame ...")
        return audioFusedDF
    # Parse embeddings safely
    parse = parseSafe
    # Read CSV with optimized memory usage
    try:
        # Read the CSV file
        audioFusedDF = pd.read_csv(url, low_memory=False)
        # Clean up DataFrame
        audioFusedDF.drop(columns=["title", "genres"], errors="ignore", inplace=True)
        # Convert embeddings from string to numpy arrays
        audioFusedDF["embedding"] = (
            audioFusedDF["embedding"].astype(str).str.replace(",", " ").apply(parse)
        )
        # Rename 'itemId' column to ensure consistency
        audioFusedDF.rename(columns={"itemId": "item_id"}, inplace=True)
        # Return the processed DataFrame
        return audioFusedDF[["item_id", "embedding"]]
    except pd.errors.EmptyDataError:
        print(
            f"- [Error] The CSV file at '{url}' is empty! Returning an empty DataFrame ..."
        )
        return pd.DataFrame()


def loadAudioFusedDF(config: dict) -> pd.DataFrame:
    """
    Load and process audio embeddings (fused with llm-augmented text) from the MMTF-14K dataset
    based on the specified variant. This function supports two types of audio embeddings:
        1. i-vector: Direct loading of i-vector features
        2. blf (Block-level Features): Combines multiple MMTF features with PCA

    Parameters
    ----------
    config: dict
        Configuration dictionary containing various settings.
    variant: str
        The audio variant to load. Supported values are: 'ivec' for i-vector features
        and 'blf' for Block-level Features.

    Returns
    -------
    dfAudio: pd.DataFrame
        A DataFrame with loaded audio embeddings (columns: item_id, audio).
    """
    # Variables
    dfAudio = pd.DataFrame()
    # Check inputs
    if config is None or config == {}:
        print(
            f"- [Error] No valid configuration provided! Returning an empty DataFrame ..."
        )
        return dfAudio
    # Extract configuration parameters
    seed = config["setup"]["seed"]
    variant = config["datasets"]["multimodal"]["mmtf"]["audio_variant"]
    pcaRatio = config["datasets"]["multimodal"]["mmtf"]["audio_blf_pca"]
    # Check variant
    if variant not in SUPPORTED_AUD_VARIANTS:
        print(
            f"- [Error] Unsupported audio variant '{variant}'! Supported variants are: {SUPPORTED_AUD_VARIANTS}. Returning an empty DataFrame ..."
        )
        return dfAudio
    # Load the audio DataFrame
    print(f"- Fetching MMTF-14K audio data for variant '{variant}' ...")
    # Handle audio variants
    if variant == "ivec":
        try:
            # Load i-vector features directly
            dfAudio = loadAudioFusedCsv(AUD_FUSED_URL + AUD_FUSED_FILE_MAP["ivec"])
            # Rename columns for consistency
            dfAudio.rename(columns={"embedding": "audio"}, inplace=True)
            print(f"- Fetched {len(dfAudio):,} audio items using 'i-vector' features.")
            # Return the processed DataFrame
            return dfAudio
        except Exception as e:
            print(f"- [Error] Failed to load 'i-vector' audio features: {e}")
            return pd.DataFrame()
    # Handle bottleneck layer features (blf) variant
    elif variant == "blf":
        # Load all MMTF block-level features
        blfDataFrameList = []
        # Read each feature and store in the list
        try:
            # Loop through each feature variant
            for key in AUD_BLF_VARIANTS:
                print(f"-- Loading feature 'audio_{key}' ...")
                # Load the feature CSV
                dfBlf = loadAudioFusedCsv(AUD_FUSED_URL + AUD_FUSED_FILE_MAP[key])
                # Rename embedding column to include the feature key
                dfBlf = dfBlf.rename(columns={"embedding": f"audio_{key}"})
                # Append to the list
                blfDataFrameList.append(dfBlf)
            # Merge all embeddings
            print("- Merging all block-level audio embeddings ...")
            mergedDF = blfDataFrameList[0]
            for d in blfDataFrameList[1:]:
                mergedDF = mergedDF.merge(d, on="item_id", how="inner")
            # Concatenate all embeddings
            print("- Concatenating all block-level audio embeddings ...")
            mergedDF["concat"] = mergedDF.apply(
                lambda r: np.concatenate(
                    [
                        r["audio_log"],
                        r["audio_corr"],
                        r["audio_delta"],
                        r["audio_spect"],
                    ]
                ),
                axis=1,
            )
            # Apply PCA for dimensionality reduction
            X = np.vstack(mergedDF["concat"].values)
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=pcaRatio, svd_solver="full", random_state=seed)
            X_pca = pca.fit_transform(X_scaled).astype(np.float32)
            # Create final DataFrame
            dfAudio = pd.DataFrame(
                {"item_id": mergedDF["item_id"], "audio": list(X_pca)}
            )
            # Log information
            print(
                f"- Fetched {len(dfAudio):,} audio items using 'blf' features (PCA {pcaRatio*100}% dims = {X_pca.shape[1]})."
            )
            # Return the processed DataFrame
            return dfAudio
        except Exception as e:
            print(f"- [Error] Failed to load 'blf' audio features: {e}")
            return pd.DataFrame()
