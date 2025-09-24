import pandas as pd
from popcorn.utils import parseSafe
from popcorn.datasets.mmtf14k.utils import (
    VIS_FUSED_BASE,
    VIS_FUSED_FILE_MAP,
    SUPPORTED_VIS_VARIANTS,
)


def loadVisualFusedDF(config: dict) -> pd.DataFrame:
    """
    Load and process visual embeddings (fused) from the MMTF-14K dataset based on the specified variant.
    This function supports two types of visual embeddings:
        1. CNN features: Extracted from a pre-trained Convolutional Neural Network (AlexNet).
        2. AVF features: Extracted from Aesthetic Visual Features (AVF) model.

    Parameters
    ----------
    config: dict
        Configuration dictionary containing various settings.

    Returns
    -------
    dfVisual: pd.DataFrame
        A DataFrame with loaded visual embeddings (columns: item_id, visual).
    """
    # Variables
    dfVisual = pd.DataFrame()
    # Check inputs
    if config is None or config == {}:
        print(
            f"- [Error] No valid configuration provided! Returning an empty DataFrame ..."
        )
        return dfVisual
    # Extract configuration parameters
    parse = parseSafe
    variant = config["datasets"]["multimodal"]["mmtf"]["visual_variant"]
    # Check variant
    if variant not in SUPPORTED_VIS_VARIANTS:
        print(
            f"- [Error] Unsupported visual variant '{variant}'! Supported variants are: {SUPPORTED_VIS_VARIANTS}. Returning an empty DataFrame ..."
        )
        return dfVisual
    # Load the visual DataFrame
    print(f"- Fetching MMTF-14K visual data for variant '{variant}' ...")
    # Handle visual variants
    try:
        # Read the CSV file
        dfVisual = pd.read_csv(VIS_FUSED_BASE + VIS_FUSED_FILE_MAP[variant])
        # Rename 'itemId' column to ensure consistency
        dfVisual.rename(columns={"itemId": "item_id"}, inplace=True)
        # Parse embeddings from string to numpy arrays
        dfVisual["visual"] = dfVisual.embedding.map(parse)
        print(f"- Fetched {len(dfVisual):,} visual items using '{variant}' features.")
        # Return the processed DataFrame
        return dfVisual[["item_id", "visual"]]
    except Exception as e:
        print(f"- [Error] Failed to load visual features: {e}")
        return pd.DataFrame()
