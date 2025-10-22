import pandas as pd
from typing import List
from popcorn.utils import parseSafe
from popcorn.datasets.poison_rag_plus.utils import PRP_ML_URL, SUPPORTED_LLMS


def loadPoisonRagPlus(config: dict) -> pd.DataFrame:
    """
    Load and prepare the Poison-RAG-Plus dataset based on the provided configuration.
    This function loads text embeddings from multiple CSV files, which can be either
    `original` or `augmented` embeddings. The embeddings are combined and deduplicated
    items are removed before being returned.

    Parameters
    ----------
    config: dict
        The configuration dictionary containing experiment and dataset settings.

    Returns
    -------
    itemsTextDF: pd.DataFrame
        The DataFrame containing item (movie) data and text embeddings.

    Notes
    -----
    The function attempts to load parts sequentially until either:
        1. The maximum number of parts is reached.
        2. A part file is not found (indicating end of available parts).
    """
    # Variables
    parse = parseSafe
    itemsTextDF = pd.DataFrame()
    dataFrames: List[pd.DataFrame] = []
    LLM = config["datasets"]["unimodal"]["poison_rag_plus"]["llm"]
    DATASET_NAME = config["datasets"]["unimodal"]["poison_rag_plus"]["name"]
    AUGMENTED = config["datasets"]["unimodal"]["poison_rag_plus"]["augmented"]
    MAX_PARTS = config["datasets"]["unimodal"]["poison_rag_plus"]["max_parts"]
    # Info
    tag = "enriched" if AUGMENTED else "original"
    print(
        f"\n- Preparing the '{DATASET_NAME}' dataset with '{LLM}'-driven {tag} embeddings ..."
    )
    # Some checks
    if LLM not in SUPPORTED_LLMS:
        print(f"- [Error] Unsupported LLM backbone '{LLM}'! Exiting ...")
        return
    if MAX_PARTS <= 0 or MAX_PARTS >= 30:
        print(f"- [Warn] Test ratio should be in (0, 30)! Setting to 15 ...")
        MAX_PARTS = 15
    # Determine base URL and prefix based on configuration
    fileNameAugmented: str = f"{LLM}_enriched_description_part"
    fileNameOriginal: str = f"{LLM}_originalraw_combined_all_part"
    fileName: str = fileNameAugmented if AUGMENTED else fileNameOriginal
    # Load and process parts sequentially
    for item in range(1, MAX_PARTS + 1):
        url: str = f"{PRP_ML_URL}{fileName}{item}.csv.gz"
        print(f"-- Loading data from '{fileName}{item}.csv.gz' ...")
        try:
            # Read and process each part
            dataFrame = pd.read_csv(url, compression="gzip")
            dataFrame["text"] = dataFrame.embeddings.map(parse)
            dataFrame.rename(columns={"itemId": "item_id"}, inplace=True)
            dataFrames.append(dataFrame[["item_id", "text"]])
        except pd.errors.EmptyDataError:
            print(f"- [Warning] Empty data file found at part {item}")
            break
        except Exception as e:
            # Stop loading if any part is not found or other error occurs
            if not dataFrames:
                raise RuntimeError(
                    f"- [Error] Failed to load any text embedding parts: {str(e)}"
                )
            break
    # Ensure at least one part was loaded successfully
    if not dataFrames:
        raise ValueError(
            "- [Error] No text embedding parts were loaded successfully! Exiting ..."
        )
    # Combine all parts and remove duplicates
    itemsTextDF = pd.concat(dataFrames).drop_duplicates("item_id")
    itemsTextDF["item_id"] = itemsTextDF.item_id.astype(str)
    print(
        f"- Finished loading {len(dataFrames)} parts of textual {tag} data with {len(itemsTextDF):,} items!"
    )
    # Return
    return itemsTextDF
