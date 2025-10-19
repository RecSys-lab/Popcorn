import os
import pandas as pd
from popcorn.utils import serializeListColumn
from popcorn.datasets.poison_rag_plus.utils import SUPPORTED_LLMS
from popcorn.datasets.mmtf14k.utils import SUPPORTED_VIS_VARIANTS
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus

def fuseTextualVisual_PoisonRag_MMTF14K(config: dict):
    """
    Fuse 'Poison-RAG-Plus' textual data with 'MMTF-14K' visual features for recommendation

    Parameters
    ----------
    config: dict
        The configuration dictionary

    Returns
    -------
    fusedDataFrameDict: dict
        A dictionary containing the fused pandas DataFrames
    """
    # Variables
    mmtfVisualDict = {}
    poisonRagTextDict = {}
    fusedDataFrameDict = {}
    # Create output directory to save fused data files
    outputPath = os.path.normpath(config["modalities"]["output_path"])
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
        print(f"- Outputs will be saved in '{outputPath}' ...")
    # Step-1: Load Poison-RAG-Plus textual data
    for augmented in [True, False]:
        for llm in SUPPORTED_LLMS:
            textAug = "enriched" if augmented else "raw"
            print(
                f"- Loading 'Poison-RAG-Plus' textual data for LLM variant '{llm}' (augmented={augmented}) ..."
            )
            config["datasets"]["unimodal"]["poison_rag_plus"]["llm"] = llm
            config["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = augmented
            poisonRagTextDF = loadPoisonRagPlus(config)
            if poisonRagTextDF is not None:
                # Apply anti-truncation to text embeddings
                poisonRagTextDF["text"] = serializeListColumn(poisonRagTextDF, "text")
                poisonRagTextDict[f"{llm}_{textAug}"] = poisonRagTextDF
            else:
                print(f"- [Warn] Failed to load Textual data '{llm}_{textAug}'!")
    # Step-2: Load MMTF-14K visual features
    for visVariant in SUPPORTED_VIS_VARIANTS:
        print(f"- Loading 'MMTF-14K' visual features variant '{visVariant}' ...")
        config["datasets"]["multimodal"]["mmtf"]["visual_variant"] = visVariant
        mmtfVisualDF = loadVisualFusedDF(config)
        if mmtfVisualDF is not None:
            mmtfVisualDict[visVariant] = mmtfVisualDF
        else:
            print(f"- [Warn] Failed to load visual variant '{visVariant}'!")
    # Step-3: Fuse textual and visual data
    for textKey, textDF in poisonRagTextDict.items():
        for visualKey, visualDF in mmtfVisualDict.items():
            print(f"- Fusing '{textKey}' with '{visualKey}' ...")
            fusedDF = pd.merge(textDF, visualDF, on="item_id", how="inner")
            if fusedDF is not None:
                # Save the fused DataFrame to CSV
                outputFilePath = os.path.join(
                    outputPath, f"fused_poisonrag_{textKey}_mmtf_visual_{visualKey}.csv"
                )
                fusedDF.to_csv(outputFilePath, index=False)
                print(
                    f"- Fused data with '{len(fusedDF)}' records saved to '{outputFilePath}'!"
                )
                fusedDataFrameDict[f"{textKey}_{visualKey}"] = fusedDF
            else:
                print(f"- [Warn] Fusion failed for '{textKey}' with '{visualKey}'!")
    # Return the fused DataFrame dictionary
    return fusedDataFrameDict