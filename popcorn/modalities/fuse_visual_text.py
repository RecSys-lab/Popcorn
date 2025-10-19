import os
import pandas as pd
from popcorn.utils import serializeListColumn
from popcorn.datasets.poison_rag_plus.utils import SUPPORTED_LLMS
from popcorn.datasets.mmtf14k.utils import SUPPORTED_VIS_VARIANTS
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus
from popcorn.datasets.popcorn.utils import AGG_EMBEDDING_SOURCES, SUPPORTED_CNNS
from popcorn.datasets.popcorn.helper_embedding_agg import (
    loadAggEmbeddings,
    generateAllAggEmbeddingUrls,
)


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


def fuseTextualVisual_PoisonRag_Popcorn(config: dict):
    """
    Fuse 'Poison-RAG-Plus' textual data with 'Popcorn' visual features for recommendation

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
    popcornVisualDict = {}
    poisonRagTextDict = {}
    fusedDataFrameDict = {}
    aggEmbeddingUrlDict = {}
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
    # Step-2: Load Popcorn visual features
    config["datasets"]["multimodal"]["popcorn"]["cnns"] = SUPPORTED_CNNS
    config["datasets"]["multimodal"]["popcorn"][
        "agg_embedding_sources"
    ] = AGG_EMBEDDING_SOURCES
    # Generate all addresses to aggregated features based on the configuration file
    aggEmbeddingUrlDict = generateAllAggEmbeddingUrls(config)
    if not aggEmbeddingUrlDict:
        print(
            f"- [Error] Failed to generate the addresses for '{AGG_EMBEDDING_SOURCES}' and '{SUPPORTED_CNNS}'! Exiting ..."
        )
        return {}
    print(
        f"-- Generated '{list(aggEmbeddingUrlDict.keys())}' variants of aggregated feature addresses!"
    )
    print(
        f"-- Sample addresses for '{AGG_EMBEDDING_SOURCES}' and '{SUPPORTED_CNNS}': {aggEmbeddingUrlDict[AGG_EMBEDDING_SOURCES[0]][SUPPORTED_CNNS[0]][:2]}"
    )
    # Load aggregated features into a DataFrame
    for embeddingSource in AGG_EMBEDDING_SOURCES:
        for cnn in SUPPORTED_CNNS:
            print(
                f"- Loading aggregated features for source '{embeddingSource}' and CNN '{cnn}' ..."
            )
            aggEmbeddingUrlList = aggEmbeddingUrlDict[embeddingSource][cnn]
            dfAggEmbedsMax, dfAggEmbedsMean = loadAggEmbeddings(aggEmbeddingUrlList)
            print(
                f"- Loaded '{len(dfAggEmbedsMax)}' sample records of aggregated features!"
            )
            popcornVisualDict[f"{embeddingSource}_{cnn}_Max"] = dfAggEmbedsMax
            popcornVisualDict[f"{embeddingSource}_{cnn}_Mean"] = dfAggEmbedsMean
    # Step-3: Fuse textual and visual data
    for textKey, textDF in poisonRagTextDict.items():
        for visualKey, visualDF in popcornVisualDict.items():
            print(f"- Fusing '{textKey}' with '{visualKey}' ...")
            fusedDF = pd.merge(textDF, visualDF, on="item_id", how="inner")
            if fusedDF is not None:
                # Save the fused DataFrame to CSV
                outputFilePath = os.path.join(
                    outputPath,
                    f"fused_poisonrag_{textKey}_popcorn_visual_{visualKey}.csv",
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
