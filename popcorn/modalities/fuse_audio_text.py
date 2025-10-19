#!/usr/bin/env python3

import os
import pandas as pd
from popcorn.utils import serializeListColumn
from popcorn.datasets.poison_rag_plus.utils import SUPPORTED_LLMS
from popcorn.datasets.mmtf14k.utils import SUPPORTED_AUD_VARIANTS
from popcorn.datasets.mmtf14k.helper_audio import loadAudioFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus


def fuseTextualAudio_PoisonRag_MMTF14K(config: dict):
    """
    Fuse 'Poison-RAG-Plus' textual data with 'MMTF-14K' audio features for recommendation

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
    mmtfAudioDict = {}
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
    # Step-2: Load MMTF-14K audio features
    for audVariant in SUPPORTED_AUD_VARIANTS:
        print(f"- Loading 'MMTF-14K' audio features variant '{audVariant}' ...")
        config["datasets"]["multimodal"]["mmtf"]["audio_variant"] = audVariant
        mmtfAudioDF = loadAudioFusedDF(config)
        if mmtfAudioDF is not None:
            mmtfAudioDict[audVariant] = mmtfAudioDF
        else:
            print(f"- [Warn] Failed to load audio variant '{audVariant}'!")
    # Step-3: Fuse textual and audio data
    for textKey, textDF in poisonRagTextDict.items():
        for audioKey, audioDF in mmtfAudioDict.items():
            print(f"- Fusing '{textKey}' with '{audioKey}' ...")
            fusedDF = pd.merge(textDF, audioDF, on="item_id", how="inner")
            if fusedDF is not None:
                # Save the fused DataFrame to CSV
                outputFilePath = os.path.join(
                    outputPath, f"fused_poisonrag_{textKey}_mmtf_audio_{audioKey}.csv"
                )
                fusedDF.to_csv(outputFilePath, index=False)
                print(
                    f"- Fused data with '{len(fusedDF)}' records saved to '{outputFilePath}'!"
                )
                fusedDataFrameDict[f"{textKey}_{audioKey}"] = fusedDF
            else:
                print(f"- [Warn] Fusion failed for '{textKey}' with '{audioKey}'!")
    # Return the fused DataFrame dictionary
    return fusedDataFrameDict
