import os
from popcorn.datasets.utils import applyKcore
from popcorn.utils import convertStrToListCol
from popcorn.modalities.fuse_all import createMultimodalDF
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.mmtf14k.helper_audio import loadAudioFusedDF
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus
from popcorn.datasets.movielens.process import applyKeepList, trainTestSplit
from popcorn.recommenders.utils import SUPPORTED_MODALITIES, SUPPORTED_FUSION_METHODS
from popcorn.datasets.popcorn.helper_embedding_agg import (
    loadAggEmbeddings,
    generateAllAggEmbeddingUrls,
)
from popcorn.recommenders.utils import (
    applyPCAModality,
    applyCCAModality,
    getImageModality,
    getFeatureModality,
)


def assembleModality(config: dict):
    """
    Assemble the different modalities (audio, visual, text) into a unified representation to be used by recommenders.

    Parameters
    ----------
    config: dict
        The configuration dictionary

    Returns
    -------
    trainDF: pd.DataFrame
        The training set DataFrame after applying keep list.
    testDF: pd.DataFrame
        The testing set DataFrame after applying keep list.
    trainSet: cornac.data.Dataset
        The Cornac Dataset object created from the training DataFrame.
    modalitiesDict: dict
        A dictionary containing the individual modality DataFrames.
    """
    # Variables
    modalitiesDict = {}
    textDF, audioDF, visualDF = None, None, None
    selectedModalities = config["modalities"]["selected"]
    fusionVars = config["modalities"]["fusion_methods"]["selected"]
    # Arguments check
    if not selectedModalities or len(selectedModalities) == 0:
        print("- [Error] No modalities selected for assembly! Exiting ...")
        return
    for modality in selectedModalities:
        if modality not in SUPPORTED_MODALITIES:
            print(f"- [Error] Modality '{modality}' is not supported! Exiting ...")
            return
    # Check for incompatible modality combinations
    if "visual_mmtf" in selectedModalities and "visual_popcorn" in selectedModalities:
        print(
            "- [Error] Cannot have both 'visual_mmtf' and 'visual_popcorn' modalities together! Exiting ..."
        )
        return
    # Step#1: Load MovieLens train and test sets
    itemsDF, usersDF, ratingsDF = loadMovieLens(config)
    if ratingsDF is None or itemsDF is None or usersDF is None:
        print("- [Error] Error in loading the MovieLens dataset! Exiting ...")
        return
    # Step#2: Apply k-core filtering (if specified in config)
    K_CORE = config["setup"]["k_core"]
    if K_CORE > 0:
        ratingsDF = applyKcore(ratingsDF, K_CORE)
        print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF):,}")
    # Step#3: Split the data into train and test sets
    trainDF, testDF = trainTestSplit(ratingsDF, config)
    if trainDF is None or testDF is None:
        print("- [Error] Error in splitting the MovieLens dataset! Exiting ...")
        return
    # Step#4: Load selected modalities and assemble them
    if "text_rag_plus" in selectedModalities:
        textDF = loadPoisonRagPlus(config)
    if "audio_mmtf" in selectedModalities:
        audioDF = loadAudioFusedDF(config)
    if "visual_mmtf" in selectedModalities:
        visualDF = loadVisualFusedDF(config)
    elif "visual_popcorn" in selectedModalities:
        aggEmbeddingUrlDict = generateAllAggEmbeddingUrls(config)
        if aggEmbeddingUrlDict:
            # Take all generated addresses
            aggEmbeddingUrlList = []
            for embedding in aggEmbeddingUrlDict:
                for cnn in aggEmbeddingUrlDict[embedding]:
                    aggEmbeddingUrlList.extend(aggEmbeddingUrlDict[embedding][cnn])
            # Load aggregated features into a DataFrame
            dfAggEmbedsMax, dfAggEmbedsMean = loadAggEmbeddings(aggEmbeddingUrlList)
            # Consider the 'Max' variant for assembly
            visualDF = dfAggEmbedsMax
            # Apply anti-truncation to visual embeddings
            visualDF["visual"] = convertStrToListCol(visualDF, "visual")
    # Step#5: Create a modality dictionary and fuse them
    multimodalDict = {
        "text": textDF,
        "audio": audioDF,
        "visual": visualDF,
    }
    fusedDF, keep = createMultimodalDF(multimodalDict)
    if fusedDF is None:
        print("- [Error] Failed to create fused DataFrame! Exiting ...")
        return
    # Step#6: Apply keep on train and test sets
    trainDF, testDF, trainSet = applyKeepList(trainDF, testDF, keep)
    # Step#7: Prepare the assembled dictionary
    print("\n- Assembling (concatenating) modalities ...")
    modalitiesDict["concat"] = {
        "audio_image": getImageModality(fusedDF, "audio"),
        "visual_image": getImageModality(fusedDF, "visual"),
        "text_image": getImageModality(fusedDF, "text"),
        "all_image": getImageModality(fusedDF, "all"),
        "all_feature": getFeatureModality(fusedDF, "all"),
    }
    # Step#8: Apply fusion methods
    for fusionMethod in fusionVars:
        if fusionMethod not in SUPPORTED_FUSION_METHODS:
            print(
                f"- [Warn] Fusion method '{fusionMethod}' is not supported! Ignoring ..."
            )
            continue
        if fusionMethod == "concat":
            continue
        if fusionMethod == "pca":
            # Apply PCA
            pcaDF, name = applyPCAModality(fusedDF, config)
            modalitiesDict[name] = {
                "all_image": getImageModality(pcaDF, name),
                "all_feature": getFeatureModality(pcaDF, name),
            }
        elif fusionMethod == "cca":
            # Apply CCA
            ccaDF, name = applyCCAModality(fusedDF, config)
            modalitiesDict[name] = {
                "all_image": getImageModality(ccaDF, name),
                "all_feature": getFeatureModality(ccaDF, name),
            }
    # Return
    return trainDF, testDF, trainSet, modalitiesDict
