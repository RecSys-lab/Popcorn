#!/usr/bin/env python3

import pandas as pd
from popcorn.utils import loadJsonFromUrl
from popcorn.datasets.popcorn.utils import METADATA_URL
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus


def checkOverlap_PoisonRag_Popcorn(config: dict):
    """
    Check the overlapped items between 'Poison-RAG-Plus' and 'Popcorn' datasets.

    Parameters
    ----------
    config: dict
        The configurations dictionary

    Returns
    -------
    overlappedDF: pd.DataFrame
        The DataFrame containing the overlapped items between the two datasets.
    """
    # Variables
    popcornCfg = config["datasets"]["multimodal"]["popcorn"]
    poisonRagCfg = config["datasets"]["unimodal"]["poison_rag_plus"]
    poisonRagDatasetName, popcornDatasetName = poisonRagCfg["name"], popcornCfg["name"]
    print(
        f"- Checking overlap between '{poisonRagDatasetName}' and '{popcornDatasetName}' datasets ..."
    )
    # Load Poison-RAG-Plus dataset
    print(f"- Loading '{poisonRagDatasetName}' dataset ...")
    poisonRagDF = loadPoisonRagPlus(config)
    if poisonRagDF is None:
        print(f"- [Error] Failed to load '{poisonRagDatasetName}' dataset! Exiting ...")
        return
    # Load Popcorn dataset
    print(f"\n- Loading '{popcornDatasetName}' dataset ...")
    jsonData = loadJsonFromUrl(METADATA_URL)
    if jsonData is None:
        print(
            f"- [Error] Failed to load '{popcornDatasetName}' dataset metadata! Exiting ..."
        )
        return
    popcornDF = pd.DataFrame(jsonData)
    # Remove unused columns from both datasets
    print(f"\n- Preprocessing and normalizing datasets ...")
    poisonRagDF = poisonRagDF[["item_id"]]
    popcornDF["id"] = popcornDF["id"].astype(int)
    popcornDF = popcornDF[["id"]]
    popcornDF.rename(columns={"id": "item_id"}, inplace=True)
    popcornDF["item_id"] = popcornDF.item_id.astype(str)
    # Merge the datasets based on 'item_id'
    print(f"- Merging the datasets based on 'item_id' to find overlapped items ...")
    overlappedDF = pd.merge(poisonRagDF, popcornDF, on="item_id", how="inner")
    print(f"- Found '{len(overlappedDF)}' overlapped items between the two datasets!")
    return overlappedDF


def checkOverlap_PoisonRag_MMTF14K(config: dict):
    """
    Check the overlapped items between 'Poison-RAG-Plus' and 'MMTF-14K' datasets.

    Parameters
    ----------
    config: dict
        The configurations dictionary

    Returns
    -------
    overlappedDF: pd.DataFrame
        The DataFrame containing the overlapped items between the two datasets.
    """
    # Variables
    mmtfCfg = config["datasets"]["multimodal"]["mmtf"]
    poisonRagCfg = config["datasets"]["unimodal"]["poison_rag_plus"]
    poisonRagDatasetName, mmtfDatasetName = poisonRagCfg["name"], mmtfCfg["name"]
    print(
        f"- Checking overlap between '{poisonRagDatasetName}' and '{mmtfDatasetName}' datasets ..."
    )
    # Load Poison-RAG-Plus dataset
    print(f"- Loading '{poisonRagDatasetName}' dataset ...")
    poisonRagDF = loadPoisonRagPlus(config)
    if poisonRagDF is None:
        print(f"- [Error] Failed to load '{poisonRagDatasetName}' dataset! Exiting ...")
        return
    # Load MMTF dataset
    print(f"\n- Loading '{mmtfDatasetName}' dataset ...")
    mmtfDF = loadVisualFusedDF(config)
    if mmtfDF is None:
        print(f"- [Error] Failed to load '{mmtfDatasetName}' dataset! Exiting ...")
        return
    # Remove unused columns from both datasets
    print(f"\n- Preprocessing and normalizing datasets ...")
    poisonRagDF = poisonRagDF[["item_id"]]
    mmtfDF = mmtfDF[["item_id"]]
    # Merge the datasets based on 'item_id'
    print(f"- Merging the datasets based on 'item_id' to find overlapped items ...")
    overlappedDF = pd.merge(poisonRagDF, mmtfDF, on="item_id", how="inner")
    print(f"- Found '{len(overlappedDF)}' overlapped items between the two datasets!")
    return overlappedDF


def checkOverlap_Popcorn_MMTF14K(config: dict):
    """
    Check the overlapped items between 'Popcorn' and 'MMTF-14K' datasets.

    Parameters
    ----------
    config: dict
        The configurations dictionary

    Returns
    -------
    overlappedDF: pd.DataFrame
        The DataFrame containing the overlapped items between the two datasets.
    """
    # Variables
    mmtfCfg = config["datasets"]["multimodal"]["mmtf"]
    popcornCfg = config["datasets"]["multimodal"]["popcorn"]
    popcornDatasetName, mmtfDatasetName = popcornCfg["name"], mmtfCfg["name"]
    print(
        f"- Checking overlap between '{popcornDatasetName}' and '{mmtfDatasetName}' datasets ..."
    )
    # Load Popcorn dataset
    print(f"- Loading '{popcornDatasetName}' dataset ...")
    jsonData = loadJsonFromUrl(METADATA_URL)
    if jsonData is None:
        print(
            f"- [Error] Failed to load '{popcornDatasetName}' dataset metadata! Exiting ..."
        )
        return
    popcornDF = pd.DataFrame(jsonData)
    # Load MMTF dataset
    print(f"\n- Loading '{mmtfDatasetName}' dataset ...")
    mmtfDF = loadVisualFusedDF(config)
    if mmtfDF is None:
        print(f"- [Error] Failed to load '{mmtfDatasetName}' dataset! Exiting ...")
        return
    # Remove unused columns from both datasets
    print(f"\n- Preprocessing and normalizing datasets ...")
    popcornDF["id"] = popcornDF["id"].astype(int)
    popcornDF = popcornDF[["id"]]
    popcornDF.rename(columns={"id": "item_id"}, inplace=True)
    popcornDF["item_id"] = popcornDF.item_id.astype(str)
    mmtfDF = mmtfDF[["item_id"]]
    # Merge the datasets based on 'item_id'
    print(f"- Merging the datasets based on 'item_id' to find overlapped items ...")
    overlappedDF = pd.merge(popcornDF, mmtfDF, on="item_id", how="inner")
    print(f"- Found '{len(overlappedDF)}' overlapped items between the two datasets!")
    return overlappedDF


def checkOverlap_All(config: dict):
    """
    Check the overlapped items between 'Poison-RAG-Plus', 'Popcorn', and 'MMTF-14K' datasets.

    Parameters
    ----------
    config: dict
        The configurations dictionary

    Returns
    -------
    overlappedDF: pd.DataFrame
        The DataFrame containing the overlapped items between the three datasets.
    """
    # First check overlap between Poison-RAG-Plus and Popcorn
    overlappedDF1 = checkOverlap_PoisonRag_Popcorn(config)
    if overlappedDF1 is None or len(overlappedDF1) == 0:
        print(
            "- No overlap found between 'Poison-RAG-Plus' and 'Popcorn' datasets! Continuing..."
        )
        return pd.DataFrame()
    # Then check overlap with MMTF-14K
    mmtfDF = loadVisualFusedDF(config)
    if mmtfDF is None:
        print(f"- [Error] Failed to load 'MMTF-14K' dataset! Exiting ...")
        return pd.DataFrame()
    # Merge the previous overlappedDF1 with mmtfDF
    print(f"- Merging the previous overlapped items with 'MMTF-14K' dataset ...")
    overlappedDF = pd.merge(overlappedDF1, mmtfDF, on="item_id", how="inner")
    print(f"- Found '{len(overlappedDF)}' overlapped items among the three datasets!")
    return overlappedDF
