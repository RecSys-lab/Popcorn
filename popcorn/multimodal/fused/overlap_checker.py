#!/usr/bin/env python3

import os
import pandas as pd
from popcorn.utils import loadDataFromCSV, loadJsonFromUrl

def runVisualTextualDatasetsOverlapChecker(cfgRecSys: dict, cfgDatasets: dict):
    """
    Check the overlap between the SceneSense and the LLM-enriched dataset for recommendation

    Parameters
    ----------
    cfgRecSys :dict
        The configurations dictionary for the recommendation system
    cfgDatasets :dict
        The configurations dictionary for the datasets
    """
    # (A) Read the LLM-enriched dataset
    textCSVFilePath = os.path.normpath(cfgRecSys['textual']['llm_enriched_file_path'])
    print(f"I. Reading the enriched text dataset CSV file from '{textCSVFilePath}' ...")
    # Load the CSV data
    enrichedLLMDataset = loadDataFromCSV(textCSVFilePath)
    if enrichedLLMDataset is None:
        return
    print(f"- Loaded {len(enrichedLLMDataset)} records from the LLM-enriched dataset!")
    # print(enrichedLLMDataset.head())
    # Filter only the 'itemId' and 'title' columns
    enrichedLLMDataset = enrichedLLMDataset[['itemId', 'title']]
    enrichedLLMDataset.rename(columns={'title': 'title_llm'}, inplace=True)
    print(f"- Filtered the dataset to contain only 'itemId' and 'title' columns! Check the first 3 records:")
    print(enrichedLLMDataset.head(3))
    # (2) Read the SceneSense dataset
    scenesenseDatasetMetadataUrl = cfgDatasets['multimodal_dataset']['scenesense']['path_metadata']
    print(f"\nII. Fetching the SceneSense meta-data file from '{scenesenseDatasetMetadataUrl}' ...")
    jsonData = loadJsonFromUrl(scenesenseDatasetMetadataUrl)
    if jsonData is None:
        return
    # Load the data into a DataFrame
    sceneSenseMovies = pd.DataFrame(jsonData)
    print(f"- Loaded {len(sceneSenseMovies)} records from the SceneSense metadata!")
    # print(sceneSenseMovies.head())
    # Normalize the 'id' column
    sceneSenseMovies['id'] = sceneSenseMovies['id'].astype(int)
    # Filter only the 'id' and 'title' columns
    sceneSenseMovies = sceneSenseMovies[['id', 'title']]
    sceneSenseMovies.rename(columns={'id': 'itemId'}, inplace=True)
    sceneSenseMovies.rename(columns={'title': 'title_scn'}, inplace=True)
    print(f"- Filtered the dataset to contain only 'itemId' and 'title' columns! Check the first 3 records:")
    print(sceneSenseMovies.head(3))
    # (3) Read the MMTF-14K dataset
    mmtfDatasetRootUrl = cfgDatasets['multimodal_dataset']['mmtf']['download_path']
    # Join the path
    mmtfCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Metadata', 'YearOfProd.csv')
    mmtfCSVFilePath = os.path.normpath(mmtfCSVFilePath)
    print(f"\nIII. Reading the MMTF-14K dataset metadata file from '{mmtfCSVFilePath}' ...")
    # Load the CSV data
    MMTFDataset = loadDataFromCSV(mmtfCSVFilePath)
    if MMTFDataset is None:
        return
    print(f"- Loaded {len(MMTFDataset)} records from the MMTF dataset!")
    # Filter only the 'movieId' column
    MMTFDataset = MMTFDataset[['movieId']]
    MMTFDataset.rename(columns={'movieId': 'itemId'}, inplace=True)
    print(f"- Filtered the MMTF dataset to contain only 'itemId' column! Check the first 3 records:")
    print(MMTFDataset.head(3))
    # (4) Merging the datasets
    print(f"\nIV. Merging the datasets to check overlapped items ...")
    # I. Merge the LLM-enriched dataset with the SceneSense dataset
    mergedTextSceneSenseDataFrame = pd.merge(enrichedLLMDataset, sceneSenseMovies, on='itemId', how='inner')
    print(f"- Merging Textual and SceneSense datasets based on the 'itemId' resulted in '{len(mergedTextSceneSenseDataFrame)}' items! Check the first 3 records:")
    print(mergedTextSceneSenseDataFrame.head(3))
    # II. Merge the LLM-enriched dataset with the MMTF dataset
    mergedTextMMTFDataFrame = pd.merge(enrichedLLMDataset, MMTFDataset, on='itemId', how='inner')
    print(f"- Merging Textual and MMTF datasets based on the 'itemId' resulted in '{len(mergedTextMMTFDataFrame)}' items! Check the first 3 records:")
    print(mergedTextMMTFDataFrame.head(3))
    # III. Merge the SceneSense dataset with the MMTF dataset
    mergedSceneSenseMMTFDataFrame = pd.merge(sceneSenseMovies, MMTFDataset, on='itemId', how='inner')
    print(f"- Merging SceneSense and MMTF datasets based on the 'itemId' resulted in '{len(mergedSceneSenseMMTFDataFrame)}' items! Check the first 3 records:")
    print(mergedSceneSenseMMTFDataFrame.head(3))
    # IV. Merge all the datasets
    mergedAllDataFrame = pd.merge(mergedTextSceneSenseDataFrame, MMTFDataset, on='itemId', how='inner')
    print(f"- Merging all the datasets based on the 'itemId' resulted in '{len(mergedAllDataFrame)}' items! Check the first 3 records:")
    print(mergedAllDataFrame.head(3))