#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from movifex.utils import loadDataFromCSV
from movifex.datasets.movifex.helper_visualfeats_agg import generatedAggFeatureAddresses, loadAggregatedFeaturesIntoDataFrame

def fuseTextualWithMMTFVisual(cfgRecSys: dict, cfgDatasets: dict):
    """
    Fuse the textual data with the MMTF-14K dataset for recommendation, generating the fused dataset as pandas DataFrame

    Parameters
    ----------
    cfgRecSys :dict
        The configurations dictionary for the recommendation system
    cfgDatasets :dict
        The configurations dictionary for the datasets
    """
    # Variables
    outputDir = os.path.normpath(cfgRecSys['fused']['output_dir'])
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
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
    enrichedLLMDataset = enrichedLLMDataset[['itemId', 'title', 'genres']]
    print(f"- Filtered the dataset to contain only 'itemId' and 'title' columns! Check the first 3 records:")
    print(enrichedLLMDataset.head(3))
    # (2) Read the MMTF-14K dataset
    mmtfDatasetRootUrl = cfgDatasets['visual_dataset']['mmtf']['download_path']
    # Join the paths
    mmtfAlexNetVisualAvgCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'AlexNet features', 'Avg', 'AlexNetFeatures - AVG - fc7.csv')
    mmtfAlexNetVisualMedCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'AlexNet features', 'Med', 'AlexNetFeatures - MED - fc7.csv')
    mmtfAlexNetVisualAvgVarCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'AlexNet features', 'AvgVar', 'AlexNetFeatures - AVGVAR - fc7.csv')
    mmtfAlexNetVisualMedMadCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'AlexNet features', 'MedMad', 'AlexNetFeatures - MEDMAD - fc7.csv')
    mmtfAVFVisualAvgCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'Aesthetic features', 'Avg', 'AestheticFeatures - AVG - All.csv')
    mmtfAVFVisualMedCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Visual', 'Aesthetic features', 'Med', 'AestheticFeatures - MED - All.csv')
    # Round#1: Load the 'Avg' dataset
    print(f"\nII-A. Reading the MMTF-14K AlexNet Visual Features (Avg) from '{mmtfAlexNetVisualAvgCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAlexNetVisualAvgCSVFilePath, 'deep')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and AlexNet Visual (Avg) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_avg.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#2: Load the 'Med' dataset
    print(f"\nII-B. Reading the MMTF-14K AlexNet Visual Features (Med) from '{mmtfAlexNetVisualMedCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAlexNetVisualMedCSVFilePath, 'deep')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and AlexNet Visual (Med) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_med.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#3: Load the 'AvgVar' dataset
    print(f"\nII-C. Reading the MMTF-14K AlexNet Visual Features (AvgVar) from '{mmtfAlexNetVisualAvgVarCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAlexNetVisualAvgVarCSVFilePath, 'deep')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and AlexNet Visual (AvgVar) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_avgvar.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#4: Load the 'MedMad' dataset
    print(f"\nII-D. Reading the MMTF-14K AlexNet Visual Features (MedMad) from '{mmtfAlexNetVisualMedMadCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAlexNetVisualMedMadCSVFilePath, 'deep')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and AlexNet Visual (MedMad) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_medmad.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#5: Load the 'Avg' dataset
    print(f"\nII-D. Reading the MMTF-14K Aesthetic Visual Features (Avg) from '{mmtfAVFVisualAvgCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAVFVisualAvgCSVFilePath, 'avf')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and Aesthetic Visual (Avg) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_avf_avg.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#6: Load the 'Med' dataset
    print(f"\nII-D. Reading the MMTF-14K Aesthetic Visual Features (Med) from '{mmtfAVFVisualMedCSVFilePath}' ...")
    tmpVisualDataFrame = loadVisualFeaturesCSVIntoDataFrame(mmtfAVFVisualMedCSVFilePath, 'avf')
    if tmpVisualDataFrame is None:
        return
    # Merging the textual and visual data
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDataFrame, on='itemId', how='inner')
    print(f"- Merging Textual and Aesthetic Visual (Med) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_mmtf_avf_med.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    print("\nFusion completed successfully!")

def fuseTextualWithSceneSense(cfgRecSys: dict, cfgDatasets: dict):
    """
    Fuse the textual data with the SceneSense dataset (aggregated) for recommendation, generating the fused dataset as pandas DataFrame

    Parameters
    ----------
    cfgRecSys :dict
        The configurations dictionary for the recommendation system
    cfgDatasets :dict
        The configurations dictionary for the datasets
    """
    # Variables
    outputDir = os.path.normpath(cfgRecSys['fused']['output_dir'])
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
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
    enrichedLLMDataset = enrichedLLMDataset[['itemId', 'title', 'genres']]
    print(f"- Filtered the dataset to contain only 'itemId' and 'title' columns! Check the first 3 records:")
    print(enrichedLLMDataset.head(3))
    # (2) Read the SceneSense (aggregated) dataset
    aggFeatureAddresses = generatedAggFeatureAddresses(cfgDatasets['visual_dataset']['movifex'])
    # Round#1: Load the 'full_movies_agg' dataset, 'incp3' model
    print(f"\nII-A. Reading the Aggregated SceneSense data for 'Full-Movies', 'Inception-3' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['full_movies_agg']['incp3'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (full-movie, inception3, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_fmovie_incp3_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (full-movie, inception3, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_fmovie_incp3_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#2: Load the 'full_movies_agg' dataset, 'vgg19' model
    print(f"\nII-B. Reading the Aggregated SceneSense data for 'Full-Movies', 'VGG-19' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['full_movies_agg']['vgg19'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (full-movie, vgg19, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_fmovie_vgg19_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (full-movie, vgg19, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_fmovie_vgg19_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#3: Load the 'movie_shots_agg' dataset, 'incp3' model
    print(f"\nII-C. Reading the Aggregated SceneSense data for 'Movie-Shots', 'Inception-3' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['movie_shots_agg']['incp3'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-shots, inception3, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mshots_incp3_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-shots, inception3, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mshots_incp3_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#4: Load the 'movie_shots_agg' dataset, 'vgg19' model
    print(f"\nII-D. Reading the Aggregated SceneSense data for 'Movie-Shots', 'VGG-19' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['movie_shots_agg']['vgg19'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-shots, vgg19, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mshots_vgg19_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-shots, vgg19, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mshots_vgg19_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#5: Load the 'movie_trailers_agg' dataset, 'incp3' model
    print(f"\nII-E. Reading the Aggregated SceneSense data for 'Movie-Trailers', 'Inception-3' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['movie_trailers_agg']['incp3'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-trailers, inception3, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mtrailer_incp3_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-trailers, inception3, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mtrailer_incp3_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Round#6: Load the 'movie_trailers_agg' dataset, 'vgg19' model
    print(f"\nII-F. Reading the Aggregated SceneSense data for 'Movie-Trailers', 'VGG-19' Model ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['movie_trailers_agg']['vgg19'])
    if tmpVisualDFMax is None or tmpVisualDFMean is None:
        return
    # Inform the user
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(tmpVisualDFMax.head(3))
    # Merging the textual and visual data (Max)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMax, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-trailers, vgg19, max) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mtrailer_vgg19_max.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    # Merging the textual and visual data (Mean)
    fusedDataset = pd.merge(enrichedLLMDataset, tmpVisualDFMean, on='itemId', how='inner')
    print(f"- Merging Textual and Visual (movie-trailers, vgg19, mean) data based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
    print(fusedDataset.head(3))
    # Save the fused dataset to a CSV file
    outputFile = os.path.join(outputDir, 'fused_llm_scenesense_mtrailer_vgg19_mean.csv')
    outputFile = os.path.normpath(outputFile)
    print(f"- Saving the fused dataset to '{outputFile}' ...")
    fusedDataset.to_csv(outputFile, index=False)
    print("\nFusion completed successfully!")

def loadVisualFeaturesCSVIntoDataFrame(givenCSVFilePath: str, character: str):
    """
    Load the visual features CSV file into a pandas DataFrame

    Parameters
    ----------
    givenCSVFilePath :str
        The path to the visual features CSV file
    character: str
        The character to be used as the column names
    """
    # Normalize the path
    givenCSVFilePath = os.path.normpath(givenCSVFilePath)
    # Load the CSV data
    MMTFDataset = loadDataFromCSV(givenCSVFilePath)
    if MMTFDataset is None:
        return
    # Prepare the data frame
    MMTFDataset.rename(columns={'movieId': 'itemId'}, inplace=True)
    # Convert deep feature columns to a single embedding column
    embeddingCols = [col for col in MMTFDataset.columns if col.startswith(character)]
    MMTFDataset['embedding'] = MMTFDataset[embeddingCols].apply(lambda row: ','.join(map(str, row)), axis=1)
    # Drop the original deep feature columns
    MMTFDataset = MMTFDataset[['itemId', 'embedding']]
    print(f"- Loaded {len(MMTFDataset)} records into the memory! Check the first 3 records:")
    print(MMTFDataset.head(3))
    return MMTFDataset