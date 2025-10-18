#!/usr/bin/env python3

import os
import pandas as pd
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
    # Step-1: Load Poison-RAG-Plus textual data
    for augmented in [True, False]:
        for llm in SUPPORTED_LLMS:
            textAug = "enriched" if augmented else "raw"
            print(f"- Loading 'Poison-RAG-Plus' textual data for LLM variant '{llm}' (augmented={augmented}) ...")
            config["datasets"]["unimodal"]["poison_rag_plus"]["llm"] = llm
            config["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = augmented
            poisonRagTextDF = loadPoisonRagPlus(config)
            if poisonRagTextDF is not None:
                poisonRagTextDict[f'{llm}_{textAug}'] = poisonRagTextDF
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
    # Create output directory to save fused data files
    outputPath = os.path.normpath(config['modalities']['output_path'])
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
        print(f"- Outputs will be saved in '{outputPath}' ...")
    # Step-3: Fuse textual and audio data
    for textKey, textDF in poisonRagTextDict.items():
        for audioKey, audioDF in mmtfAudioDict.items():
            print(f"- Fusing '{textKey}' with '{audioKey}' ...")
            fusedDF = pd.merge(textDF, audioDF, on='item_id', how='inner')
            if fusedDF is not None:
                # Save the fused DataFrame to CSV
                outputFilePath = os.path.join(outputPath, f'fused_poisonrag_{textKey}_mmtf_audio_{audioKey}.csv')
                fusedDF.to_csv(outputFilePath, index=False)
                print(f"- Fused data with '{len(fusedDF)}' records saved to '{outputFilePath}'!")
                fusedDataFrameDict[f'{textKey}_{audioKey}'] = fusedDF
            else:
                print(f"- [Warn] Fusion failed for '{textKey}' with '{audioKey}'!")
    # Return the fused DataFrame dictionary
    return fusedDataFrameDict

# def fuseTextualWithMMTFAudio(cfgRecSys: dict, cfgDatasets: dict):
#     """
#     Fuse the textual data with the MMTF-14K dataset for recommendation, generating the fused dataset as pandas DataFrame

#     Parameters
#     ----------
#     cfgRecSys :dict
#         The configurations dictionary for the recommendation system
#     cfgDatasets :dict
#         The configurations dictionary for the datasets
#     """
#     # Variables
#     outputDir = os.path.normpath(cfgRecSys['fused']['output_dir'])
#     if not os.path.exists(outputDir):
#         os.makedirs(outputDir)
#     # (A) Read the LLM-enriched dataset
#     textCSVFilePath = os.path.normpath(cfgRecSys['textual']['llm_enriched_file_path'])
#     print(f"I. Reading the enriched text dataset CSV file from '{textCSVFilePath}' ...")
#     # Load the CSV data
#     enrichedLLMDataset = loadDataFromCSV(textCSVFilePath)
#     if enrichedLLMDataset is None:
#         return
#     print(f"- Loaded {len(enrichedLLMDataset)} records from the LLM-enriched dataset!")
#     # print(enrichedLLMDataset.head())
#     # Filter only the 'itemId' and 'title' columns
#     enrichedLLMDataset = enrichedLLMDataset[['itemId', 'title', 'genres']]
#     print(f"- Filtered the dataset to contain only 'itemId' and 'title' columns! Check the first 3 records:")
#     print(enrichedLLMDataset.head(3))
#     # (2) Read the MMTF-14K dataset
#     mmtfDatasetRootUrl = cfgDatasets['multimodal']['mmtf']['download_path']
#     # Join the paths
#     mmtfAudioCorrCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Audio', 'Block level features', 'Component6', 'BLF_CORRELATIONfeat.csv')
#     mmtfAudioDeltaCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Audio', 'Block level features', 'Component6', 'BLF_DELTASPECTRALfeat.csv')
#     mmtfAudioLogCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Audio', 'Block level features', 'Component6', 'BLF_LOGARITHMICFLUCTUATIONfeat.csv')
#     mmtfAudioSpectralCSVFilePath = os.path.join(mmtfDatasetRootUrl, 'Audio', 'Block level features', 'Component6', 'BLF_SPECTRALfeat.csv')
#     mmtfAudioIVecRootPath = os.path.join(mmtfDatasetRootUrl, 'Audio', 'ivector features')
#     # Round#1: Load the 'Correlation' dataset
#     print(f"\nII-A. Reading the MMTF-14K Audio Features (Correlation) from '{mmtfAudioCorrCSVFilePath}' ...")
#     tmpAudioDataFrame = loadAudioFeaturesCSVIntoDataFrame(mmtfAudioCorrCSVFilePath, 'CORRE')
#     if tmpAudioDataFrame is None:
#         return
#     # Merging the textual and audio data
#     fusedDataset = pd.merge(enrichedLLMDataset, tmpAudioDataFrame, on='itemId', how='inner')
#     print(f"- Merging Textual and Audio (Correlation) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
#     print(fusedDataset.head(3))
#     # Save the fused dataset to a CSV file
#     outputFile = os.path.join(outputDir, 'fused_llm_mmtf_audio_correlation.csv')
#     outputFile = os.path.normpath(outputFile)
#     print(f"- Saving the fused dataset to '{outputFile}' ...")
#     fusedDataset.to_csv(outputFile, index=False)
#     # Round#2: Load the 'Delta' dataset
#     print(f"\nII-B. Reading the MMTF-14K Audio Features (Delta) from '{mmtfAudioDeltaCSVFilePath}' ...")
#     tmpAudioDataFrame = loadAudioFeaturesCSVIntoDataFrame(mmtfAudioDeltaCSVFilePath, 'DELTA')
#     if tmpAudioDataFrame is None:
#         return
#     # Merging the textual and audio data
#     fusedDataset = pd.merge(enrichedLLMDataset, tmpAudioDataFrame, on='itemId', how='inner')
#     print(f"- Merging Textual and Audio (Delta) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
#     print(fusedDataset.head(3))
#     # Save the fused dataset to a CSV file
#     outputFile = os.path.join(outputDir, 'fused_llm_mmtf_audio_delta.csv')
#     outputFile = os.path.normpath(outputFile)
#     print(f"- Saving the fused dataset to '{outputFile}' ...")
#     fusedDataset.to_csv(outputFile, index=False)
#     # Round#3: Load the 'Log' dataset
#     print(f"\nII-C. Reading the MMTF-14K Audio Features (Log) from '{mmtfAudioLogCSVFilePath}' ...")
#     tmpAudioDataFrame = loadAudioFeaturesCSVIntoDataFrame(mmtfAudioLogCSVFilePath, 'LOGAR')
#     if tmpAudioDataFrame is None:
#         return
#     # Merging the textual and audio data
#     fusedDataset = pd.merge(enrichedLLMDataset, tmpAudioDataFrame, on='itemId', how='inner')
#     print(f"- Merging Textual and Audio (Log) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
#     print(fusedDataset.head(3))
#     # Save the fused dataset to a CSV file
#     outputFile = os.path.join(outputDir, 'fused_llm_mmtf_audio_log.csv')
#     outputFile = os.path.normpath(outputFile)
#     print(f"- Saving the fused dataset to '{outputFile}' ...")
#     fusedDataset.to_csv(outputFile, index=False)
#     # Round#4: Load the 'Spectral' dataset
#     print(f"\nII-D. Reading the MMTF-14K Audio Features (Spectral) from '{mmtfAudioSpectralCSVFilePath}' ...")
#     tmpAudioDataFrame = loadAudioFeaturesCSVIntoDataFrame(mmtfAudioSpectralCSVFilePath, 'SPECT')
#     if tmpAudioDataFrame is None:
#         return
#     # Merging the textual and audio data
#     fusedDataset = pd.merge(enrichedLLMDataset, tmpAudioDataFrame, on='itemId', how='inner')
#     print(f"- Merging Textual and Audio (Spectral) datasets based on the 'itemId' resulted in '{len(fusedDataset)}' items! Check the first 3 records:")
#     print(fusedDataset.head(3))
#     # Save the fused dataset to a CSV file
#     outputFile = os.path.join(outputDir, 'fused_llm_mmtf_audio_special.csv')
#     outputFile = os.path.normpath(outputFile)
#     print(f"- Saving the fused dataset to '{outputFile}' ...")
#     fusedDataset.to_csv(outputFile, index=False)
#     # Round#5: Load all 'i-vector' dataset
#     print(f"\nII-E. Reading the MMTF-14K Audio Features (i-vetors) from '{mmtfAudioIVecRootPath}' ...")
#     # Loop over all the i-vector CSV files in the directory
#     mmtfiVectorFiles = [f for f in os.listdir(mmtfAudioIVecRootPath) if f.endswith('.csv')]
#     for ivectorFile in mmtfiVectorFiles:
#         # Get the file path
#         ivectorFilePath = os.path.join(mmtfAudioIVecRootPath, ivectorFile)
#         # Load the i-vector CSV file into a DataFrame
#         tmpAudioDataFrame = loadAudioFeaturesCSVIntoDataFrame(ivectorFilePath, 'ivec')
#         if tmpAudioDataFrame is None:
#             continue
#         # Merging the textual and audio data
#         fusedDataset = pd.merge(enrichedLLMDataset, tmpAudioDataFrame, on='itemId', how='inner')
#         print(f"- Merging Textual and Audio (i-vector {ivectorFile}) based on the 'itemId' resulted in '{len(fusedDataset)}' items!")
#         # Save the fused dataset to a CSV file
#         outputFile = os.path.join(outputDir, f'fused_llm_mmtf_audio_{ivectorFile}')
#         outputFile = os.path.normpath(outputFile)
#         print(f"- Saving the fused dataset to '{outputFile}' ...")
#         fusedDataset.to_csv(outputFile, index=False)
#     print("\nFusion completed successfully!")

# def loadAudioFeaturesCSVIntoDataFrame(givenCSVFilePath: str, character: str):
#     """
#     Load the visual features CSV file into a pandas DataFrame

#     Parameters
#     ----------
#     givenCSVFilePath: str
#         The path to the visual features CSV file
#     character: str
#         The character to be used as the column names
#     """
#     # Normalize the path
#     givenCSVFilePath = os.path.normpath(givenCSVFilePath)
#     # Load the CSV data
#     MMTFDataset = loadDataFromCSV(givenCSVFilePath)
#     if MMTFDataset is None:
#         return
#     # Prepare the data frame
#     MMTFDataset.rename(columns={'movieId': 'itemId'}, inplace=True)
#     # Convert deep feature columns to a single embedding column
#     embeddingCols = [col for col in MMTFDataset.columns if col.startswith(character)]
#     MMTFDataset['embedding'] = MMTFDataset[embeddingCols].apply(lambda row: ','.join(map(str, row)), axis=1)
#     # Drop the original deep feature columns
#     MMTFDataset = MMTFDataset[['itemId', 'embedding']]
#     print(f"- Loaded {len(MMTFDataset)} records into the memory! Check the first 3 records:")
#     print(MMTFDataset.head(3))
#     return MMTFDataset
