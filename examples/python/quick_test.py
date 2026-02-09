#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.optimizers.grid_search import gridSearch
from popcorn.recommenders.reclist import generateLists
from popcorn.recommenders.assembler import assembleModality


def main():
    print("Welcome to 'Popcorn' 🍿! Starting the framework ...\n")
    # Read the configuration file
    configs = readConfigs("popcorn/config/config.yml")
    # If properly read, print the configurations
    if not configs:
        print("- Error reading the configuration file!")
        return
    
    # Step-0: Override some configurations for a quick test
    configs["setup"]["use_gpu"] = False  # Disable GPU
    configs["setup"]["model_choice"] = "vbpr"  # Model choice
    configs["setup"]["use_parallel"] = True  # Enable parallel processing
    configs["setup"]["is_fast_prototype"] = True  # Enable fast prototype mode
    configs["modalities"]["fusion_methods"]["selected"] = ["cca", "pca"]
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"  # Use MovieLens 1M
    configs["modalities"]["selected"] = ["visual_mmtf", "text_rag_plus"]
    
    # Step-1: Data ingestion and modality assembly
    trainDF, testDF, trainSet, modalitiesDict, genreDict = assembleModality(configs)
    if trainDF is None or testDF is None or trainSet is None:
        print("- Error in assembling modalities! Exiting ...")
        return
    print("\n- Modalities assembled successfully!")
    
    # Step-2: Apply grid search to find the best model configurations
    finalModels = gridSearch(configs, trainDF, trainSet, modalitiesDict)
    print("\n- Grid search completed successfully!")
    
    # Step-3: Generate recommendation lists
    generateLists(configs, trainDF, trainSet, testDF, genreDict, finalModels)
    
    # Stop
    print("\n- Stopping 'Popcorn'!")


if __name__ == "__main__":
    main()
