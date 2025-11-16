#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.optimizers.grid_search import gridSearch
from popcorn.recommenders.assembler import assembleModality


def main():
    print(
        "Welcome to 'Popcorn' 🍿! Starting the framework for your movie recommendation ...\n"
    )
    # Read the configuration file
    configs = readConfigs("popcorn/config/config.yml")
    # If properly read, print the configurations
    if not configs:
        print("Error reading the configuration file!")
        return
    # Assemble modalities - MMTF (audio+video)
    print("\n----------- MMTF (audio+video) + Poison-RAG-Plus (text) -----------")
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"  # Use MovieLens 1M
    configs["datasets"]["multimodal"]["mmtf"][
        "audio_variant"
    ] = "ivec"  # Use i-vector audio
    configs["modalities"]["fusion_methods"]["selected"] = ["concat"]
    configs["modalities"]["selected"] = ["audio_mmtf", "visual_mmtf", "text_rag_plus"]
    trainDF, testDF, trainSet, modalitiesDict, genreDict = assembleModality(configs)
    if trainDF is None or testDF is None or trainSet is None:
        print("- Error in assembling modalities! Exiting ...")
        return
    print("\n✔ Modalities assembled successfully!")
    print(f"- Training set size: {trainDF.shape}, Testing set size: {testDF.shape}")
    print(f"- Available modalities: {list(modalitiesDict.keys())}")
    # Apply grid search to find the best model configurations
    configs["setup"]["use_gpu"] = False  # Disable GPU for grid search
    configs["setup"]["model_choice"] = "amr"  # Model choice for grid search
    configs["setup"][
        "use_parallel"
    ] = True  # Enable parallel processing for grid search
    configs["setup"][
        "is_fast_prototype"
    ] = True  # Enable fast prototype mode for grid search
    finalModels = gridSearch(configs, trainDF, trainSet, modalitiesDict)
    print("\n✔ Grid search completed successfully!")
    print(f"- Final models after HPO: {list(finalModels.keys())}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
