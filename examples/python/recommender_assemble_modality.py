#!/usr/bin/env python3

from popcorn.utils import readConfigs
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
    # Sample#1: Assemble modalities - MMTF (audio+video)
    print("\n----------- MMTF (audio+video) + Poison-RAG-Plus (text) -----------")
    configs["modalities"]['selected'] = ["audio_mmtf", "visual_mmtf", "text_rag_plus"]
    trainDF, testDF, trainSet, modalitiesDict = assembleModality(configs)
    if trainDF is None or testDF is None or trainSet is None:
        print("- Error in assembling modalities! Exiting ...")
        return
    print("\n✔ Modalities assembled successfully!")
    print(f"- Training set size: {trainDF.shape}, Testing set size: {testDF.shape}")
    print(f"- Available modalities: {list(modalitiesDict.keys())}")
    # Sample#2: Assemble modalities - Popcorn (visual) + MMTF (audio) + Poison-RAG-Plus (text)
    print("\n----------- Popcorn (visual) + MMTF (audio) + Poison-RAG-Plus (text) -----------")
    configs["modalities"]['selected'] = ["audio_mmtf", "visual_popcorn", "text_rag_plus"]
    trainDF, testDF, trainSet, modalitiesDict = assembleModality(configs)
    if trainDF is None or testDF is None or trainSet is None:
        print("- Error in assembling modalities! Exiting ...")
        return
    print("\n✔ Modalities assembled successfully!")
    print(f"- Training set size: {trainDF.shape}, Testing set size: {testDF.shape}")
    print(f"- Available modalities: {list(modalitiesDict.keys())}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
