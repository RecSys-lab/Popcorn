#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.modalities.fuse_all import createMultimodalDF
from popcorn.datasets.mmtf14k.helper_audio import loadAudioFusedDF
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus



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
    # Modality#1: Textual data from Poison-RAG-Plus
    print("\n----------- Loading Poison-RAG-Plus Textual Data -----------")
    textDF = loadPoisonRagPlus(configs)
    if textDF is not None:
        print(f"\n- textDF shape: {textDF.shape}")
    # Modality#2: Audio and Visual data from MMTF-14K
    print("\n----------- Loading MMTF-14K Audio and Visual Data -----------")
    configs["datasets"]["multimodal"]["mmtf"]["audio_variant"] = "ivec"
    audioDF = loadAudioFusedDF(configs)
    if audioDF is not None:
        print(f"- audioDF shape: {audioDF.shape}")
    visualDF = loadVisualFusedDF(configs)
    if visualDF is not None:
        print(f"- visualDF shape: {visualDF.shape}")
    # Check overlap between Poison-RAG-Plus and Popcorn datasets
    modalitiesDict = {
        "text": textDF,
        "audio": audioDF,
        "visual": visualDF,
    }
    fusedDF, keep = createMultimodalDF(modalitiesDict)
    if fusedDF is None:
        print("\n- [Error] Failed to create fused DataFrame!")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
