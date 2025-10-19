#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.modalities.fuse_visual_text import (
    fuseTextualVisual_PoisonRag_MMTF14K,
    fuseTextualVisual_PoisonRag_Popcorn
)


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
    # Fuse Poison-RAG-Plus and Popcorn datasets
    print("\n----------- Poison-RAG-Plus & MMTF14K Visual -----------")
    fusedDF = fuseTextualVisual_PoisonRag_MMTF14K(configs)
    if fusedDF is None:
        print("- [Error] Fusion failed!")
    # Fuse Poison-RAG-Plus and Popcorn datasets
    print("\n----------- Poison-RAG-Plus & Popcorn Visual -----------")
    fusedDF = fuseTextualVisual_PoisonRag_Popcorn(configs)
    if fusedDF is None:
        print("- [Error] Fusion failed!")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
