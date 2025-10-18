#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.modalities.fuse_audio_text import (
    fuseTextualAudio_PoisonRag_MMTF14K
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
    # Check overlap between Poison-RAG-Plus and Popcorn datasets
    print("\n----------- Poison-RAG-Plus & MMTF14K Audio -----------")
    fusedDF = fuseTextualAudio_PoisonRag_MMTF14K(configs)
    if fusedDF is None:
        print("- [Error] Fusion failed!")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
