#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.modalities.helper_overlap import (
    checkOverlap_PoisonRag_Popcorn,
    checkOverlap_PoisonRag_MMTF14K,
    checkOverlap_Popcorn_MMTF14K,
    checkOverlap_All
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
    print("\n----------- Overlap Check #1 -----------")
    overlappedDF = checkOverlap_PoisonRag_Popcorn(configs)
    if overlappedDF is None:
        print("- Overlap check failed!")
    # Check overlap between Poison-RAG-Plus and MMTF-14K datasets
    print("\n----------- Overlap Check #2 -----------")
    overlappedDF = checkOverlap_PoisonRag_MMTF14K(configs)
    if overlappedDF is None:
        print("- Overlap check failed!")
    # Check overlap between Popcorn and MMTF-14K datasets
    # print("\n----------- Overlap Check #3 -----------")
    overlappedDF = checkOverlap_Popcorn_MMTF14K(configs)
    if overlappedDF is None:
        print("- Overlap check failed!")
    # Check overlap among Poison-RAG-Plus, Popcorn, and MMTF-14K datasets
    print("\n----------- Overlap Check #4 -----------")
    overlappedDF = checkOverlap_All(configs)
    if overlappedDF is None:
        print("- Overlap check failed!")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
