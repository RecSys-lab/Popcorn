#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.mmtf14k.downloader import downloadMMTF14k


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
    # Get common configurations
    cfgMovieLens = configs["datasets"]["multimodal"]["mmtf"]
    downloadPath = cfgMovieLens["download_path"]
    # Download MMTF-14K dataset (Base)
    print("\n----------- MMTF-14K (Base) -----------")
    downloadMMTF14k(downloadPath)
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
