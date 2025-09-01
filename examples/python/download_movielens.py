#!/usr/bin/env python3

import os
from popcorn.utils import readConfigs
from popcorn.datasets.movielens.downloader import downloadMovieLens

def main():
    print("Welcome to 'Popcorn' 🍿! Starting the framework for your movie recommendation ...\n")
    # Read the configuration file
    configs = readConfigs("popcorn/config/config.yml")
    # If properly read, print the configurations
    if not configs:
        print("Error reading the configuration file!")
        return
    # Get common configurations
    cfgMovieLens = configs['datasets']['unimodal_dataset']['movielens']
    downloadPath = cfgMovieLens["download_path"]
    # Download MovieLens dataset - 100k version
    isDownloadSuccessful = downloadMovieLens("100k", os.path.join(downloadPath, "ml-100k"))
    # Download MovieLens dataset - 1m version
    isDownloadSuccessful = downloadMovieLens("1m", os.path.join(downloadPath, "ml-1m"))
    # Download MovieLens dataset - 25m version
    isDownloadSuccessful = downloadMovieLens("25m", os.path.join(downloadPath, "ml-25m"))
    print("\nStopping 'Popcorn'!")

if __name__ == "__main__":
    main()