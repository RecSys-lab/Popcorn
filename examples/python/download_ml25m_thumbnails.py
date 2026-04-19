#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.ml_thumbnail.downloader import downloadMovieLensThumbnailImages


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
    downloadPath = configs["datasets"]["unimodal"]["ml_thumbnail"]["download_path"]
    # Download MovieLens-25M thumbnails
    downloadMovieLensThumbnailImages(1, downloadPath)
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
