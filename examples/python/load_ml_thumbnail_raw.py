#!/usr/bin/env python3

import pandas as pd
from popcorn.utils import readConfigs
from popcorn.datasets.ml_thumbnail.helper_raw_frame import (
    indexThumbnails,
    loadMovieThumbnail,
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
    # Load thumbnails
    print("\n----------- Thumbnail Indexing -----------")
    root = configs["datasets"]["unimodal"]["ml_thumbnail"]["download_path"]
    thumbnailsDF = indexThumbnails(root)
    # Load a specific movie thumbnail
    print("\n----------- Thumbnail Loading -----------")
    movieId = 5
    print(f"Loading thumbnail for movie ID '{movieId}'...")
    thumbnail = loadMovieThumbnail(movieId, configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
