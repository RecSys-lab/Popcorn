#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.popcorn.utils import RAW_DATA_URL
from popcorn.datasets.popcorn.helper_embedding import (
    countMovies
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
    # Load Popcorn Dataset metadata
    datasetName = configs["datasets"]["multimodal"]["popcorn"]["name"]
    datasetMetadataUrl = configs["datasets"]["multimodal"]["popcorn"]["path_metadata"]
    print(
        f"- Loading the '{datasetName}' dataset metadata from '{datasetMetadataUrl}' ..."
    )
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
