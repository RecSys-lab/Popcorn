#!/usr/bin/env python3

from popcorn.utils import readConfigs, loadJsonFromUrl
from popcorn.datasets.popcorn.helper_metadata import countMovies


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
    datasetName = configs['datasets']['multimodal']['popcorn']['name']
    datasetMetadataUrl = configs['datasets']['multimodal']['popcorn']['path_metadata']
    print(f"- Loading the '{datasetName}' dataset metadata from '{datasetMetadataUrl}' ...")
    jsonData = loadJsonFromUrl(datasetMetadataUrl)
    if jsonData is None:
        print("- Error in loading the Popcorn dataset metadata! Exiting ...")
        return
    # [Util-1] Count the number of movies in the dataset
    print("\n[Util-1] Counting the number of movies in the dataset ...")
    moviesCount = countMovies(jsonData)
    if moviesCount == -1:
        print("- Error in counting the number of movies!")
    else:
        print(f"- Number of movies in the dataset (from metadata): {moviesCount}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
