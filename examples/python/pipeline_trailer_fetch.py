#!/usr/bin/env python3

from popcorn.utils import readConfigs, loadJsonFromUrl
from popcorn.datasets.popcorn.utils import METADATA_URL
from popcorn.datasets.popcorn.helper_metadata import (
    fetchRandomMovies
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
    # Step-1 - Load Popcorn Dataset metadata
    datasetName = configs["datasets"]["multimodal"]["popcorn"]["name"]
    print(
        f"\n- Loading the '{datasetName}' dataset metadata from '{METADATA_URL}' ..."
    )
    jsonData = loadJsonFromUrl(METADATA_URL)
    if jsonData is None:
        print("- Error in loading the Popcorn dataset metadata! Exiting ...")
        return
    # Step-2 - Pick a sample number of random movies from the dataset
    print(f"\n- Picking 5 random movies from the dataset ...")
    pickedMovies = fetchRandomMovies(jsonData, count=5)
    if not pickedMovies:
        print("- Error in fetching random movies!")
        return
    # Drop the 'genres' field for better visualization
    for i in range(len(pickedMovies)):
        pickedMovies[i] = {
            'id': pickedMovies[i]['id'],
            'year': pickedMovies[i]['year'],
            'title': pickedMovies[i]['title']
            }
        print(f"- Picked Movie-{i+1}: {pickedMovies[i]}")
    # # Load configurations
    # pipelineName = configs["pipelines"]["trailer_fetch"]["name"]
    # trailerDownloadPath = configs["pipelines"]["trailer_fetch"]["download_path"]
    # # Strart the pipeline
    # print(f"Starting the '{pipelineName}' pipeline ...\n")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
