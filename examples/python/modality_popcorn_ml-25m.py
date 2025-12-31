#!/usr/bin/env python3

from popcorn.datasets.utils import applyKcore
from popcorn.utils import readConfigs, loadJsonFromUrl
from popcorn.datasets.popcorn.utils import METADATA_URL
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.utils import printTextualDatasetStats
from popcorn.datasets.popcorn.helper_metadata import fetchAllMovieIds


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
    print(f"- Loading the '{datasetName}' dataset metadata from '{METADATA_URL}' ...")
    jsonData = loadJsonFromUrl(METADATA_URL)
    if jsonData is None:
        print("- Error in loading the Popcorn dataset metadata! Exiting ...")
        return
    # Fetch all movie IDs in the dataset
    print("- Fetching all movie IDs in the dataset ...")
    movieIds = fetchAllMovieIds(jsonData)
    if not movieIds:
        print("- Error in fetching movie IDs!")
    else:
        print(
            f"- {len(movieIds)} movie IDs have been fetched successfully. Sample IDs: {movieIds[:5]}"
        )
    # Load MovieLens dataset 25M version
    configs["datasets"]["unimodal"]["movielens"]["version"] = "25m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if ratingsDF is not None:
        print(f"\n- RatingsDF original row count: {len(ratingsDF):,}")
    printTextualDatasetStats(ratingsDF)
    # Now, merge with Popcorn dataset to keep only relevant movies
    movieIds = [int(mid) for mid in movieIds]
    print(f"\n- Merging MovieLens ratings with Popcorn movie IDs to keep relevant movies ...")
    print(ratingsDF.head())
    print(movieIds[:5])
    ratingsDF = ratingsDF[ratingsDF["item_id"].isin(movieIds)]
    print(f"- After merging, row count: {len(ratingsDF):,}")
    printTextualDatasetStats(ratingsDF)
    # Now, apply k-core filtering and see the difference
    K_CORE = configs["setup"]["k_core"]
    ratingsDF_filtered = applyKcore(ratingsDF, K_CORE)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    printTextualDatasetStats(ratingsDF_filtered)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
