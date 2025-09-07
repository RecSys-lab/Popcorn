#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.movielens.helper_ratings import filterRatingsByUserInteraction


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
    # Load MovieLens dataset - 1m version
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if ratingsDF is None:
        print("Error in loading the MovieLens dataset! Exiting ...")
        return
    print(f"\n- ratingsDF (shape: {ratingsDF.shape}): \n{ratingsDF.head()}\n")
    # [Util-1] Apply interaction limits
    ratingsDF_filtered = filterRatingsByUserInteraction(ratingsDF, 15, 30)
    print(f"- Filtered ratingsDF: \n{ratingsDF_filtered.head(3)}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
